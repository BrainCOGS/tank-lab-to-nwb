"""Read a Kilosort or SpikeInterface sorter folder into spike samples, labels and metrics, with what sorted it.

Spike trains are split from `spike_times.npy` by `spike_clusters.npy` with numpy, not with SpikeInterface's Kilosort
extractor (0.105.4). That extractor takes its unit list from the folder's tsv files, joined with an inner merge, and drops
the spikes of any cluster missing from the result without a word: a phy folder whose `cluster_group.tsv` has only its
header (job 1053 of S3) would lose clusters unless `cluster_info.tsv` is present, and a stale `cluster_info.tsv` decides the units on its
own. `tests/test_ephys_sorting.py` shows the two readers give identical trains wherever the extractor keeps every spike.
Labels and metrics are read file by file with pandas, so Kilosort's `KSLabel`, phy's curated `group` and any stale
`cluster_info.tsv` stay apart instead of being merged into one `quality`.

Never read here: `temp_wh.dat`, `pc_features.npy`, `_phy_spikes_subset.waveforms.npy`. Their sizes are read with stat only.
"""

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ("spike_times.npy", "params.py", "channel_map.npy", "channel_positions.npy", "templates.npy")
LABELS = ("good", "mua", "noise")
MATLAB_BATCH_SAMPLES = 64 * 1024 + 64
# Kilosort's own per-cluster files: column name in the file, column in the units table, and its description.
KILOSORT_TABLES = {
    "cluster_KSLabel.tsv": ("KSLabel", "ks_label", "Kilosort's automatic label: good (single unit by its refractory-period test) or mua."),
    "cluster_Amplitude.tsv": ("Amplitude", "ks_amplitude", "Kilosort's template amplitude (cluster_Amplitude.tsv), in Kilosort's whitened units, not microvolts."),
    "cluster_ContamPct.tsv": ("ContamPct", "ks_contam_pct", "Kilosort's estimated contamination, in percent (cluster_ContamPct.tsv)."),
}
CLUSTER_INFO_COLUMNS = {
    "amp": "phy's amplitude of the cluster's mean waveform on its best channel (cluster_info.tsv amp), in raw units.",
    "ch": "phy's best channel (cluster_info.tsv ch), an index into the sorted recording's channels.",
    "depth": "phy's depth of the cluster (cluster_info.tsv depth), micrometers from the probe tip along the shank.",
    "fr": "phy's firing rate (cluster_info.tsv fr), spikes per second over the recording.",
    "n_spikes": "phy's spike count (cluster_info.tsv n_spikes).",
    "sh": "phy's shank (cluster_info.tsv sh).",
}


class SortingError(ValueError):
    """A sorter folder is missing something, or holds a layout or sorter this converter doesn't support."""


@dataclass(frozen=True)
class Column:
    name: str
    description: str
    data: np.ndarray


@dataclass(frozen=True)
class SortedProbe:
    """One probe's sorting, still in samples of the file the sorter read."""

    folder: Path
    sorter: str  # e.g. "Kilosort 4"
    sorter_version: str  # "4.0.37", or "unknown" when the files don't say
    sorter_evidence: str  # which files said so
    sorted_input: str  # the binary the sorter read, as its files name it, or "unknown"
    sample_origin: str  # what sample 0 of spike_times.npy is
    n_sorted_samples: tuple | None  # (fewest, most) samples the sorter read, from temp_wh.dat or ops.npy, or None if nothing says
    sample_rate_params: float  # params.py sample_rate, usually the nominal 30000; times use the meta's calibrated rate
    unit_ids: np.ndarray  # cluster ids, as in spike_clusters.npy
    spike_samples: list  # per unit, int64 sample indices
    peak_channel: np.ndarray  # per unit, the sorter channel (index into channel_map.npy) of its main template's largest peak-to-peak
    channel_map: np.ndarray
    channel_positions: np.ndarray
    columns: list  # of Column, one value per unit
    files: dict = field(default_factory=dict)  # name -> what was read from it, for the record
    notes: list = field(default_factory=list)
    n_spikes_read: int | None = None  # kept when spike_samples is let go after mapping

    @property
    def n_spikes(self):
        return self.n_spikes_read if self.n_spikes_read is not None else int(sum(train.size for train in self.spike_samples))


def _read_params(folder):
    text = (folder / "params.py").read_text()
    params = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                params[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                params[node.targets[0].id] = ast.unparse(node.value)
    return params


def check_folder(folder):
    """Raise SortingError naming the folder and what's missing, unless it looks like a Kilosort/phy folder."""
    folder = Path(folder)
    if not folder.is_dir():
        raise SortingError(f"{folder}: not a folder")
    missing = [name for name in REQUIRED if not (folder / name).is_file()]
    if not (folder / "spike_clusters.npy").is_file() and not (folder / "spike_templates.npy").is_file():
        missing.append("spike_clusters.npy or spike_templates.npy")
    if missing:
        hint = ""
        if (folder / "sorter_output").is_dir():
            hint = " It holds a sorter_output folder: SpikeInterface's run_sorter layout, before the lab's pipeline moved the phy files up. Point at sorter_output."
        elif (folder / "spikeinterface").is_dir() and not (folder / "spike_times.npy").is_file():
            hint = " It holds only SpikeInterface's own files; the sorter left no phy folder here."
        raise SortingError(f"{folder}: not a Kilosort/phy folder this converter can read; missing {', '.join(missing)}.{hint}")


def find_sorter_folders(job_folder):
    """The sorter output folders of a job folder: `kilosort_output`, `kilosort<N>_output`, or any `<method>_output` with phy files.

    U19's `_find_kilosort_output` globs only `kilosort*_output`, so it misses a SpikeInterface sorter that isn't Kilosort.
    """
    job_folder = Path(job_folder)
    found = [path for path in sorted(job_folder.glob("*_output")) if path.is_dir() and (path / "spike_times.npy").is_file()]
    if not found:
        raise SortingError(f"{job_folder}: no *_output folder with spike_times.npy")
    return found


_KS_VERSION = re.compile(r"Kilosort version (\S+)")
_KS_INPUT = re.compile(r"Sorting \[PosixPath\('([^']+)'\)")


def identify_sorter(folder):
    """(sorter, version, evidence, sorted input) from the files alone."""
    folder = Path(folder)
    provenance = folder / "sorter_provenance.json"
    si_recording = folder / "spikeinterface" / "spikeinterface_recording.json"
    sorted_input = "unknown"
    if si_recording.is_file():
        sorted_input = _si_recording_source(si_recording)
    if provenance.is_file():
        record = json.loads(provenance.read_text())
        name = record.get("sorter", "unknown")
        version = record.get("installed_version") or (record.get("sorter_version", "").partition("@")[2]) or "unknown"
        sorter = {"kilosort2": "Kilosort 2", "kilosort2_5": "Kilosort 2.5", "kilosort3": "Kilosort 3", "kilosort4": "Kilosort 4"}.get(name, name)
        via = f"SpikeInterface {record['spikeinterface']}" if "spikeinterface" in record else "SpikeInterface"
        mode = f", {record['mode']}" if "mode" in record else ""
        evidence = f"sorter_provenance.json: {record.get('sorter_version', name)} through {via}{mode}"
        if (folder / "kilosort4.log").is_file():
            sorted_input = _ks4_input(folder) or sorted_input
        return sorter, version, evidence, sorted_input
    if (folder / "kilosort4.log").is_file():
        text = (folder / "kilosort4.log").read_text(errors="replace")
        match = _KS_VERSION.search(text)
        return "Kilosort 4", match[1] if match else "unknown", "kilosort4.log" + (" names the version" if match else " has no version line"), _ks4_input(folder) or sorted_input
    if (folder / "ops.npy").is_file() and not (folder / "kilosort_overview.fig").is_file():
        return "Kilosort 4", "unknown", "ops.npy without kilosort4.log", sorted_input
    features = (folder / "pc_features.npy").is_file() or (folder / "template_features.npy").is_file()
    if (folder / "kilosort_overview.fig").is_file() and not features:
        return (
            "Kilosort 3",
            "unknown",
            "kilosort_overview.fig is written by the lab's kilosortbatch.m, which runs Kilosort 3 (datashift2, then rezToPhy2); rezToPhy2 writes no pc_features.npy or template_features.npy, and there are none",
            sorted_input,
        )
    if (folder / "template_features.npy").is_file() and (folder / "pc_features.npy").is_file():
        return "Kilosort 2 or 2.5", "unknown", "template_features.npy and pc_features.npy, which only MATLAB rezToPhy (Kilosort 2/2.5) writes", sorted_input
    raise SortingError(f"{folder}: can't tell which sorter wrote this folder: no sorter_provenance.json, kilosort4.log, ops.npy or kilosort_overview.fig")


def _ks4_input(folder):
    match = _KS_INPUT.search((folder / "kilosort4.log").read_text(errors="replace"))
    return match[1] if match else None


def _si_recording_source(path):
    """The first file path in SpikeInterface's serialized recording, which names the binary it read."""
    text = path.read_text(errors="replace")
    match = re.search(r'"(?:file_paths?|folder_path)"\s*:\s*\[?\s*"([^"]+)"', text)
    return match[1] if match else "unknown"


def sample_origin(sorted_input):
    """What sample 0 of spike_times.npy is, from the name of the binary the sorter read."""
    name = Path(sorted_input).name
    if sorted_input == "unknown":
        return "unknown: no file names the binary the sorter read"
    if "_tcat." in name:
        return "sample 0 of CatGT's tcat AP file, which starts at sample 0 of the t0 AP file when the sample counts match"
    if re.search(r"_t\d+\.imec\d+\.ap\.bin$", name):
        return "sample 0 of the raw t0 AP file"
    if "dredge" in sorted_input.lower() or "spikeinterface_output" in sorted_input:
        return "sample 0 of a preprocessed copy of the AP file, which keeps its sample count"
    return f"sample 0 of {name}"


def n_sorted_samples(folder, params, sorter):
    """Samples the sorter read, as (fewest, most): exact from temp_wh.dat's size (stat only), within one batch from a
    Kilosort 4 ops.npy (Nbatches x batch_size), or None if nothing says."""
    folder = Path(folder)
    wh = folder / "temp_wh.dat"
    if wh.is_file():
        channels = int(params.get("n_channels_dat", 0)) or None
        # KS4 writes temp_wh.dat with every saved channel; MATLAB Kilosort with only the mapped ones.
        for count in filter(None, (channels, np.load(folder / "channel_map.npy").size)):
            if wh.stat().st_size % (2 * count) == 0:
                n = wh.stat().st_size // (2 * count)
                if sorter != "Kilosort 4":
                    # MATLAB Kilosort 2/3 pads its last batch with zeros, so temp_wh.dat holds whole batches of
                    # ops.NT samples, which no file in the folder records: Kilosort's default, 64 x 1024 + 64.
                    return (n - MATLAB_BATCH_SAMPLES + 1, n)
                return (n, n)
    ops = folder / "ops.npy"
    if sorter == "Kilosort 4" and ops.is_file():
        record = np.load(ops, allow_pickle=True).item()
        if "Nbatches" in record and "batch_size" in record:
            batches, size = int(record["Nbatches"]), int(record["batch_size"])
            return ((batches - 1) * size + 1, batches * size)
    return None


def _cluster_table(folder, name, column):
    path = folder / name
    if not path.is_file():
        return None
    table = pd.read_csv(path, sep="\t")
    if "cluster_id" not in table or column not in table:
        raise SortingError(f"{path}: expected columns cluster_id and {column}, found {list(table.columns)}")
    if table["cluster_id"].duplicated().any():
        raise SortingError(f"{path}: a cluster is listed twice")
    return table.set_index("cluster_id")[column]


def _per_unit(series, unit_ids, fill):
    return np.array([series.get(int(unit), fill) for unit in unit_ids])


def _labels(values, name, path):
    bad = sorted(set(map(str, values)) - set(LABELS) - {"", "unsorted"})
    if bad:
        raise SortingError(f"{path}: {name} holds {bad}, not good/mua/noise")
    return values


def _peak_channels(folder, unit_ids, spike_clusters, spike_templates, channel_map):
    """Per cluster, the sorter channel (index into channel_map.npy) of the largest peak-to-peak of its most frequent template."""
    templates = np.load(folder / "templates.npy", mmap_mode="r")
    peak_to_peak = np.ptp(np.asarray(templates), axis=1)  # template x sorted channel
    order = np.argsort(spike_clusters, kind="stable")
    clusters, starts = np.unique(spike_clusters[order], return_index=True)
    bounds = dict(zip(clusters.tolist(), zip(starts, np.append(starts[1:], order.size))))
    out = np.empty(len(unit_ids), dtype=np.int64)
    for i, unit in enumerate(unit_ids):
        start, stop = bounds[int(unit)]
        main = np.bincount(spike_templates[order[start:stop]]).argmax()
        out[i] = int(np.argmax(peak_to_peak[main]))
    return out


def split_trains(spike_times, spike_clusters):
    """(sorted cluster ids, per cluster its spike samples as int64), keeping every spike."""
    spike_times = np.asarray(spike_times).ravel().astype(np.int64)
    spike_clusters = np.asarray(spike_clusters).ravel().astype(np.int64)
    order = np.argsort(spike_clusters, kind="stable")
    unit_ids, starts = np.unique(spike_clusters[order], return_index=True)
    stops = np.append(starts[1:], order.size)
    return unit_ids, [spike_times[order[start:stop]] for start, stop in zip(starts, stops)]


def read_sorting(folder):
    """Read one sorter output folder. Raises SortingError naming the folder and the problem."""
    folder = Path(folder)
    check_folder(folder)
    sorter, version, evidence, sorted_input = identify_sorter(folder)
    params = _read_params(folder)
    spike_times = np.load(folder / "spike_times.npy").ravel()
    clusters_file = "spike_clusters.npy" if (folder / "spike_clusters.npy").is_file() else "spike_templates.npy"
    spike_clusters = np.load(folder / clusters_file).ravel().astype(np.int64)
    spike_templates = np.load(folder / "spike_templates.npy").ravel().astype(np.int64) if (folder / "spike_templates.npy").is_file() else spike_clusters
    if not (spike_times.size == spike_clusters.size == spike_templates.size):
        raise SortingError(f"{folder}: spike_times.npy has {spike_times.size} spikes but {clusters_file} has {spike_clusters.size}")
    if np.any(np.diff(spike_times.astype(np.int64)) < 0):
        raise SortingError(f"{folder}: spike_times.npy isn't sorted")
    unit_ids, trains = split_trains(spike_times, spike_clusters)
    channel_map = np.load(folder / "channel_map.npy").ravel().astype(np.int64)
    channel_positions = np.load(folder / "channel_positions.npy").astype("float64")
    if channel_positions.shape != (channel_map.size, 2):
        raise SortingError(f"{folder}: channel_positions.npy is {channel_positions.shape} for {channel_map.size} mapped channels")
    columns, files, notes = [], {}, []
    for name, (source, column, description) in KILOSORT_TABLES.items():
        table = _cluster_table(folder, name, source)
        if table is None:
            continue
        missing = [int(u) for u in unit_ids if int(u) not in table.index]
        if missing:
            notes.append(f"{name} has no row for {len(missing)} clusters (merged or split in phy, which doesn't rerun Kilosort); they get an empty value")
        if source == "KSLabel":
            values = _labels(_per_unit(table.astype(str), unit_ids, ""), source, folder / name)
            columns.append(Column(column, description, values.astype(str)))
        else:
            columns.append(Column(column, description + " NaN for a cluster Kilosort didn't make.", _per_unit(table.astype("float64"), unit_ids, np.nan).astype("float64")))
        files[name] = f"{len(table)} rows"
    curated = _curation(folder, unit_ids, spike_clusters, files, notes)
    columns.extend(curated)
    quality = _quality(columns)
    columns.insert(0, quality)
    columns.extend(_metrics(folder, unit_ids, files, notes))
    peak = _peak_channels(folder, unit_ids, spike_clusters, spike_templates, channel_map)
    return SortedProbe(
        folder=folder,
        sorter=sorter,
        sorter_version=version,
        sorter_evidence=evidence,
        sorted_input=sorted_input,
        sample_origin=sample_origin(sorted_input),
        n_sorted_samples=n_sorted_samples(folder, params, sorter),
        sample_rate_params=float(params.get("sample_rate", np.nan)),
        unit_ids=unit_ids,
        spike_samples=trains,
        peak_channel=peak,
        channel_map=channel_map,
        channel_positions=channel_positions,
        columns=columns,
        files=files,
        notes=notes,
    )


def _curation(folder, unit_ids, spike_clusters, files, notes):
    """phy's curated label, from cluster_group.tsv, and cluster_info.tsv's columns, if they belong to this sorting."""
    columns = []
    group = _cluster_table(folder, "cluster_group.tsv", "group")
    info_path = folder / "cluster_info.tsv"
    info = pd.read_csv(info_path, sep="\t") if info_path.is_file() else None
    if info is not None:
        counts = pd.Series(np.bincount(spike_clusters - spike_clusters.min())[np.asarray(unit_ids) - spike_clusters.min()], index=unit_ids)
        listed = info.set_index("cluster_id")
        same_units = set(listed.index) == set(int(u) for u in unit_ids)
        same_counts = same_units and "n_spikes" in listed and np.array_equal(listed.loc[list(unit_ids), "n_spikes"].to_numpy(), counts.to_numpy())
        if not same_counts:
            raise SortingError(f"{info_path}: lists {len(listed)} clusters whose spike counts don't match spike_clusters.npy ({len(unit_ids)} clusters), so it's from another sorting or another curation. Remove or move it to convert this folder.")
        files["cluster_info.tsv"] = f"{len(listed)} rows, spike counts match"
        for name, description in CLUSTER_INFO_COLUMNS.items():
            if name in listed:
                columns.append(Column(f"phy_{name}", description, listed.loc[list(unit_ids), name].to_numpy()))
        if "group" in listed:
            info_group = listed["group"].fillna("").astype(str)
            if group is not None:
                file_group = group.astype(str)
                differ = [int(c) for c in file_group.index if c in info_group.index and file_group[c] != info_group[c]]
                if differ:
                    raise SortingError(f"{folder}: cluster_group.tsv and cluster_info.tsv disagree on the group of clusters {differ[:5]}")
            group = info_group
    if group is not None:
        values = np.array(["" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in _per_unit(group, unit_ids, "")])
        _labels(values, "group", folder / "cluster_group.tsv")
        files.setdefault("cluster_group.tsv", f"{len(group)} rows")
        columns.append(Column("curated_group", "phy's group label (cluster_group.tsv, or cluster_info.tsv's group): good, mua or noise as set by a person in phy, or Kilosort's label copied there if nobody changed it. Empty where neither file gives one.", values))
    return columns


def _quality(columns):
    """One label per unit: the curated group where there is one, else Kilosort's label."""
    by_name = {column.name: column.data for column in columns}
    ks = by_name.get("ks_label")
    curated = by_name.get("curated_group")
    n = len(ks if ks is not None else curated if curated is not None else [])
    quality = np.full(n, "", dtype=object)
    if ks is not None:
        quality[:] = ks
    if curated is not None:
        quality = np.where(curated != "", curated, quality)
    quality = np.where(quality == "unsorted", "", quality)
    return Column("quality", "good, mua or noise: phy's curated group where one is set, else Kilosort's KSLabel. Empty where neither says.", np.asarray(quality, dtype=str))


def _metrics(folder, unit_ids, files, notes):
    """Per-unit metrics beside the phy files: SpikeInterface's quality and template metrics, and the IBL step's cluster_metrics.csv."""
    columns = []
    sources = [
        (folder / "spikeinterface" / "quality_metrics.csv", "unit_id", "si_", "SpikeInterface quality metric {name} (spikeinterface/quality_metrics.csv)."),
        (folder / "spikeinterface" / "template_metrics.csv", "unit_id", "si_", "SpikeInterface template metric {name} (spikeinterface/template_metrics.csv)."),
        (folder / "metrics.csv", "cluster_id", "ecephys_", "Metric {name} from metrics.csv in the sorter folder."),
        (folder.parent / "ibl_data" / "cluster_metrics.csv", "cluster_id", "ibl_", "IBL quality metric {name} from ibl_data/cluster_metrics.csv, which the lab's IBL step computes from this sorting."),
    ]
    for path, key, prefix, description in sources:
        if not path.is_file():
            continue
        table = pd.read_csv(path)
        if key not in table and "cluster_id" in table:
            key = "cluster_id"
        if key not in table:
            if table.columns[0].startswith("Unnamed") or table.columns[0] == "":
                key = table.columns[0]
            else:
                raise SortingError(f"{path}: no {key} column")
        table = table.set_index(key)
        if not set(table.index) <= set(int(u) for u in unit_ids) or len(table) != len(unit_ids):
            notes.append(f"{path.name}: {len(table)} rows for {len(unit_ids)} clusters; rows are matched by id, and clusters without a row get NaN or an empty value")
            if not set(table.index) & set(int(u) for u in unit_ids):
                raise SortingError(f"{path}: none of its ids are clusters of {folder}, so it belongs to another sorting")
        files[str(path.relative_to(folder.parent))] = f"{len(table)} rows"
        for name in table.columns:
            if name == key:
                continue
            values = table[name]
            if values.dtype.kind in "fiub":
                data = _per_unit(values.astype("float64"), unit_ids, np.nan).astype("float64")
            else:
                data = np.array(["" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in _per_unit(values, unit_ids, "")])
            columns.append(Column(prefix + re.sub(r"\W+", "_", str(name)).strip("_"), description.format(name=name), data))
    return columns
