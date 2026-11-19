"""Reading every sorter folder layout the lab has produced, and refusing the ones that can't be trusted."""

import numpy as np
import pytest

from tank_lab_to_nwb.ephys.sorting import (
    SortingError,
    check_folder,
    find_sorter_folders,
    identify_sorter,
    read_sorting,
    split_trains,
)
from tests.synthetic_ephys import phy_folder


def _column(sorted_probe, name):
    return {column.name: column.data for column in sorted_probe.columns}[name]


@pytest.mark.parametrize(
    "kind, pc, sorter, version",
    [
        ("ks4", True, "Kilosort 4", "4.0.37"),
        ("ks4", False, "Kilosort 4", "4.0.37"),
        ("ks3", False, "Kilosort 3", "unknown"),
        ("ks2", True, "Kilosort 2", "2.0.2"),
    ],
)
def test_every_layout_reads_back_exactly(tmp_path, kind, pc, sorter, version):
    folder, trains, cluster_ids = phy_folder(tmp_path / f"{kind}_output", kind=kind, with_pc_features=pc)
    if kind == "ks3":
        (folder / "kilosort_overview.fig").write_bytes(b"")
    result = read_sorting(folder)
    assert (result.sorter, result.sorter_version) == (sorter, version)
    assert np.array_equal(result.unit_ids, cluster_ids)
    for unit, train in zip(result.unit_ids, result.spike_samples):
        assert train.dtype == np.int64
        assert np.array_equal(train, trains[int(unit)].astype(np.int64))
    assert result.n_spikes == sum(v.size for v in trains.values())
    assert list(_column(result, "ks_label")) == ["good", "mua", "noise", "good", "mua"]
    assert np.allclose(_column(result, "ks_amplitude"), 10 + cluster_ids)


def test_numpy_trains_equal_spikeinterface_where_it_keeps_every_spike(tmp_path):
    import spikeinterface.extractors as se

    folder, trains, _ = phy_folder(tmp_path / "kilosort4_output", kind="ks4")
    extractor = se.read_kilosort(folder, keep_good_only=False)
    unit_ids, ours = split_trains(np.load(folder / "spike_times.npy"), np.load(folder / "spike_clusters.npy"))
    assert sorted(extractor.get_unit_ids().tolist()) == unit_ids.tolist()
    for unit, train in zip(unit_ids, ours):
        assert np.array_equal(extractor.get_unit_spike_train(unit_id=unit), train)


def test_spikeinterface_drops_clusters_a_header_only_group_file_leaves_out(tmp_path):
    """Why the trains aren't read through SpikeInterface: phy writes cluster_group.tsv with only a header when nobody labels anything."""
    import spikeinterface.extractors as se

    folder, trains, cluster_ids = phy_folder(tmp_path / "kilosort4_output", kind="ks4")
    (folder / "cluster_group.tsv").write_text("cluster_id\tgroup\n")
    extractor = se.read_kilosort(folder, keep_good_only=False)
    assert len(extractor.get_unit_ids()) < cluster_ids.size
    result = read_sorting(folder)
    assert result.n_spikes == sum(v.size for v in trains.values())
    assert set(_column(result, "curated_group")) == {""}
    assert list(_column(result, "quality")) == list(_column(result, "ks_label"))


def test_curated_labels_stay_apart_from_kilosorts(tmp_path):
    folder, _, cluster_ids = phy_folder(tmp_path / "kilosort4_output", curated=True, with_cluster_info=True)
    result = read_sorting(folder)
    ks, curated, quality = (_column(result, name) for name in ("ks_label", "curated_group", "quality"))
    assert (ks[0], curated[0], quality[0]) == ("good", "noise", "noise")
    assert list(quality[1:]) == list(ks[1:])
    assert np.array_equal(_column(result, "phy_n_spikes"), [train.size for train in result.spike_samples])


def test_cluster_group_and_cluster_info_must_agree(tmp_path):
    folder, _, _ = phy_folder(tmp_path / "kilosort4_output", curated=True, with_cluster_info=True)
    text = (folder / "cluster_group.tsv").read_text().replace("2\tmua", "2\tnoise")
    (folder / "cluster_group.tsv").write_text(text)
    with pytest.raises(SortingError, match="disagree"):
        read_sorting(folder)


def test_stale_cluster_info_is_refused(tmp_path):
    folder, _, _ = phy_folder(tmp_path / "kilosort4_output", with_cluster_info="stale")
    with pytest.raises(SortingError, match=r"cluster_info.tsv.*another sorting"):
        read_sorting(folder)


def test_metrics_from_spikeinterface_and_ibl(tmp_path):
    folder, _, cluster_ids = phy_folder(tmp_path / "job" / "kilosort4_output", spikeinterface=True, ibl=True)
    result = read_sorting(folder)
    assert np.allclose(_column(result, "si_snr"), 5.0 + cluster_ids)
    assert np.allclose(_column(result, "ibl_label"), 0.5 + cluster_ids / 100)
    assert list(_column(result, "ibl_ks2_label")) == list(_column(result, "ks_label"))
    assert "Kilosort 4" == result.sorter and "sorter_provenance.json" in result.sorter_evidence


def test_sorted_input_and_sample_origin(tmp_path):
    folder, _, _ = phy_folder(tmp_path / "kilosort4_output")
    assert identify_sorter(folder)[3] == "/mnt/cup/raw/x_t0.imec0.ap.bin"
    assert read_sorting(folder).sample_origin == "sample 0 of the raw t0 AP file"
    log = folder / "kilosort4.log"
    log.write_text(log.read_text().replace("x_t0.imec0.ap.bin", "catgt_output/x_tcat.imec0.ap.bin"))
    assert "CatGT" in read_sorting(folder).sample_origin


@pytest.mark.parametrize(
    "remove, message",
    [("spike_times.npy", "missing spike_times.npy"), ("params.py", "missing params.py"), ("channel_positions.npy", "missing channel_positions.npy")],
)
def test_missing_files_are_named(tmp_path, remove, message):
    folder, _, _ = phy_folder(tmp_path / "kilosort4_output")
    (folder / remove).unlink()
    with pytest.raises(SortingError, match=message) as error:
        check_folder(folder)
    assert str(folder) in str(error.value)


def test_spikeinterface_work_folder_is_explained(tmp_path):
    folder = tmp_path / "kilosort2_output_si_work"
    (folder / "sorter_output").mkdir(parents=True)
    with pytest.raises(SortingError, match="sorter_output"):
        check_folder(folder)


def test_unknown_sorter_is_refused(tmp_path):
    folder, _, _ = phy_folder(tmp_path / "mystery_output", kind="ks4")
    (folder / "kilosort4.log").unlink()
    (folder / "pc_features.npy").unlink()
    with pytest.raises(SortingError, match="can't tell which sorter"):
        read_sorting(folder)


def test_mismatched_lengths_are_refused(tmp_path):
    folder, _, _ = phy_folder(tmp_path / "kilosort4_output")
    np.save(folder / "spike_clusters.npy", np.load(folder / "spike_clusters.npy")[:-1])
    with pytest.raises(SortingError, match="spikes but"):
        read_sorting(folder)


def test_find_sorter_folders_includes_non_kilosort(tmp_path):
    for name in ("kilosort3_output", "kilosort4_output", "mountainsort5_output"):
        phy_folder(tmp_path / name)
    (tmp_path / "ibl_data").mkdir()
    assert [p.name for p in find_sorter_folders(tmp_path)] == ["kilosort3_output", "kilosort4_output", "mountainsort5_output"]
    with pytest.raises(SortingError, match="no \\*_output"):
        find_sorter_folders(tmp_path / "ibl_data")


def test_ibl_data_of_another_sorting_is_left_out(tmp_path):
    job = tmp_path / "job_id_603"
    phy_folder(job / "kilosort4_output", ibl=True)
    other, _, _ = phy_folder(job / "kilosort3_output", kind="ks3", seed=7, with_pc_features=False)
    (other / "kilosort_overview.fig").write_bytes(b"")
    result = read_sorting(other)
    assert not any(column.name.startswith("ibl_") for column in result.columns)
    assert any("ibl_data belongs to another sorting" in note for note in result.notes)
    assert any(column.name.startswith("ibl_") for column in read_sorting(job / "kilosort4_output").columns)


def test_kilosort4s_own_cluster_group_copy_is_not_curation(tmp_path):
    folder, _, _ = phy_folder(tmp_path / "kilosort4_output")
    (folder / "cluster_group.tsv").write_text((folder / "cluster_KSLabel.tsv").read_text())
    result = read_sorting(folder)
    assert "curated_group" not in {column.name for column in result.columns}
    assert any("holds no curation" in note for note in result.notes)
    assert list(_column(result, "quality")) == list(_column(result, "ks_label"))


def test_clusters_listed_without_spikes_are_noted(tmp_path):
    folder, _, cluster_ids = phy_folder(tmp_path / "kilosort4_output")
    text = (folder / "cluster_KSLabel.tsv").read_text() + "99\tmua\n"
    (folder / "cluster_KSLabel.tsv").write_text(text)
    result = read_sorting(folder)
    assert 99 not in result.unit_ids
    assert any("1 cluster" in note and "no spikes" in note for note in result.notes)


def test_older_phy_cluster_info_with_an_id_column(tmp_path):
    """phy before 2.0b5 wrote cluster_info.tsv with an `id` column (rec 628)."""
    folder, _, _ = phy_folder(tmp_path / "kilosort4_output", curated=True, with_cluster_info=True)
    info = folder / "cluster_info.tsv"
    info.write_text(info.read_text().replace("cluster_id\t", "id\t", 1))
    result = read_sorting(folder)
    assert _column(result, "curated_group")[0] == "noise"
    assert "cluster_info.tsv" in result.files
