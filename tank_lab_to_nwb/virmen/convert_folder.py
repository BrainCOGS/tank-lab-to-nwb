"""Convert a folder of ViRMEn logs and lay them out for DANDI, locally. Nothing is uploaded.

Usage: python -m tank_lab_to_nwb.virmen.convert_folder <logs folder> <output folder> [--only NAME.mat ...] [--choose SESSION=GATES ...]

Logs are found at <user>/<user>_<subject>/<date>_g<N>/*.mat. Each subject and day is one session, so a lone g1 is simply
that day's session: someone can start a session, delete the first attempt and keep the second. Several gates on one day
stop the conversion until you choose: --choose user_subject/20260807=g1 converts g1 as the session, and
--choose user_subject/20260807=g0,g1 keeps both, as sessions 20260807-g0 and 20260807-g1.

NWB files go to <output>/staging, and dandi organize names them under <output>/dandiset from each file's subject_id and
session_id. Run `dandi validate <output>/dandiset` after.
"""

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

from . import convert
from .layout import organize_for_dandi
from .loader import load_log
from .session import normalize_log

_GATE_FOLDER = re.compile(r"^(?P<day>\d{8})_(?P<gate>g\d+)$")
_CHOICE = re.compile(r"^(?P<subject>[^/=]+)/(?P<day>\d{8})=(?P<gates>g\d+(?:,g\d+)*)$")


class SessionSelectionError(ValueError):
    """The logs don't name one session each, or what a log says disagrees with where it's kept."""


@dataclass(frozen=True)
class SelectedLog:
    path: Path
    subject: str
    day: str
    gate: str
    session_suffix: str = ""


def parse_choice(text):
    """user_subject/20260807=g0,g1 as ((subject, day), [gates])."""
    match = _CHOICE.match(text)
    if not match:
        raise SessionSelectionError(f"--choose {text}: write it as user_subject/YYYYMMDD=g1 or user_subject/YYYYMMDD=g0,g1")
    return (match["subject"], match["day"]), match["gates"].split(",")


def _sessions(source, only):
    """Every log under `source`, as {(subject, day): {gate: [paths]}}."""
    sessions = defaultdict(lambda: defaultdict(list))
    for path in sorted(Path(source).glob("*/*/*/*.mat")):
        match = _GATE_FOLDER.match(path.parent.name)
        if match and (not only or path.name in only):
            sessions[(path.parent.parent.name, match["day"])][match["gate"]].append(path)
    if not sessions:
        raise SessionSelectionError(f"{source}: no logs to convert")
    return sessions


def select_logs(source, only=None, choices=None):
    """The logs to convert. Several gates on one day are an error unless `choices` or `only` settles which."""
    choices = choices or {}
    sessions = _sessions(source, only)
    unknown = [key for key in choices if key not in sessions]
    if unknown:
        raise SessionSelectionError("--choose names no session here: " + ", ".join(f"{subject}/{day}" for subject, day in unknown))
    conflicts, selected = [], []
    for key, gates in sorted(sessions.items()):
        subject, day = key
        chosen = choices.get(key)
        if chosen is None and len(gates) > 1:
            conflicts.append(f"{subject}/{day} has {', '.join(sorted(gates))}")
            continue
        chosen = chosen or list(gates)
        missing = [gate for gate in chosen if gate not in gates]
        if missing:
            raise SessionSelectionError(f"--choose {subject}/{day}: no {', '.join(missing)}; it has {', '.join(sorted(gates))}")
        for gate in sorted(chosen):
            if len(gates[gate]) > 1:
                raise SessionSelectionError(f"{subject}/{day}_{gate} holds more than one log: {', '.join(path.name for path in gates[gate])}. Use --only to pick one.")
            suffix = f"-{gate}" if len(chosen) > 1 else ""
            selected.append(SelectedLog(gates[gate][0], subject, day, gate, suffix))
    if conflicts:
        raise SessionSelectionError(
            "several gates for one session, so which is the session is your choice:\n  "
            + "\n  ".join(conflicts)
            + "\nAdd --choose user_subject/YYYYMMDD=g1 for one, or --choose user_subject/YYYYMMDD=g0,g1 to keep both as separate sessions."
        )
    return selected


def check_session(selected, session):
    """What the log says about itself must agree with the folder it's kept in."""
    where = f"{selected.subject}/{selected.path.parent.name}/{selected.path.name}"
    if session.session_id != selected.day:
        raise SessionSelectionError(f"{where}: the log's session is {session.session_id}, but its folder says {selected.day}")
    if session.subject_id != selected.subject:
        raise SessionSelectionError(f"{where}: the log's subject is {session.subject_id}, but its folder says {selected.subject}")


def _unchosen_logs(source, selected, only):
    """Logs of the converted sessions that weren't chosen, whose earlier output must not linger under the same session."""
    sessions = _sessions(source, only)
    kept = {item.path for item in selected}
    converted = {(item.subject, item.day) for item in selected}
    return [path for key in converted for paths in sessions[key].values() for path in paths if path not in kept]


def _remove_stale_organized(organized):
    """Files of the same subject and day from an earlier choice of gates, which this run's choice replaces."""
    for path in organized:
        day = re.search(r"_ses-(\d{8})", path.name)
        for other in path.parent.glob("*.nwb"):
            if other not in organized and day and re.search(rf"_ses-{day[1]}(-g\d+)?(_|\.)", other.name):
                other.unlink()


def convert_folder(source, output, only=None, choices=None):
    """Convert the chosen logs under `source` into `output`/staging, and return the organized paths under `output`/dandiset."""
    source, output = Path(source), Path(output)
    selected = select_logs(source, only=only, choices=choices)
    staging = output / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    for path in _unchosen_logs(source, selected, only):
        (staging / f"{path.stem}.nwb").unlink(missing_ok=True)
        (staging / f"{path.stem}.coverage.json").unlink(missing_ok=True)
    for item in selected:
        session = normalize_log(load_log(item.path), file=item.path)
        check_session(item, session)
        convert.write_and_verify(replace(session, session_id=session.session_id + item.session_suffix), staging / f"{item.path.stem}.nwb")
    organized = organize_for_dandi(staging, output / "dandiset", files_mode="copy")
    _remove_stale_organized(organized)
    return organized


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--only", nargs="*", help="Convert only logs with these file names.")
    parser.add_argument("--choose", action="append", default=[], metavar="SESSION=GATES", help="Which gates of a session to convert: user_subject/YYYYMMDD=g1, or =g0,g1 to keep both.")
    arguments = parser.parse_args()
    for path in convert_folder(arguments.source, arguments.output, only=arguments.only, choices=dict(map(parse_choice, arguments.choose))):
        print(path)


if __name__ == "__main__":
    main()
