"""Restructure and segment the Zhang et al. battery-cycling dataset.

The module converts cycle-specific Excel worksheets into a standardized text-file
hierarchy and partitions each charge or discharge record at its global current
extremum.
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

COLUMN_FILES = (
    "voltage.txt",
    "current.txt",
    "capacity.txt",
    "energy.txt",
    "time.txt",
)
STAGES = ("charge", "discharge")
ZONE_PATTERN = re.compile(r"^(.*_zone_)(\d+)(\.txt)$")


def _sorted_cycles(sheet_names: Sequence[str]) -> list[str]:
    """Return numerically ordered names when every sheet name is an integer."""
    names = [str(name) for name in sheet_names]
    try:
        return [str(number) for number in sorted(map(int, names))]
    except ValueError:
        return names


def _validate_columns(frame: pd.DataFrame, source: Path) -> None:
    if frame.shape[1] < len(COLUMN_FILES):
        raise ValueError(
            f"{source} contains {frame.shape[1]} columns; "
            f"at least {len(COLUMN_FILES)} are required."
        )


def _write_columns(frame: pd.DataFrame, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for column_index, filename in enumerate(COLUMN_FILES):
        frame.iloc[:, column_index].to_csv(
            destination / filename,
            index=False,
            header=False,
            na_rep="",
        )


def restructure_dataset(
    dataset_path: str | Path = "dataset",
    relaxation_time_offset: float = 1.0,
    cell_mapping: Mapping[str, str] | None = None,
    verbose: bool = False,
) -> None:
    """Convert source workbooks to a standardized cell/cycle/stage hierarchy."""
    root = Path(dataset_path)
    mapping = dict(cell_mapping or {"Cell 1": "B0001", "Cell 2": "B0002"})

    for source_name, target_name in mapping.items():
        source_dir = root / source_name
        if not source_dir.is_dir():
            if verbose:
                print(f"Skipped missing source directory: {source_dir}")
            continue

        workbook_paths = {
            "charge": source_dir / "Charging stage.xlsx",
            "discharge": source_dir / "Discharging stage.xlsx",
            "relaxation": source_dir / "Relaxation stage.xlsx",
        }
        missing = [path for path in workbook_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing required workbook(s): " + ", ".join(map(str, missing))
            )

        workbooks = {
            name: pd.ExcelFile(path) for name, path in workbook_paths.items()
        }
        common_sheets = set.intersection(
            *(set(workbook.sheet_names) for workbook in workbooks.values())
        )
        cycles = _sorted_cycles(list(common_sheets))

        for cycle in cycles:
            target_cycle = root / target_name / cycle
            charge = workbooks["charge"].parse(cycle, header=None)
            discharge = workbooks["discharge"].parse(cycle, header=None)
            relaxation = workbooks["relaxation"].parse(cycle, header=None)

            for frame, source in (
                (charge, workbook_paths["charge"]),
                (discharge, workbook_paths["discharge"]),
                (relaxation, workbook_paths["relaxation"]),
            ):
                _validate_columns(frame, source)

            _write_columns(charge, target_cycle / "charge")

            discharge_time = pd.to_numeric(discharge.iloc[:, 4], errors="coerce")
            relaxation_time = pd.to_numeric(relaxation.iloc[:, 4], errors="coerce")
            valid_discharge_time = discharge_time.dropna()
            valid_relaxation_time = relaxation_time.dropna()

            last_discharge = (
                float(valid_discharge_time.iloc[-1])
                if not valid_discharge_time.empty
                else 0.0
            )
            first_relaxation = (
                float(valid_relaxation_time.iloc[0])
                if not valid_relaxation_time.empty
                else 0.0
            )
            shift = last_discharge + relaxation_time_offset - first_relaxation

            adjusted_relaxation = relaxation.copy()
            shifted_time = relaxation_time + shift
            adjusted_relaxation.iloc[:, 4] = shifted_time.where(
                shifted_time.notna(), relaxation.iloc[:, 4]
            )
            combined = pd.concat(
                [discharge, adjusted_relaxation], ignore_index=True
            )
            _write_columns(combined, target_cycle / "discharge")

            if verbose:
                print(f"Processed {target_name}, cycle {cycle}")


def _read_stage(stage_dir: Path) -> pd.DataFrame:
    missing = [name for name in COLUMN_FILES if not (stage_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {stage_dir}: {', '.join(missing)}"
        )
    columns = [pd.read_csv(stage_dir / name, header=None) for name in COLUMN_FILES]
    lengths = {len(column) for column in columns}
    if len(lengths) != 1:
        raise ValueError(f"Column lengths are inconsistent in {stage_dir}")
    return pd.concat(columns, axis=1)


def split_current_zones(
    dataset_path: str | Path = "dataset",
    battery_pattern: str = "B000*",
    verbose: bool = False,
) -> None:
    """Split each stage immediately after its global current extremum."""
    root = Path(dataset_path)
    for battery_dir in sorted(path for path in root.glob(battery_pattern) if path.is_dir()):
        for cycle_dir in sorted(
            (path for path in battery_dir.iterdir() if path.is_dir()),
            key=lambda path: (not path.name.isdigit(), int(path.name) if path.name.isdigit() else path.name),
        ):
            for stage in STAGES:
                stage_dir = cycle_dir / stage
                if not stage_dir.is_dir():
                    continue

                frame = _read_stage(stage_dir)
                current = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
                if current.dropna().empty:
                    raise ValueError(f"No numeric current values in {stage_dir}")

                boundary_label = current.idxmax() if stage == "charge" else current.idxmin()
                boundary_position = frame.index.get_loc(boundary_label)
                zones = (
                    frame.iloc[: boundary_position + 1],
                    frame.iloc[boundary_position + 1 :],
                )

                for zone_index, zone in enumerate(zones):
                    for column_index, filename in enumerate(COLUMN_FILES):
                        output = stage_dir / filename.replace(
                            ".txt", f"_zone_{zone_index}.txt"
                        )
                        zone.iloc[:, column_index].to_csv(
                            output, index=False, header=False, na_rep=""
                        )

                if verbose:
                    print(f"Segmented {battery_dir.name}/{cycle_dir.name}/{stage}")


def shift_zone_indices(dataset_path: str | Path) -> None:
    """Increase every zone index by one using collision-safe temporary names."""
    pending: list[tuple[Path, Path]] = []
    for path in Path(dataset_path).rglob("*.txt"):
        match = ZONE_PATTERN.match(path.name)
        if not match:
            continue
        prefix, zone_number, extension = match.groups()
        target = path.with_name(f"{prefix}{int(zone_number) + 1}{extension}")
        temporary = path.with_name(path.name + ".tmp")
        path.replace(temporary)
        pending.append((temporary, target))

    for temporary, target in pending:
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {target}")
        temporary.replace(target)


def delete_zone_files(dataset_path: str | Path) -> int:
    """Delete generated zone files and return the number removed."""
    removed = 0
    for path in Path(dataset_path).rglob("*.txt"):
        if ZONE_PATTERN.match(path.name):
            path.unlink()
            removed += 1
    return removed


def cleanup_outputs(
    dataset_path: str | Path,
    targets: Sequence[str] = ("B0001", "B0002"),
) -> None:
    """Remove only explicitly named generated battery directories."""
    root = Path(dataset_path).resolve()
    for target in targets:
        path = (root / target).resolve()
        if path.parent != root or target not in targets:
            raise ValueError(f"Unsafe cleanup target: {target}")
        if path.is_dir():
            shutil.rmtree(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    restructure = subparsers.add_parser("restructure")
    restructure.add_argument("dataset", nargs="?", default="dataset")
    restructure.add_argument("--relaxation-offset", type=float, default=1.0)
    restructure.add_argument("--verbose", action="store_true")

    split = subparsers.add_parser("split")
    split.add_argument("dataset", nargs="?", default="dataset")
    split.add_argument("--verbose", action="store_true")

    shift = subparsers.add_parser("shift-zones")
    shift.add_argument("dataset", nargs="?", default="dataset")

    delete = subparsers.add_parser("delete-zones")
    delete.add_argument("dataset", nargs="?", default="dataset")

    cleanup = subparsers.add_parser("cleanup")
    cleanup.add_argument("dataset", nargs="?", default="dataset")
    cleanup.add_argument("--targets", nargs="+", default=["B0001", "B0002"])
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "restructure":
        restructure_dataset(
            args.dataset,
            relaxation_time_offset=args.relaxation_offset,
            verbose=args.verbose,
        )
    elif args.command == "split":
        split_current_zones(args.dataset, verbose=args.verbose)
    elif args.command == "shift-zones":
        shift_zone_indices(args.dataset)
    elif args.command == "delete-zones":
        print(f"Deleted {delete_zone_files(args.dataset)} zone files.")
    elif args.command == "cleanup":
        cleanup_outputs(args.dataset, args.targets)


if __name__ == "__main__":
    main()
