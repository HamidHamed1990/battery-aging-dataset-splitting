"""Preprocess and segment current-defined battery cycling records.

The module converts battery-level CSV files into a standardized hierarchy of
battery, cycle, and operating stage. Charge data are divided into two zones at
the first sufficiently negative change in the smoothed current gradient, while
discharge data are retained as a single zone.
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

CYCLE_COLUMN = "cycle number"
CURRENT_COLUMN = "<I>/mA"
TIME_COLUMN = "time/s"
REQUIRED_COLUMNS = (CYCLE_COLUMN, CURRENT_COLUMN, TIME_COLUMN)
INVALID_FILENAME_CHARACTERS = re.compile(r'[<>:"/\\|?*]')
ZONE_FILE_PATTERN = re.compile(r".*_zone_\d+\.txt$")


def sanitize_filename(name: object) -> str:
    """Return a filesystem-safe representation of a column name."""
    return INVALID_FILENAME_CHARACTERS.sub("_", str(name))


def format_cycle_name(value: object) -> str:
    """Format integral numeric cycle identifiers without a decimal suffix."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(numeric)) if numeric.is_integer() else str(value)


def detect_operating_regions(
    cycle_data: pd.DataFrame,
    current_threshold_ma: float = 0.1,
    minimum_samples: int = 3,
) -> dict[str, pd.DataFrame]:
    """Classify observations as charge or discharge from current polarity."""
    if current_threshold_ma < 0:
        raise ValueError("current_threshold_ma must be non-negative.")
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be at least one.")

    current = pd.to_numeric(cycle_data[CURRENT_COLUMN], errors="coerce")
    masks = {
        "charge": current > current_threshold_ma,
        "discharge": current < -current_threshold_ma,
    }
    return {
        stage: cycle_data.loc[mask].reset_index(drop=True)
        for stage, mask in masks.items()
        if int(mask.sum()) >= minimum_samples
    }


def write_frame_columns(frame: pd.DataFrame, destination: Path) -> None:
    """Write each dataframe column to an aligned, headerless text file."""
    destination.mkdir(parents=True, exist_ok=True)
    output_names = [sanitize_filename(column) for column in frame.columns]
    if len(output_names) != len(set(output_names)):
        raise ValueError(
            f"Sanitized column names are not unique in {destination}: {output_names}"
        )

    for column, output_name in zip(frame.columns, output_names, strict=True):
        frame[column].to_csv(
            destination / f"{output_name}.txt",
            index=False,
            header=False,
            na_rep="",
        )


def create_battery_structure(
    dataset_path: str | Path,
    current_threshold_ma: float = 0.1,
    minimum_samples: int = 3,
    verbose: bool = False,
) -> None:
    """Convert every top-level CSV file into battery/cycle/stage directories."""
    root = Path(dataset_path)
    if not root.is_dir():
        raise NotADirectoryError(root)

    for csv_path in sorted(root.glob("*.csv")):
        data = pd.read_csv(csv_path)
        missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
        if missing:
            if verbose:
                print(f"Skipped {csv_path.name}; missing columns: {', '.join(missing)}")
            continue

        battery_dir = root / csv_path.stem
        cycles = data[CYCLE_COLUMN].dropna().unique()
        try:
            cycles = sorted(cycles, key=float)
        except (TypeError, ValueError):
            cycles = sorted(cycles, key=str)

        for cycle in cycles:
            cycle_data = data.loc[data[CYCLE_COLUMN] == cycle].reset_index(drop=True)
            regions = detect_operating_regions(
                cycle_data,
                current_threshold_ma=current_threshold_ma,
                minimum_samples=minimum_samples,
            )
            for stage, region in regions.items():
                write_frame_columns(
                    region,
                    battery_dir / format_cycle_name(cycle) / stage,
                )

        if verbose:
            print(f"Processed {csv_path.name}")


def read_numeric_column(path: Path) -> np.ndarray:
    """Read a headerless text column as a finite one-dimensional float array."""
    values = pd.to_numeric(pd.read_csv(path, header=None).iloc[:, 0], errors="coerce")
    if values.isna().any():
        invalid = int(values.isna().sum())
        raise ValueError(f"{path} contains {invalid} non-numeric or missing values.")
    return values.to_numpy(dtype=float)


def write_numeric_column(values: np.ndarray, path: Path) -> None:
    """Write a numeric array using a reproducible text representation."""
    np.savetxt(path, values, fmt="%.17g")


def find_charge_split_index(
    current_ma: np.ndarray,
    gradient_threshold_ma_per_sample: float = -0.05,
    smoothing_window: int = 10,
) -> int:
    """Locate the first smoothed current gradient below a negative threshold."""
    current_ma = np.asarray(current_ma, dtype=float)
    if current_ma.ndim != 1:
        raise ValueError("current_ma must be one-dimensional.")
    if current_ma.size < 2:
        return current_ma.size
    if not np.isfinite(current_ma).all():
        raise ValueError("current_ma contains non-finite values.")
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be at least one.")
    if gradient_threshold_ma_per_sample >= 0:
        raise ValueError("gradient threshold must be negative.")

    gradient = np.gradient(current_ma)
    smoothed_gradient = uniform_filter1d(
        gradient,
        size=min(smoothing_window, gradient.size),
        mode="nearest",
    )
    candidates = np.flatnonzero(smoothed_gradient < gradient_threshold_ma_per_sample)
    return int(candidates[0]) if candidates.size else current_ma.size


def _unsegmented_text_files(stage_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in stage_dir.glob("*.txt")
        if not ZONE_FILE_PATTERN.fullmatch(path.name)
    )


def _load_aligned_columns(paths: Iterable[Path]) -> dict[Path, np.ndarray]:
    columns = {path: read_numeric_column(path) for path in paths}
    lengths = {values.size for values in columns.values()}
    if len(lengths) > 1:
        details = ", ".join(f"{path.name}={data.size}" for path, data in columns.items())
        raise ValueError(f"Unaligned columns: {details}")
    return columns


def split_zones_in_dataset(
    dataset_path: str | Path,
    gradient_threshold_ma_per_sample: float = -0.05,
    smoothing_window: int = 10,
    verbose: bool = False,
) -> None:
    """Split charge stages into two zones and copy discharge as one zone."""
    root = Path(dataset_path)
    current_filename = f"{sanitize_filename(CURRENT_COLUMN)}.txt"

    for battery_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for cycle_dir in sorted(
            (path for path in battery_dir.iterdir() if path.is_dir()),
            key=lambda path: (
                not path.name.isdigit(),
                int(path.name) if path.name.isdigit() else path.name,
            ),
        ):
            for stage in ("charge", "discharge"):
                stage_dir = cycle_dir / stage
                if not stage_dir.is_dir():
                    continue

                files = _unsegmented_text_files(stage_dir)
                if not files:
                    continue
                columns = _load_aligned_columns(files)

                if stage == "discharge":
                    for source in files:
                        target = source.with_name(f"{source.stem}_zone_1.txt")
                        shutil.copyfile(source, target)
                else:
                    current_path = stage_dir / current_filename
                    if current_path not in columns:
                        if verbose:
                            print(f"Skipped {stage_dir}; missing {current_filename}")
                        continue
                    split_index = find_charge_split_index(
                        columns[current_path],
                        gradient_threshold_ma_per_sample,
                        smoothing_window,
                    )
                    for source, values in columns.items():
                        write_numeric_column(
                            values[:split_index],
                            source.with_name(f"{source.stem}_zone_1.txt"),
                        )
                        write_numeric_column(
                            values[split_index:],
                            source.with_name(f"{source.stem}_zone_2.txt"),
                        )

                if verbose:
                    print(f"Segmented {battery_dir.name}/{cycle_dir.name}/{stage}")


def remove_zone_files(dataset_path: str | Path) -> int:
    """Remove generated zone files and return the number deleted."""
    removed = 0
    for path in Path(dataset_path).rglob("*.txt"):
        if ZONE_FILE_PATTERN.fullmatch(path.name):
            path.unlink()
            removed += 1
    return removed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    structure = commands.add_parser("structure")
    structure.add_argument("dataset", nargs="?", default="dataset")
    structure.add_argument("--current-threshold", type=float, default=0.1)
    structure.add_argument("--minimum-samples", type=int, default=3)
    structure.add_argument("--verbose", action="store_true")

    split = commands.add_parser("split")
    split.add_argument("dataset", nargs="?", default="dataset")
    split.add_argument("--gradient-threshold", type=float, default=-0.05)
    split.add_argument("--smoothing-window", type=int, default=10)
    split.add_argument("--verbose", action="store_true")

    clean = commands.add_parser("clean")
    clean.add_argument("dataset", nargs="?", default="dataset")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "structure":
        create_battery_structure(
            args.dataset,
            current_threshold_ma=args.current_threshold,
            minimum_samples=args.minimum_samples,
            verbose=args.verbose,
        )
    elif args.command == "split":
        split_zones_in_dataset(
            args.dataset,
            gradient_threshold_ma_per_sample=args.gradient_threshold,
            smoothing_window=args.smoothing_window,
            verbose=args.verbose,
        )
    elif args.command == "clean":
        print(f"Removed {remove_zone_files(args.dataset)} zone files.")


if __name__ == "__main__":
    main()
