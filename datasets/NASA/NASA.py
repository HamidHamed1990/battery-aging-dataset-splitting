"""Convert and segment MATLAB battery-cycling records.

The module reconstructs complete electrochemical cycles from ordered charge,
discharge, and impedance operations; exports MATLAB fields as aligned text files;
segments charge and discharge records by measured current; and separates complex
impedance channels into real and imaginary components.
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

CYCLE_PATTERNS = (
    ("charge", "impedance", "discharge", "impedance"),
    ("charge", "impedance", "discharge"),
    ("charge", "discharge", "impedance"),
    ("charge", "discharge"),
)
CHARGE_CHANNELS = (
    "Time.txt",
    "Voltage_measured.txt",
    "Voltage_charge.txt",
    "Current_charge.txt",
    "Temperature_measured.txt",
    "Current_measured.txt",
)
DISCHARGE_CHANNELS = (
    "Time.txt",
    "Voltage_measured.txt",
    "Voltage_load.txt",
    "Current_load.txt",
    "Temperature_measured.txt",
    "Current_measured.txt",
)
IMPEDANCE_CHANNELS = (
    "Battery_impedance.txt",
    "Battery_current.txt",
    "Current_ratio.txt",
    "Sense_current.txt",
)
ZONE_FILE = re.compile(r"_zone_\d+\.txt$")


@dataclass(frozen=True)
class CurrentSplitConfig:
    """Parameters controlling current-based stage segmentation."""

    charge_tolerance_a: float = 0.01
    discharge_tolerance_a: float = 0.01
    initial_samples_to_exclude: int = 5
    impedance_stability_fraction: float = 0.05
    maximum_impedance_ramp_samples: int = 5
    minimum_impedance_samples: int = 10

    def validate(self) -> None:
        if self.charge_tolerance_a < 0 or self.discharge_tolerance_a < 0:
            raise ValueError("Current tolerances must be non-negative.")
        if self.initial_samples_to_exclude < 0:
            raise ValueError("initial_samples_to_exclude must be non-negative.")
        if not 0 <= self.impedance_stability_fraction < 1:
            raise ValueError("impedance_stability_fraction must lie in [0, 1).")
        if self.maximum_impedance_ramp_samples < 0:
            raise ValueError("maximum_impedance_ramp_samples must be non-negative.")
        if self.minimum_impedance_samples < 1:
            raise ValueError("minimum_impedance_samples must be positive.")


def _matlab_datetime(values: Any) -> datetime:
    components = np.asarray(values).reshape(-1)
    if components.size < 6:
        raise ValueError("MATLAB timestamp contains fewer than six components.")
    return datetime(*(int(value) for value in components[:6]))


def _flatten(values: Any) -> list[Any]:
    array = np.asarray(values)
    return [array.item()] if array.ndim == 0 else array.reshape(-1).tolist()


def load_mat_operations(mat_path: str | Path) -> list[dict[str, Any]]:
    """Load the operation sequence from one MATLAB battery file."""
    path = Path(mat_path)
    contents = scipy.io.loadmat(path)
    candidates = [key for key in contents if not key.startswith("__")]
    preferred = path.stem if path.stem in contents else None
    variable = preferred or (candidates[0] if len(candidates) == 1 else None)
    if variable is None:
        raise KeyError(
            f"Could not identify battery variable in {path}; candidates={candidates}"
        )

    root = contents[variable]
    try:
        operations = root[0][0][0][0]
    except (IndexError, TypeError) as error:
        raise ValueError(f"Unexpected MATLAB structure in {path}") from error

    records: list[dict[str, Any]] = []
    for operation in operations:
        data_struct = operation[3][0]
        fields = list(data_struct.dtype.fields or {})
        data = {
            field: _flatten(data_struct[0][index])
            for index, field in enumerate(fields)
        }
        records.append(
            {
                "operation": str(operation[0][0]).strip(),
                "temperature": int(operation[1][0]),
                "timestamp": _matlab_datetime(operation[2][0]).isoformat(sep=" "),
                "data": data,
            }
        )
    return records


def remove_consecutive_duplicates(
    operations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply the dataset-specific duplicate-operation selection rules."""
    cleaned: list[dict[str, Any]] = []
    index = 0
    while index < len(operations):
        current = operations[index]
        following = operations[index + 1] if index + 1 < len(operations) else None
        if following and current["operation"] == following["operation"]:
            if current["operation"] == "charge":
                cleaned.append(following)
                index += 2
                continue
            if current["operation"] in {"discharge", "impedance"}:
                cleaned.append(current)
                index += 2
                continue
        cleaned.append(current)
        index += 1
    return cleaned


def reconstruct_cycles(
    operations: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Greedily match complete cycles using longest protocol patterns first."""
    cleaned = remove_consecutive_duplicates(operations)
    cycles: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(cleaned):
        if cleaned[index]["operation"] != "charge":
            index += 1
            continue
        match = None
        for pattern in CYCLE_PATTERNS:
            window = cleaned[index : index + len(pattern)]
            if tuple(item["operation"] for item in window) == pattern:
                match = window
                break
        if match is None:
            index += 1
            continue
        cycles.append(match)
        index += len(match)
    return cycles


def _format_value(value: Any) -> str:
    scalar = np.asarray(value).item() if np.asarray(value).ndim == 0 else value
    if np.iscomplexobj(scalar):
        number = complex(scalar)
        return f"{number.real:.17g}{number.imag:+.17g}j"
    return str(scalar)


def _write_values(values: list[Any] | np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(_format_value(value) for value in values))
        if len(values):
            stream.write("\n")


def export_battery(
    mat_path: str | Path,
    output_root: str | Path,
    overwrite_battery: bool = False,
) -> int:
    """Reconstruct and export all complete cycles from one MATLAB file."""
    mat_path = Path(mat_path)
    battery_output = Path(output_root) / mat_path.stem
    if battery_output.exists() and overwrite_battery:
        shutil.rmtree(battery_output)
    battery_output.mkdir(parents=True, exist_ok=True)

    cycles = reconstruct_cycles(load_mat_operations(mat_path))
    for cycle_number, operations in enumerate(cycles, start=1):
        impedance_number = 0
        for operation in operations:
            operation_type = operation["operation"]
            if operation_type == "impedance":
                impedance_number += 1
                folder_name = f"impedance_{impedance_number}"
            else:
                folder_name = operation_type
            destination = battery_output / str(cycle_number) / folder_name
            for field, values in operation["data"].items():
                _write_values(values, destination / f"{field}.txt")
    return len(cycles)


def export_dataset(
    input_root: str | Path,
    output_root: str | Path,
    overwrite: bool = False,
    verbose: bool = False,
) -> dict[str, int]:
    """Export every top-level MAT file to a separate output hierarchy."""
    source = Path(input_root).resolve()
    destination = Path(output_root).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if source == destination:
        raise ValueError("Input and output directories must differ.")
    mat_files = sorted(source.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No MAT files found in {source}")
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    counts = {}
    for mat_path in mat_files:
        counts[mat_path.stem] = export_battery(mat_path, destination)
        if verbose:
            print(f"{mat_path.stem}: {counts[mat_path.stem]} cycles")
    return counts


def _load_signal(path: Path) -> np.ndarray:
    values = np.loadtxt(path, ndmin=1)
    if np.iscomplexobj(values):
        if np.any(np.abs(values.imag) > 0):
            raise ValueError(f"Expected a real-valued signal: {path}")
        values = values.real
    return np.asarray(values, dtype=float).reshape(-1)


def _validate_alignment(folder: Path, filenames: tuple[str, ...]) -> dict[str, np.ndarray]:
    signals = {
        filename: _load_signal(folder / filename)
        for filename in filenames
        if (folder / filename).is_file()
    }
    lengths = {len(values) for values in signals.values()}
    if len(lengths) > 1:
        raise ValueError(f"Channel lengths are inconsistent in {folder}")
    return signals


def _write_zones(
    folder: Path,
    signals: dict[str, np.ndarray],
    zones: tuple[np.ndarray, ...],
) -> None:
    for filename, signal in signals.items():
        for zone_number, indices in enumerate(zones, start=1):
            if indices.size:
                _write_values(
                    signal[indices],
                    folder / f"{Path(filename).stem}_zone_{zone_number}.txt",
                )


def split_charge(folder: Path, config: CurrentSplitConfig) -> None:
    signals = _validate_alignment(folder, CHARGE_CHANNELS)
    if "Current_measured.txt" not in signals or "Time.txt" not in signals:
        return
    current = signals["Current_measured.txt"]
    maximum = float(np.max(current))
    near_maximum = current >= maximum - config.charge_tolerance_a
    near_zero = np.abs(current) <= config.charge_tolerance_a
    transition = ~(near_maximum | near_zero)
    retained = np.arange(len(current)) >= config.initial_samples_to_exclude
    zones = tuple(np.flatnonzero(mask & retained) for mask in (near_maximum, transition, near_zero))
    _write_zones(folder, signals, zones)


def split_discharge(folder: Path, config: CurrentSplitConfig) -> None:
    signals = _validate_alignment(folder, DISCHARGE_CHANNELS)
    if "Current_measured.txt" not in signals or "Time.txt" not in signals:
        return
    current = signals["Current_measured.txt"]
    minimum = float(np.min(current))
    near_minimum = current <= minimum + config.discharge_tolerance_a
    near_zero = np.abs(current) <= config.discharge_tolerance_a
    retained = np.arange(len(current)) >= config.initial_samples_to_exclude
    zones = tuple(np.flatnonzero(mask & retained) for mask in (near_minimum, near_zero))
    _write_zones(folder, signals, zones)


def separate_impedance_components(dataset_root: str | Path) -> int:
    """Write real and imaginary parts of supported impedance channels."""
    processed = 0
    for folder in Path(dataset_root).rglob("impedance_*"):
        if not folder.is_dir():
            continue
        for filename in IMPEDANCE_CHANNELS:
            path = folder / filename
            if not path.is_file():
                continue
            values = np.loadtxt(path, dtype=complex, ndmin=1)
            _write_values(values.real, folder / f"{path.stem}_real.txt")
            _write_values(values.imag, folder / f"{path.stem}_imaginary.txt")
            processed += 1
    return processed


def _longest_contiguous(indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return indices
    groups = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    return max(groups, key=len)


def split_impedance_by_current(
    dataset_root: str | Path,
    config: CurrentSplitConfig,
) -> int:
    """Separate the longest near-maximum current plateau from remaining samples."""
    processed = 0
    for current_path in Path(dataset_root).rglob("Battery_current_real.txt"):
        folder = current_path.parent
        current = _load_signal(current_path)
        if len(current) < config.minimum_impedance_samples:
            continue

        differences = np.diff(current)
        ramp_end = 1 if len(current) > 1 else 0
        limit = min(config.maximum_impedance_ramp_samples, len(current) - 1)
        while ramp_end < limit and differences[ramp_end - 1] > 0:
            ramp_end += 1
        trimmed = current[ramp_end:]
        if len(trimmed) < config.minimum_impedance_samples:
            continue

        reference = float(np.max(trimmed))
        tolerance = config.impedance_stability_fraction * max(abs(reference), np.finfo(float).eps)
        candidates = np.flatnonzero(np.abs(trimmed - reference) < tolerance)
        stable = _longest_contiguous(candidates)
        if stable.size == 0:
            continue
        stable = stable + ramp_end
        considered = np.arange(ramp_end, len(current))
        remainder = np.setdiff1d(considered, stable, assume_unique=True)

        for path in sorted(folder.glob("*.txt")):
            if ZONE_FILE.search(path.name) or path.name == "Battery_impedance.txt":
                continue
            try:
                signal = _load_signal(path)
            except ValueError:
                continue
            if len(signal) != len(current):
                continue
            for zone, indices in enumerate((stable, remainder), start=1):
                _write_values(signal[indices], folder / f"{path.stem}_zone_{zone}.txt")
        processed += 1
    return processed


def segment_dataset(dataset_root: str | Path, config: CurrentSplitConfig) -> None:
    config.validate()
    root = Path(dataset_root)
    for battery in sorted(path for path in root.iterdir() if path.is_dir()):
        for cycle in sorted(
            (path for path in battery.iterdir() if path.is_dir()),
            key=lambda path: (not path.name.isdigit(), int(path.name) if path.name.isdigit() else path.name),
        ):
            split_charge(cycle / "charge", config) if (cycle / "charge").is_dir() else None
            split_discharge(cycle / "discharge", config) if (cycle / "discharge").is_dir() else None
    separate_impedance_components(root)
    split_impedance_by_current(root, config)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export")
    export.add_argument("input_root")
    export.add_argument("output_root")
    export.add_argument("--overwrite", action="store_true")
    export.add_argument("--verbose", action="store_true")

    segment = commands.add_parser("segment")
    segment.add_argument("dataset_root")
    segment.add_argument("--charge-tolerance", type=float, default=0.01)
    segment.add_argument("--discharge-tolerance", type=float, default=0.01)
    segment.add_argument("--exclude-initial", type=int, default=5)
    segment.add_argument("--impedance-stability-fraction", type=float, default=0.05)
    segment.add_argument("--maximum-ramp-samples", type=int, default=5)
    segment.add_argument("--minimum-impedance-samples", type=int, default=10)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "export":
        export_dataset(
            args.input_root,
            args.output_root,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
    else:
        config = CurrentSplitConfig(
            charge_tolerance_a=args.charge_tolerance,
            discharge_tolerance_a=args.discharge_tolerance,
            initial_samples_to_exclude=args.exclude_initial,
            impedance_stability_fraction=args.impedance_stability_fraction,
            maximum_impedance_ramp_samples=args.maximum_ramp_samples,
            minimum_impedance_samples=args.minimum_impedance_samples,
        )
        segment_dataset(args.dataset_root, config)


if __name__ == "__main__":
    main()
