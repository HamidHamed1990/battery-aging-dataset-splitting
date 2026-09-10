"""Extract and segment v7.3 MATLAB battery-aging batches.

The module reads a battery batch through h5py, exports aligned cycle signals, divides
cycles into charge and discharge observations, and partitions current profiles into
robust plateau-defined zones. Source files are never modified.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

SIGNALS = ("time", "temperature", "Qc", "Qd", "I", "V")
FIELD_MAP = {"time": "t", "temperature": "T", "Qc": "Qc", "Qd": "Qd", "I": "I", "V": "V"}
BATTERY_PATTERN = re.compile(r"^B\d{4}$")
ZONE_PATTERN = re.compile(r"^(?P<signal>.+)_zone_(?P<zone>\d+)\.txt$")


@dataclass(frozen=True)
class ExtractionConfig:
    """Criteria for charge and discharge identification."""

    battery_index_offset: int = 0
    current_epsilon: float = 1e-8
    use_capacity_fallback: bool = True


@dataclass(frozen=True)
class ZoneConfig:
    """Parameters controlling robust current-zone detection."""

    zero_current_tolerance: float = 0.1
    minimum_plateau_tolerance: float = 0.1
    plateau_break_window: int = 5
    plateau_initial_samples: int = 10
    plateau_initial_fraction: float = 0.10
    discharge_step_threshold: float = 0.5
    minimum_zone_samples: int = 1
    merge_short_zone_threshold: int = 5
    merge_nonplateau_zones: bool = False
    plateau_test_tolerance: float = 0.005
    plateau_test_minimum_length: int = 20

    def validate(self) -> None:
        nonnegative = (
            self.zero_current_tolerance,
            self.minimum_plateau_tolerance,
            self.discharge_step_threshold,
            self.plateau_test_tolerance,
        )
        if any(value < 0 for value in nonnegative):
            raise ValueError("All tolerances must be non-negative.")
        if self.plateau_break_window < 1 or self.plateau_initial_samples < 1:
            raise ValueError("Plateau windows must be positive.")
        if not 0 < self.plateau_initial_fraction <= 1:
            raise ValueError("plateau_initial_fraction must lie in (0, 1].")
        if self.minimum_zone_samples < 1:
            raise ValueError("minimum_zone_samples must be positive.")


def _cell_length(cell_array: h5py.Dataset) -> int:
    shape = cell_array.shape
    if len(shape) == 1:
        return shape[0]
    return shape[1] if shape[0] == 1 else shape[0]


def _cell_reference(cell_array: h5py.Dataset, index: int):
    if len(cell_array.shape) == 1:
        return cell_array[index]
    return cell_array[0, index] if cell_array.shape[0] == 1 else cell_array[index, 0]


def _read_cell_numeric(
    file: h5py.File, cell_array: h5py.Dataset, index: int
) -> Optional[np.ndarray]:
    try:
        reference = _cell_reference(cell_array, index)
        values = np.asarray(file[reference][()]).squeeze()
        return np.atleast_1d(values).astype(np.float64).reshape(-1)
    except (KeyError, IndexError, TypeError, ValueError, OSError):
        return None


def _number_of_cells(batch: h5py.Group) -> int:
    if "cycles" not in batch:
        raise KeyError("The batch group does not contain 'cycles'.")
    return _cell_length(batch["cycles"])


def _aligned_signals(
    file: h5py.File, cycles: h5py.Group, cycle_index: int
) -> dict[str, np.ndarray]:
    arrays = {}
    for output_name, field_name in FIELD_MAP.items():
        if field_name in cycles:
            values = _read_cell_numeric(file, cycles[field_name], cycle_index)
            if values is not None and values.size:
                arrays[output_name] = values
    if not arrays:
        return {}
    minimum = min(len(values) for values in arrays.values())
    arrays = {name: values[:minimum] for name, values in arrays.items()}
    arrays.setdefault("time", np.arange(minimum, dtype=float))
    return arrays


def _phase_masks(
    arrays: dict[str, np.ndarray], config: ExtractionConfig
) -> tuple[np.ndarray, np.ndarray, str]:
    length = len(next(iter(arrays.values())))
    current = arrays.get("I")
    if current is not None and np.nanmax(np.abs(current)) > config.current_epsilon:
        return current > config.current_epsilon, current < -config.current_epsilon, "current_sign"

    charge = np.zeros(length, dtype=bool)
    discharge = np.zeros(length, dtype=bool)
    if config.use_capacity_fallback:
        if "Qc" in arrays:
            charge = np.diff(arrays["Qc"], prepend=arrays["Qc"][0]) > 0
        if "Qd" in arrays:
            discharge = np.diff(arrays["Qd"], prepend=arrays["Qd"][0]) > 0
    return charge, discharge, "capacity_gradient"


def _write_signal(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(values).reshape(-1), fmt="%.12e")


def extract_batch(
    mat_file: str | Path,
    output_root: str | Path,
    config: ExtractionConfig = ExtractionConfig(),
    overwrite: bool = False,
    verbose: bool = False,
) -> list[dict[str, object]]:
    """Extract all cells and cycles from one MATLAB v7.3 batch file."""
    source = Path(mat_file).resolve()
    destination = Path(output_root).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    audit = []
    with h5py.File(source, "r") as file:
        if "batch" not in file:
            raise KeyError("The MATLAB file does not contain a 'batch' group.")
        batch = file["batch"]
        for cell_index in range(_number_of_cells(batch)):
            battery_number = config.battery_index_offset + cell_index + 1
            battery = f"B{battery_number:04d}"
            try:
                cycles_ref = _cell_reference(batch["cycles"], cell_index)
                cycles = file[cycles_ref]
            except (KeyError, IndexError, TypeError, OSError) as error:
                audit.append({"battery": battery, "cycle": "ALL", "status": "ERROR", "message": str(error)})
                continue
            available_fields = [field for field in FIELD_MAP.values() if field in cycles]
            if not available_fields:
                audit.append({"battery": battery, "cycle": "ALL", "status": "NO_CYCLE_FIELDS"})
                continue
            cycle_count = _cell_length(cycles[available_fields[0]])
            for cycle_index in range(cycle_count):
                arrays = _aligned_signals(file, cycles, cycle_index)
                if not arrays:
                    audit.append({"battery": battery, "cycle": cycle_index + 1, "status": "EMPTY"})
                    continue
                charge, discharge, method = _phase_masks(arrays, config)
                cycle_folder = destination / battery / str(cycle_index + 1)
                for phase, mask in (("charge", charge), ("discharge", discharge)):
                    for name, values in arrays.items():
                        _write_signal(cycle_folder / phase / f"{name}.txt", values[mask])
                audit.append(
                    {
                        "battery": battery,
                        "cycle": cycle_index + 1,
                        "status": "OK",
                        "classification_method": method,
                        "samples": len(next(iter(arrays.values()))),
                        "charge_samples": int(charge.sum()),
                        "discharge_samples": int(discharge.sum()),
                    }
                )
            if verbose:
                print(f"{battery}: {cycle_count} cycles")
    return audit


def _load_mode(folder: Path) -> dict[str, np.ndarray]:
    arrays = {}
    for signal in SIGNALS:
        path = folder / f"{signal}.txt"
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty signal: {path}")
        arrays[signal] = np.atleast_1d(np.loadtxt(path)).astype(float)
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"Unaligned signals in {folder}")
    return arrays


def _charge_zones(current: np.ndarray, config: ZoneConfig) -> list[tuple[int, int]]:
    """Detect sequential near-zero boundary and robust current plateaus."""
    length = len(current)
    if length == 0:
        return []
    zero_indices = np.flatnonzero(np.abs(current) <= config.zero_current_tolerance)
    if zero_indices.size == 0:
        return [(0, length)]

    first_zero = int(zero_indices[0])
    zones = [(0, first_zero)] if first_zero > 0 else []
    cursor = first_zero
    while cursor < length:
        remaining = current[cursor:]
        if len(remaining) < config.plateau_initial_samples:
            zones.append((cursor, length))
            break
        initial_length = max(
            config.plateau_initial_samples,
            int(np.ceil(config.plateau_initial_fraction * len(remaining))),
        )
        initial_length = min(initial_length, len(remaining))
        initial = remaining[:initial_length]
        center = float(np.median(initial))
        mad = float(np.median(np.abs(initial - center)))
        tolerance = max(config.minimum_plateau_tolerance, 3.0 * mad + np.finfo(float).eps)

        outside_count = 0
        end_relative = len(remaining)
        for index in range(initial_length, len(remaining)):
            outside_count = outside_count + 1 if abs(remaining[index] - center) > tolerance else 0
            if outside_count >= config.plateau_break_window:
                end_relative = index - config.plateau_break_window + 1
                break
        if end_relative <= 0:
            end_relative = min(len(remaining), config.plateau_initial_samples)
        end = cursor + end_relative
        zones.append((cursor, end))
        cursor = end
    return zones


def _discharge_zones(current: np.ndarray, config: ZoneConfig) -> list[tuple[int, int]]:
    if len(current) == 0:
        return []
    steps = np.flatnonzero(np.abs(np.diff(current)) > config.discharge_step_threshold)
    if steps.size == 0:
        return [(0, len(current))]
    boundary = int(steps[-1] + 1)
    return [(0, boundary), (boundary, len(current))]


def _save_zones(folder: Path, arrays: dict[str, np.ndarray], zones: list[tuple[int, int]], minimum: int) -> None:
    for zone_number, (start, end) in enumerate(zones, start=1):
        if end - start < minimum:
            continue
        for signal, values in arrays.items():
            _write_signal(folder / f"{signal}_zone_{zone_number}.txt", values[start:end])


def _zone_has_plateau(values: np.ndarray, tolerance: float, minimum_length: int) -> bool:
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) < minimum_length:
        return False
    for start in range(len(values) - minimum_length + 1):
        window = values[start : start + minimum_length]
        if np.ptp(window) <= tolerance:
            return True
    return False


def _zone_numbers(folder: Path) -> list[int]:
    numbers = set()
    for path in folder.glob("*_zone_*.txt"):
        match = ZONE_PATTERN.match(path.name)
        if match:
            numbers.add(int(match.group("zone")))
    return sorted(numbers)


def _merge_zone_pair(folder: Path, first: int, second: int) -> None:
    for signal in SIGNALS:
        first_path = folder / f"{signal}_zone_{first}.txt"
        second_path = folder / f"{signal}_zone_{second}.txt"
        if first_path.is_file() and second_path.is_file():
            merged = np.concatenate((np.atleast_1d(np.loadtxt(first_path)), np.atleast_1d(np.loadtxt(second_path))))
            _write_signal(first_path, merged)
            second_path.unlink()
    for old_number in [number for number in _zone_numbers(folder) if number > second]:
        for signal in SIGNALS:
            old = folder / f"{signal}_zone_{old_number}.txt"
            if old.is_file():
                old.rename(folder / f"{signal}_zone_{old_number - 1}.txt")


def _postprocess_zones(folder: Path, config: ZoneConfig) -> None:
    numbers = _zone_numbers(folder)
    if 3 in numbers and 4 in numbers:
        zone_three = folder / "I_zone_3.txt"
        if zone_three.is_file() and len(np.atleast_1d(np.loadtxt(zone_three))) < config.merge_short_zone_threshold:
            _merge_zone_pair(folder, 3, 4)

    if not config.merge_nonplateau_zones:
        return
    zone = 1
    while zone < max(_zone_numbers(folder), default=0):
        first_path = folder / f"I_zone_{zone}.txt"
        second_path = folder / f"I_zone_{zone + 1}.txt"
        if not first_path.is_file() or not second_path.is_file():
            zone += 1
            continue
        first = np.atleast_1d(np.loadtxt(first_path))
        second = np.atleast_1d(np.loadtxt(second_path))
        first_has = _zone_has_plateau(first, config.plateau_test_tolerance, config.plateau_test_minimum_length)
        second_has = _zone_has_plateau(second, config.plateau_test_tolerance, config.plateau_test_minimum_length)
        if not first_has and not second_has:
            _merge_zone_pair(folder, zone, zone + 1)
        else:
            zone += 1


def split_zones(dataset_root: str | Path, config: ZoneConfig = ZoneConfig(), verbose: bool = False) -> None:
    """Generate synchronized charge and discharge zone files without deleting cycles."""
    config.validate()
    root = Path(dataset_root)
    for battery in sorted(path for path in root.iterdir() if path.is_dir() and BATTERY_PATTERN.fullmatch(path.name)):
        for cycle in sorted((path for path in battery.iterdir() if path.is_dir() and path.name.isdigit()), key=lambda path: int(path.name)):
            for mode in ("charge", "discharge"):
                folder = cycle / mode
                if not folder.is_dir():
                    continue
                try:
                    arrays = _load_mode(folder)
                    zones = _charge_zones(arrays["I"], config) if mode == "charge" else _discharge_zones(arrays["I"], config)
                    _save_zones(folder, arrays, zones, config.minimum_zone_samples)
                    _postprocess_zones(folder, config)
                except ValueError as error:
                    if verbose:
                        print(f"Skipped {folder}: {error}")


def remove_zone_files(dataset_root: str | Path) -> int:
    removed = 0
    for path in Path(dataset_root).rglob("*_zone_*.txt"):
        if ZONE_PATTERN.match(path.name):
            path.unlink()
            removed += 1
    return removed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract")
    extract.add_argument("mat_file")
    extract.add_argument("output_root")
    extract.add_argument("--battery-offset", type=int, default=0)
    extract.add_argument("--current-epsilon", type=float, default=1e-8)
    extract.add_argument("--overwrite", action="store_true")
    extract.add_argument("--verbose", action="store_true")

    split = commands.add_parser("split")
    split.add_argument("dataset_root")
    split.add_argument("--zero-tolerance", type=float, default=0.1)
    split.add_argument("--plateau-tolerance", type=float, default=0.1)
    split.add_argument("--break-window", type=int, default=5)
    split.add_argument("--discharge-step-threshold", type=float, default=0.5)
    split.add_argument("--merge-nonplateau", action="store_true")
    split.add_argument("--verbose", action="store_true")

    clean = commands.add_parser("clean-zones")
    clean.add_argument("dataset_root")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "extract":
        audit = extract_batch(
            args.mat_file,
            args.output_root,
            ExtractionConfig(args.battery_offset, args.current_epsilon),
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        print(f"Processed {len(audit)} cycle records.")
    elif args.command == "split":
        config = ZoneConfig(
            zero_current_tolerance=args.zero_tolerance,
            minimum_plateau_tolerance=args.plateau_tolerance,
            plateau_break_window=args.break_window,
            discharge_step_threshold=args.discharge_step_threshold,
            merge_nonplateau_zones=args.merge_nonplateau,
        )
        split_zones(args.dataset_root, config, verbose=args.verbose)
    else:
        print(f"Removed {remove_zone_files(args.dataset_root)} zone files.")


if __name__ == "__main__":
    main()
