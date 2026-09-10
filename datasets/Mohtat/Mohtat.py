"""Protocol-aware segmentation of the Mohtat battery-aging dataset.

Each cycle is classified from current polarity into charge, discharge, and rest.
Active charge and discharge records are further labeled as constant-current (CC)
or constant-voltage (CV) using joint voltage-level and current-taper criteria.
Rest intervals are associated with the immediately preceding active stage.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CYCLE_COLUMN = "Cycle number"
TIME_COLUMN = "Time [s]"
CURRENT_COLUMN = "Current [mA]"
VOLTAGE_COLUMN = "Voltage [V]"
REQUIRED_COLUMNS = (CYCLE_COLUMN, TIME_COLUMN, CURRENT_COLUMN, VOLTAGE_COLUMN)
INVALID_FILENAME_CHARACTERS = re.compile(r'[<>:"/\\|?*]')


@dataclass(frozen=True)
class SegmentationConfig:
    """Numerical criteria used by the phase-classification algorithm."""

    rest_current_threshold_ma: float = 2.0
    cc_current_tolerance_ma: float = 5.0
    cv_voltage_tolerance_v: float = 0.01
    minimum_phase_samples: int = 5
    minimum_cv_samples: int = 10
    minimum_cv_duration_s: float = 10.0
    cv_current_gradient_tolerance_ma_per_sample: float = 1.0
    discard_cycle_zero: bool = True
    save_all_data: bool = True


def _sanitize_filename(name: object) -> str:
    return INVALID_FILENAME_CHARACTERS.sub("_", str(name))


def _cycle_name(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _clean_cycle(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS[1:] if column not in frame]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    result = frame.copy()
    for column in REQUIRED_COLUMNS[1:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=list(REQUIRED_COLUMNS[1:]))
    result = result.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)
    return result


def _contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    padded = np.r_[False, values, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _merge_short_regions(labels: np.ndarray, minimum_length: int) -> np.ndarray:
    result = np.asarray(labels, dtype=object).copy()
    while result.size:
        boundaries = np.r_[0, np.flatnonzero(result[1:] != result[:-1]) + 1, result.size]
        regions = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
        short = next(
            ((index, start, end) for index, (start, end) in enumerate(regions)
             if end - start < minimum_length),
            None,
        )
        if short is None or len(regions) == 1:
            return result
        index, start, end = short
        if index == 0:
            replacement = result[regions[1][0]]
        elif index == len(regions) - 1:
            replacement = result[regions[-2][0]]
        else:
            left = regions[index - 1][1] - regions[index - 1][0]
            right = regions[index + 1][1] - regions[index + 1][0]
            replacement = result[regions[index - 1][0] if left >= right else regions[index + 1][0]]
        result[start:end] = replacement
    return result


def _infer_charge_sign(current: np.ndarray, rest_threshold: float) -> int:
    active = current[np.isfinite(current) & (np.abs(current) > rest_threshold)]
    if active.size == 0:
        return 1
    positive = int(np.sum(active > 0))
    negative = int(np.sum(active < 0))
    return -1 if positive > 0 and negative > 5 * positive else 1


def _main_labels(frame: pd.DataFrame, config: SegmentationConfig) -> tuple[np.ndarray, int]:
    current = frame[CURRENT_COLUMN].to_numpy(float)
    charge_sign = _infer_charge_sign(current, config.rest_current_threshold_ma)
    signed = current * charge_sign
    labels = np.full(len(frame), "relaxation", dtype=object)
    labels[signed > config.rest_current_threshold_ma] = "charge"
    labels[signed < -config.rest_current_threshold_ma] = "discharge"
    return labels, charge_sign


def _associate_relaxation(main_labels: np.ndarray) -> np.ndarray:
    """Associate each rest sample with the closest preceding active stage."""
    association = np.full(main_labels.size, "unassigned", dtype=object)
    previous: str | None = None
    for index, label in enumerate(main_labels):
        if label in {"charge", "discharge"}:
            previous = str(label)
            association[index] = label
        elif previous is not None:
            association[index] = previous
    return association


def _classify_active_phase(
    frame: pd.DataFrame,
    stage: str,
    charge_sign: int,
    config: SegmentationConfig,
) -> np.ndarray:
    """Classify one active stage as CC or CV using voltage and current behavior."""
    n = len(frame)
    labels = np.full(n, "CC", dtype=object)
    if n < config.minimum_phase_samples:
        return labels

    current = frame[CURRENT_COLUMN].to_numpy(float)
    voltage = frame[VOLTAGE_COLUMN].to_numpy(float)
    time = frame[TIME_COLUMN].to_numpy(float)
    magnitude = np.abs(current)
    high_current = magnitude >= np.percentile(magnitude, 70)
    typical_cc = float(np.median(magnitude[high_current]))
    below_cc = magnitude < typical_cc - config.cc_current_tolerance_ma

    if stage == "charge":
        voltage_limit = float(np.max(voltage))
        near_limit = voltage >= voltage_limit - config.cv_voltage_tolerance_v
        correct_polarity = current * charge_sign > config.rest_current_threshold_ma
    else:
        voltage_limit = float(np.min(voltage))
        near_limit = voltage <= voltage_limit + config.cv_voltage_tolerance_v
        correct_polarity = current * charge_sign < -config.rest_current_threshold_ma

    smoothed = pd.Series(magnitude).rolling(5, center=True, min_periods=1).median().to_numpy()
    tapering = np.gradient(smoothed) <= config.cv_current_gradient_tolerance_ma_per_sample
    candidate = near_limit & below_cc & correct_polarity & tapering

    for start, end in _contiguous_regions(candidate):
        duration = time[end] - time[start]
        if end - start + 1 >= config.minimum_cv_samples and duration >= config.minimum_cv_duration_s:
            labels[start:end + 1] = "CV"
    return _merge_short_regions(labels, config.minimum_phase_samples)


def _write_columns(frame: pd.DataFrame, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    names = [_sanitize_filename(column) for column in frame.columns if not str(column).startswith("_")]
    if len(names) != len(set(names)):
        raise ValueError(f"Sanitized column-name collision in {destination}")
    for column in (column for column in frame.columns if not str(column).startswith("_")):
        frame[column].to_csv(destination / f"{_sanitize_filename(column)}.txt", index=False, header=False, na_rep="")


def _write_subset(frame: pd.DataFrame, destination: Path) -> int:
    if frame.empty:
        return 0
    _write_columns(frame, destination)
    frame.loc[:, [c for c in frame.columns if not str(c).startswith("_")]].to_csv(
        destination / "data.csv", index=False
    )
    return len(frame)


def _protocol(counts: dict[str, int], stage: str) -> str:
    phases = [phase for phase in ("CC", "CV", "relaxation") if counts[phase] > 0]
    return "_".join(phases) + f"_{stage}" if phases else f"no_valid_{stage}_detected"


def process_cycle(
    cycle_frame: pd.DataFrame,
    battery: str,
    cycle: float,
    output: Path,
    config: SegmentationConfig,
) -> dict[str, object]:
    if config.discard_cycle_zero and float(cycle) == 0:
        return {"battery": battery, "cycle": cycle, "status": "DISCARDED_CYCLE_0"}

    frame = _clean_cycle(cycle_frame)
    if len(frame) < config.minimum_phase_samples:
        return {"battery": battery, "cycle": cycle, "status": "TOO_SHORT", "samples": len(frame)}

    main, charge_sign = _main_labels(frame, config)
    association = _associate_relaxation(main)
    physical = np.full(len(frame), "unassigned", dtype=object)
    records: dict[str, object] = {
        "battery": battery, "cycle": cycle, "status": "OK", "samples": len(frame),
        "charge_sign": charge_sign,
    }

    for stage in ("charge", "discharge"):
        stage_positions = np.flatnonzero(association == stage)
        stage_frame = frame.iloc[stage_positions].copy()
        stage_main = main[stage_positions]
        stage_labels = np.full(len(stage_frame), "relaxation", dtype=object)
        active_local = np.flatnonzero(stage_main == stage)
        if active_local.size:
            active_frame = stage_frame.iloc[active_local]
            stage_labels[active_local] = _classify_active_phase(
                active_frame, stage, charge_sign, config
            )
        physical[stage_positions] = stage_labels

        stage_output = output / stage
        if config.save_all_data and not stage_frame.empty:
            _write_columns(stage_frame, stage_output / "all")
            stage_frame.to_csv(stage_output / "all.csv", index=False)

        counts = {}
        for phase in ("CC", "CV", "relaxation"):
            subset = stage_frame.iloc[np.flatnonzero(stage_labels == phase)]
            counts[phase] = _write_subset(subset, stage_output / phase)
            records[f"{stage}_{phase}_samples"] = counts[phase]
        records[f"{stage}_samples"] = len(stage_frame)
        records[f"{stage}_protocol"] = _protocol(counts, stage)
        records[f"{stage}_has_CV"] = counts["CV"] >= config.minimum_cv_samples

    diagnostic = frame.copy()
    diagnostic["_main_mode"] = main
    diagnostic["_stage_association"] = association
    diagnostic["_physical_phase"] = physical
    output.mkdir(parents=True, exist_ok=True)
    diagnostic.to_csv(output / "cycle_with_phase_labels.csv", index=False)
    return records


def process_dataset(
    input_dataset: str | Path,
    output_dataset: str | Path,
    config: SegmentationConfig = SegmentationConfig(),
    overwrite: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    source = Path(input_dataset).resolve()
    destination = Path(output_dataset).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if source == destination:
        raise ValueError("Input and output directories must differ.")
    csv_files = sorted(source.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {source}")
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace it: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    audit: list[dict[str, object]] = []
    for csv_path in csv_files:
        try:
            data = pd.read_csv(csv_path)
        except Exception as exc:
            audit.append({"battery": csv_path.stem, "cycle": "ALL", "status": "ERROR_READING_CSV", "message": str(exc)})
            continue
        missing = [column for column in REQUIRED_COLUMNS if column not in data]
        if missing:
            audit.append({"battery": csv_path.stem, "cycle": "ALL", "status": "MISSING_COLUMNS", "message": ", ".join(missing)})
            continue
        cycle_values = pd.to_numeric(data[CYCLE_COLUMN], errors="coerce")
        for cycle in sorted(cycle_values.dropna().unique()):
            cycle_data = data.loc[cycle_values == cycle].copy()
            try:
                record = process_cycle(
                    cycle_data, csv_path.stem, float(cycle),
                    destination / csv_path.stem / _cycle_name(float(cycle)), config,
                )
            except Exception as exc:
                record = {"battery": csv_path.stem, "cycle": cycle, "status": "ERROR_PROCESSING_CYCLE", "message": str(exc)}
            audit.append(record)
            if verbose:
                print(f"{csv_path.stem}, cycle {_cycle_name(float(cycle))}: {record['status']}")

    audit_frame = pd.DataFrame(audit)
    audit_frame.to_csv(destination / "splitting_audit.csv", index=False)
    return audit_frame


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dataset")
    parser.add_argument("output_dataset")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--rest-threshold-ma", type=float, default=2.0)
    parser.add_argument("--cc-tolerance-ma", type=float, default=5.0)
    parser.add_argument("--cv-voltage-tolerance-v", type=float, default=0.01)
    parser.add_argument("--minimum-phase-samples", type=int, default=5)
    parser.add_argument("--minimum-cv-samples", type=int, default=10)
    parser.add_argument("--minimum-cv-duration-s", type=float, default=10.0)
    parser.add_argument("--cv-gradient-tolerance", type=float, default=1.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = SegmentationConfig(
        rest_current_threshold_ma=args.rest_threshold_ma,
        cc_current_tolerance_ma=args.cc_tolerance_ma,
        cv_voltage_tolerance_v=args.cv_voltage_tolerance_v,
        minimum_phase_samples=args.minimum_phase_samples,
        minimum_cv_samples=args.minimum_cv_samples,
        minimum_cv_duration_s=args.minimum_cv_duration_s,
        cv_current_gradient_tolerance_ma_per_sample=args.cv_gradient_tolerance,
    )
    audit = process_dataset(
        args.input_dataset, args.output_dataset, config,
        overwrite=args.overwrite, verbose=args.verbose,
    )
    print(audit["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
