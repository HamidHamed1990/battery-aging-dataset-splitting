"""Split the Lu battery-aging dataset into electrochemical operating phases.

Battery-level Excel workbooks are separated by cycle. Current polarity identifies
charge, discharge, and rest observations; charge observations are further
classified as constant-current (CC) or constant-voltage (CV). Rest intervals are
assigned to the immediately preceding active stage, and a cycle-level audit
records protocol and time-quality diagnostics.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COLUMN_MAP = {
    "Data_Point": "Data_Point",
    "Test_Time(s)": "Test_Times",
    "Current(A)": "CurrentA",
    "Capacity(Ah)": "CapacityAh",
    "Voltage(V)": "VoltageV",
    "Energy(Wh)": "EnergyWh",
    "Temperature(°C)": "Temperature_degC",
    "Date_Time": "Date_Time",
    "Cycle_Index": "Cycle_Index",
}
REQUIRED_COLUMNS = ("Current(A)", "Voltage(V)", "Cycle_Index")


@dataclass(frozen=True)
class SplitConfig:
    """Numerical criteria controlling phase identification."""

    rest_current_threshold_a: float = 0.05
    cc_current_tolerance_a: float = 0.10
    cv_voltage_tolerance_v: float = 0.01
    minimum_phase_samples: int = 3
    rolling_window: int = 5
    cv_expected_until_cycle: int = 20

    def validate(self) -> None:
        if min(
            self.rest_current_threshold_a,
            self.cc_current_tolerance_a,
            self.cv_voltage_tolerance_v,
        ) < 0:
            raise ValueError("All tolerances must be non-negative.")
        if self.minimum_phase_samples < 1 or self.rolling_window < 1:
            raise ValueError("Window and minimum phase lengths must be positive.")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def cycle_name(value: float) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(value)


def reconstruct_time(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    """Return the best available elapsed-time coordinate and its provenance."""
    if "Test_Time(s)" in frame:
        time = numeric(frame["Test_Time(s)"])
        if time.notna().sum() >= 2:
            return time.reset_index(drop=True), "original_Test_Time(s)"

    if "Date_Time" in frame:
        timestamps = pd.to_datetime(frame["Date_Time"], errors="coerce")
        if timestamps.notna().sum() >= 2:
            origin = timestamps.dropna().iloc[0]
            return (
                (timestamps - origin).dt.total_seconds().reset_index(drop=True),
                "reconstructed_from_Date_Time",
            )

    if "Data_Point" in frame:
        points = numeric(frame["Data_Point"]).reset_index(drop=True)
        if points.notna().sum() >= 2:
            first = points.dropna().iloc[0]
            return points - first, "reconstructed_from_Data_Point"

    return pd.Series(np.arange(len(frame), dtype=float)), "sample_index_fallback"


def infer_charge_sign(frame: pd.DataFrame, config: SplitConfig) -> int:
    """Infer whether positive or negative current corresponds to charging."""
    current = numeric(frame["Current(A)"])
    voltage = numeric(frame["Voltage(V)"])
    valid = current.notna() & voltage.notna()
    if valid.sum() < 5:
        return 1

    i = current[valid].to_numpy()
    dv = np.diff(voltage[valid].to_numpy())
    preceding_current = i[:-1]
    active = np.abs(preceding_current) > config.rest_current_threshold_a
    if active.sum() < 3:
        return 1

    scores: dict[int, float] = {}
    for sign in (1, -1):
        mask = active & (np.sign(preceding_current) == sign)
        scores[sign] = float(np.nanmedian(dv[mask])) if mask.any() else np.nan

    if np.isfinite(scores[1]) and np.isfinite(scores[-1]):
        return 1 if scores[1] >= scores[-1] else -1
    return 1


def run_boundaries(labels: np.ndarray) -> list[tuple[int, int, str]]:
    if labels.size == 0:
        return []
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    ends = np.r_[starts[1:] - 1, labels.size - 1]
    return [(int(a), int(b), str(labels[a])) for a, b in zip(starts, ends)]


def merge_short_runs(labels: pd.Series, minimum: int) -> pd.Series:
    """Iteratively replace short runs with the longer adjacent run."""
    values = labels.astype(object).to_numpy(copy=True)
    while True:
        runs = run_boundaries(values)
        short_index = next(
            (j for j, (start, end, _) in enumerate(runs) if end - start + 1 < minimum),
            None,
        )
        if short_index is None or len(runs) == 1:
            break
        start, end, _ = runs[short_index]
        if short_index == 0:
            replacement = runs[1][2]
        elif short_index == len(runs) - 1:
            replacement = runs[-2][2]
        else:
            left = runs[short_index - 1]
            right = runs[short_index + 1]
            left_length = left[1] - left[0] + 1
            right_length = right[1] - right[0] + 1
            replacement = left[2] if left_length >= right_length else right[2]
        values[start : end + 1] = replacement
    return pd.Series(values, index=labels.index)


def classify_main_stage(
    frame: pd.DataFrame, charge_sign: int, config: SplitConfig
) -> pd.Series:
    """Classify active samples by polarity and attach rest to the prior stage."""
    current = numeric(frame["Current(A)"])
    signed = current * charge_sign
    labels = np.full(len(frame), "unknown", dtype=object)
    labels[(signed > config.rest_current_threshold_a).fillna(False)] = "charge"
    labels[(signed < -config.rest_current_threshold_a).fillna(False)] = "discharge"
    labels[(current.abs() <= config.rest_current_threshold_a).fillna(False)] = "rest"

    last_active: str | None = None
    for position, label in enumerate(labels):
        if label in {"charge", "discharge"}:
            last_active = label
        elif label == "rest" and last_active is not None:
            labels[position] = f"{last_active}_relaxation"
    return pd.Series(labels, index=frame.index)


def classify_charge(
    charge_frame: pd.DataFrame, config: SplitConfig
) -> pd.Series:
    """Classify charge-stage observations as CC, CV, or relaxation."""
    current = numeric(charge_frame["Current(A)"]).reset_index(drop=True)
    voltage = numeric(charge_frame["Voltage(V)"]).reset_index(drop=True)
    labels = np.full(len(charge_frame), "unknown", dtype=object)
    rest = current.abs() <= config.rest_current_threshold_a
    labels[rest.fillna(False)] = "relaxation"

    active = current.abs() > config.rest_current_threshold_a
    active_current = current[active].abs()
    if active_current.empty:
        return pd.Series(labels)

    typical_current = float(active_current.median())
    rolling_current = current.abs().rolling(
        config.rolling_window, center=True, min_periods=1
    ).median()
    current_is_lower = rolling_current < typical_current - config.cc_current_tolerance_a

    if voltage.notna().any():
        maximum_voltage = float(voltage.max())
        voltage_near_max = (maximum_voltage - voltage).abs() <= config.cv_voltage_tolerance_v
        cv = active & current_is_lower & voltage_near_max
        labels[cv.fillna(False)] = "CV"

    labels[(active & (labels == "unknown")).fillna(False)] = "CC"
    return merge_short_runs(pd.Series(labels), config.minimum_phase_samples)


def classify_discharge(discharge_frame: pd.DataFrame, config: SplitConfig) -> pd.Series:
    current = numeric(discharge_frame["Current(A)"]).reset_index(drop=True)
    labels = pd.Series(
        np.where(current.abs() <= config.rest_current_threshold_a, "relaxation", "CC")
    )
    labels[current.isna()] = "unknown"
    return merge_short_runs(labels, config.minimum_phase_samples)


def write_columns(frame: pd.DataFrame, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for source, target in COLUMN_MAP.items():
        if source in frame:
            frame[source].to_csv(folder / f"{target}.txt", index=False, header=False)


def save_phase(frame: pd.DataFrame, labels: pd.Series, phase: str, folder: Path) -> int:
    selected = frame.loc[labels.to_numpy() == phase].copy()
    if selected.empty:
        return 0
    write_columns(selected, folder)
    selected.to_csv(folder / "data.csv", index=False)
    return len(selected)


def process_cycle(
    cycle_frame: pd.DataFrame,
    battery_id: str,
    cycle_index: float,
    cycle_folder: Path,
    config: SplitConfig,
) -> dict[str, object]:
    """Split one cycle, write its phase files, and return an audit record."""
    frame = cycle_frame.reset_index(drop=True).copy()
    missing = [name for name in REQUIRED_COLUMNS[:2] if name not in frame]
    if missing:
        return {
            "battery": battery_id,
            "cycle": cycle_index,
            "status": "ERROR",
            "warnings": "missing_required_columns",
            "message": ", ".join(missing),
        }

    frame["Test_Time(s)"], time_method = reconstruct_time(frame)
    charge_sign = infer_charge_sign(frame, config)
    main_stage = classify_main_stage(frame, charge_sign, config)
    frame["_main_phase"] = main_stage
    frame["_phase"] = "unclassified"

    charge_mask = main_stage.isin(["charge", "charge_relaxation"])
    discharge_mask = main_stage.isin(["discharge", "discharge_relaxation"])
    charge = frame.loc[charge_mask].copy().reset_index().rename(columns={"index": "_source_index"})
    discharge = frame.loc[discharge_mask].copy().reset_index().rename(columns={"index": "_source_index"})

    charge_labels = classify_charge(charge, config) if not charge.empty else pd.Series(dtype=object)
    discharge_labels = classify_discharge(discharge, config) if not discharge.empty else pd.Series(dtype=object)

    for subset, labels in ((charge, charge_labels), (discharge, discharge_labels)):
        if not subset.empty:
            frame.loc[subset["_source_index"].to_numpy(), "_phase"] = labels.to_numpy()

    for stage, subset, labels, phases in (
        ("charge", charge, charge_labels, ("CC", "CV", "relaxation")),
        ("discharge", discharge, discharge_labels, ("CC", "relaxation")),
    ):
        if subset.empty:
            continue
        stage_folder = cycle_folder / stage
        write_columns(subset, stage_folder / "all")
        for phase in phases:
            save_phase(subset, labels, phase, stage_folder / phase)

    cycle_folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cycle_folder / "cycle_with_phase_labels.csv", index=False)

    cv_detected = bool((charge_labels == "CV").any())
    cv_expected = float(cycle_index) <= config.cv_expected_until_cycle
    if cv_expected and cv_detected:
        cv_status = "CV_present_as_expected"
    elif cv_expected:
        cv_status = "WARNING_missing_expected_CV"
    elif cv_detected:
        cv_status = "WARNING_unexpected_CV"
    else:
        cv_status = "CC_only_as_expected"

    warnings_found = []
    if time_method != "original_Test_Time(s)":
        warnings_found.append("time_reconstructed")
    if cv_expected != cv_detected:
        warnings_found.append("CV_protocol_mismatch")
    if charge.empty:
        warnings_found.append("no_charge_detected")
    if discharge.empty:
        warnings_found.append("no_discharge_detected")

    return {
        "battery": battery_id,
        "cycle": cycle_index,
        "status": "WARNING" if warnings_found else "OK",
        "warnings": "; ".join(warnings_found),
        "number_of_samples": len(frame),
        "time_valid_samples": int(frame["Test_Time(s)"].notna().sum()),
        "time_method": time_method,
        "charge_samples": len(charge),
        "discharge_samples": len(discharge),
        "charge_CC_samples": int((charge_labels == "CC").sum()),
        "charge_CV_samples": int((charge_labels == "CV").sum()),
        "charge_relaxation_samples": int((charge_labels == "relaxation").sum()),
        "discharge_CC_samples": int((discharge_labels == "CC").sum()),
        "discharge_relaxation_samples": int((discharge_labels == "relaxation").sum()),
        "CV_expected": cv_expected,
        "CV_detected": cv_detected,
        "CV_status": cv_status,
        "charge_sign": charge_sign,
    }


def process_dataset(
    input_dataset: str | Path,
    output_dataset: str | Path,
    config: SplitConfig = SplitConfig(),
    overwrite: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Process all B*.xlsx workbooks and write a complete audit table."""
    config.validate()
    source = Path(input_dataset)
    destination = Path(output_dataset)
    if not source.is_dir():
        raise NotADirectoryError(source)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    workbooks = sorted(source.glob("B*.xlsx"))
    if not workbooks:
        raise FileNotFoundError(f"No B*.xlsx workbooks found in {source}")

    audit: list[dict[str, object]] = []
    for workbook in workbooks:
        battery_id = workbook.stem
        try:
            frame = pd.read_excel(workbook)
        except Exception as error:
            audit.append({"battery": battery_id, "cycle": "ALL", "status": "ERROR", "message": str(error)})
            continue
        if "Cycle_Index" not in frame:
            audit.append({"battery": battery_id, "cycle": "ALL", "status": "ERROR", "message": "Missing Cycle_Index"})
            continue

        cycle_numbers = numeric(frame["Cycle_Index"])
        for cycle_index in sorted(cycle_numbers.dropna().unique()):
            cycle_frame = frame.loc[cycle_numbers == cycle_index].copy()
            try:
                record = process_cycle(
                    cycle_frame,
                    battery_id,
                    float(cycle_index),
                    destination / battery_id / cycle_name(cycle_index),
                    config,
                )
            except Exception as error:
                record = {
                    "battery": battery_id,
                    "cycle": cycle_index,
                    "status": "ERROR",
                    "warnings": "processing_error",
                    "message": str(error),
                }
            audit.append(record)
            if verbose:
                print(f"{battery_id} cycle {cycle_index}: {record['status']}")

    audit_frame = pd.DataFrame(audit)
    audit_frame.to_csv(destination / "splitting_audit.csv", index=False)
    return audit_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dataset", type=Path)
    parser.add_argument("output_dataset", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--rest-threshold", type=float, default=0.05)
    parser.add_argument("--cc-tolerance", type=float, default=0.10)
    parser.add_argument("--cv-voltage-tolerance", type=float, default=0.01)
    parser.add_argument("--minimum-phase-samples", type=int, default=3)
    parser.add_argument("--rolling-window", type=int, default=5)
    parser.add_argument("--cv-expected-until", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SplitConfig(
        rest_current_threshold_a=args.rest_threshold,
        cc_current_tolerance_a=args.cc_tolerance,
        cv_voltage_tolerance_v=args.cv_voltage_tolerance,
        minimum_phase_samples=args.minimum_phase_samples,
        rolling_window=args.rolling_window,
        cv_expected_until_cycle=args.cv_expected_until,
    )
    audit = process_dataset(
        args.input_dataset,
        args.output_dataset,
        config=config,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
    print(audit["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
