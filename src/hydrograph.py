from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

CFS_TO_CMS = 0.028316846592


@dataclass
class HydrographVolumeQA:
    source: str
    point_count: int
    start_time_utc: str | None
    end_time_utc: str | None
    horizon_hours: float
    excess_volume_m3: float
    peak_flow_cfs: float
    bankfull_flow_cfs: float
    intervals_used: int
    intervals_skipped_large_gap: int
    exceedance_interval_hours: float
    max_interval_hours: float

    def to_dict(self):
        return asdict(self)


def _normalize_hydrograph(hydrograph, flow_unit="cfs") -> pd.DataFrame:
    """Return clean UTC time + flow_cfs rows from a dataframe/list-like input."""
    if hydrograph is None:
        return pd.DataFrame(columns=["time", "flow_cfs"])

    if isinstance(hydrograph, pd.DataFrame):
        df = hydrograph.copy()
    else:
        df = pd.DataFrame(hydrograph)

    if df.empty:
        return pd.DataFrame(columns=["time", "flow_cfs"])

    time_col = next((c for c in ["time", "valid_time", "datetime", "forecast_time"] if c in df.columns), None)
    flow_col = next((c for c in ["flow_cfs", "streamflow", "flow", "discharge", "value"] if c in df.columns), None)
    if time_col is None or flow_col is None:
        raise ValueError("Hydrograph must contain a time column and a flow/streamflow column.")

    out = pd.DataFrame()
    out["time"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    out["flow_cfs"] = pd.to_numeric(df[flow_col], errors="coerce")

    unit = str(flow_unit).strip().lower()
    if unit in {"cms", "m3/s", "m^3/s", "m3s"}:
        out["flow_cfs"] = out["flow_cfs"] / CFS_TO_CMS
    elif unit not in {"cfs", "ft3/s", "ft^3/s", "ft3s"}:
        raise ValueError(f"Unsupported hydrograph flow unit: {flow_unit}")

    out = out.dropna(subset=["time", "flow_cfs"])
    out = out[np.isfinite(out["flow_cfs"].to_numpy(dtype=float, copy=True))]
    out = out.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    return out


def integrate_excess_hydrograph(
    hydrograph,
    bankfull_flow_cfs,
    flow_unit="cfs",
    max_interval_hours=3.0,
):
    """Integrate only discharge above bankfull using the actual forecast hydrograph.

    The integration is piecewise trapezoidal. Intervals larger than
    ``max_interval_hours`` are skipped rather than implicitly assuming a long
    linear bridge across missing forecast records.
    """
    df = _normalize_hydrograph(hydrograph, flow_unit=flow_unit)
    qbf = max(float(bankfull_flow_cfs), 0.0)
    max_gap_h = max(float(max_interval_hours), 0.01)

    if len(df) < 2:
        qa = HydrographVolumeQA(
            source="integrated_hydrograph",
            point_count=int(len(df)),
            start_time_utc=(df.iloc[0]["time"].isoformat() if len(df) else None),
            end_time_utc=(df.iloc[-1]["time"].isoformat() if len(df) else None),
            horizon_hours=0.0,
            excess_volume_m3=0.0,
            peak_flow_cfs=float(df["flow_cfs"].max()) if len(df) else 0.0,
            bankfull_flow_cfs=qbf,
            intervals_used=0,
            intervals_skipped_large_gap=0,
            exceedance_interval_hours=0.0,
            max_interval_hours=max_gap_h,
        )
        return qa.to_dict()

    # Use elapsed seconds explicitly. Do not assume pandas stores datetime
    # integers in nanoseconds; newer pandas can preserve another resolution.
    t0 = df["time"].iloc[0]
    t = (
        (df["time"] - t0)
        .dt.total_seconds()
        .to_numpy(dtype="float64", copy=True)
    )
    q = df["flow_cfs"].to_numpy(dtype="float64", copy=True)
    excess_cms = np.maximum(q - qbf, 0.0) * CFS_TO_CMS

    dt = np.diff(t)
    valid_dt = np.isfinite(dt) & (dt > 0.0) & (dt <= max_gap_h * 3600.0)
    interval_volume = 0.5 * (excess_cms[:-1] + excess_cms[1:]) * dt
    used_volume = float(np.sum(interval_volume[valid_dt])) if valid_dt.any() else 0.0

    exceed_mask = valid_dt & ((excess_cms[:-1] > 0.0) | (excess_cms[1:] > 0.0))
    exceed_hours = float(np.sum(dt[exceed_mask]) / 3600.0) if exceed_mask.any() else 0.0

    horizon_hours = float((t[-1] - t[0]) / 3600.0)
    qa = HydrographVolumeQA(
        source="integrated_hydrograph",
        point_count=int(len(df)),
        start_time_utc=df.iloc[0]["time"].isoformat(),
        end_time_utc=df.iloc[-1]["time"].isoformat(),
        horizon_hours=horizon_hours,
        excess_volume_m3=max(used_volume, 0.0),
        peak_flow_cfs=float(np.nanmax(q)),
        bankfull_flow_cfs=qbf,
        intervals_used=int(valid_dt.sum()),
        intervals_skipped_large_gap=int((np.isfinite(dt) & (dt > max_gap_h * 3600.0)).sum()),
        exceedance_interval_hours=exceed_hours,
        max_interval_hours=max_gap_h,
    )
    return qa.to_dict()
