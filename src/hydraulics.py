from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

FT_TO_M = 0.3048
M_TO_FT = 3.280839895013123


@dataclass
class GaugeControl:
    """Future-ready gauge control point for USGS/HCFCD/NWS controls."""

    gauge_id: str
    station_m: float
    flow_cfs: float
    stage_ft_navd88: float
    bank_ft_navd88: float
    upstream_area_km2: float | None = None
    network: str = "USGS"


def wetted_geometry(distances_m, elevations_m, stage_m, left_limit_m=None, right_limit_m=None):
    """Approximate cross-section wetted geometry from the LiDAR terrain surface.

    This is a relative geometry proxy only; airborne LiDAR is not channel
    bathymetry. V4.1 therefore uses it to perturb a drainage-area-scaled
    capacity expectation rather than allowing it to determine absolute capacity
    without bounds.
    """

    x = np.asarray(distances_m, dtype=float)
    z = np.asarray(elevations_m, dtype=float)
    valid = np.isfinite(x) & np.isfinite(z)
    if left_limit_m is not None:
        valid &= x >= float(left_limit_m)
    if right_limit_m is not None:
        valid &= x <= float(right_limit_m)
    x = x[valid]
    z = z[valid]
    if x.size < 3:
        return {
            "area_m2": np.nan,
            "wetted_perimeter_m": np.nan,
            "hydraulic_radius_m": np.nan,
            "top_width_m": np.nan,
        }

    order = np.argsort(x)
    x = x[order]
    z = z[order]
    d = np.maximum(float(stage_m) - z, 0.0)

    area = float(np.trapezoid(d, x))

    dx = np.diff(x)
    dz = np.diff(z)
    mid_z = 0.5 * (z[:-1] + z[1:])
    wet_seg = mid_z < float(stage_m)
    perimeter = float(np.sum(np.sqrt(dx[wet_seg] ** 2 + dz[wet_seg] ** 2)))

    wet = d > 0
    top_width = float(x[wet].max() - x[wet].min()) if wet.any() else 0.0
    radius = area / perimeter if area > 0 and perimeter > 0 else np.nan
    return {
        "area_m2": area,
        "wetted_perimeter_m": perimeter,
        "hydraulic_radius_m": radius,
        "top_width_m": top_width,
    }


def conveyance_index(area_m2, hydraulic_radius_m, slope):
    """Manning conveyance index without 1/n: A * R^(2/3) * S^(1/2)."""

    a = float(area_m2)
    r = float(hydraulic_radius_m)
    s = float(slope)
    if not (np.isfinite(a) and np.isfinite(r) and np.isfinite(s)):
        return np.nan
    if a <= 0 or r <= 0 or s <= 0:
        return np.nan
    return float(a * (r ** (2.0 / 3.0)) * math.sqrt(s))


def local_flow_profile(gauge_flow_cfs, local_area_km2, gauge_area_km2, exponent=0.85):
    """Drainage-area-scaled mainstem discharge screening profile."""

    qg = max(float(gauge_flow_cfs), 0.0)
    ag = max(float(gauge_area_km2), 1e-6)
    a = np.asarray(local_area_km2, dtype=float)
    ratio = np.clip(a / ag, 0.0, 1.5)
    return qg * np.power(ratio, float(exponent))


def expected_bankfull_capacity_profile(
    gauge_bankfull_flow_cfs,
    local_area_km2,
    gauge_area_km2,
    exponent=0.85,
):
    """Drainage-area-scaled expected bankfull-capacity profile.

    This is not a regional flood-frequency regression. It is a stabilizing
    screening prior anchored to the local rating-curve bankfull proxy at the
    gauge. LiDAR geometry is allowed to adjust this expectation within bounded
    factors in ``robust_capacity_profile``.
    """

    return local_flow_profile(
        gauge_flow_cfs=float(gauge_bankfull_flow_cfs),
        local_area_km2=local_area_km2,
        gauge_area_km2=float(gauge_area_km2),
        exponent=float(exponent),
    )


def robust_capacity_profile(
    bank_df,
    gauge_bankfull_flow_cfs,
    gauge_area_km2,
    area_exponent=0.85,
    outlier_factor=3.0,
    capacity_floor_factor=1.00,
    capacity_ceiling_factor=1.50,
):
    """Build a stabilized local bankfull-capacity profile.

    V4 allowed imperfect LiDAR geometry/slope to make local capacity collapse to
    unrealistically low values. In V4.1:

    1. the gauge rating curve anchors absolute bankfull capacity;
    2. drainage-area scaling provides a longitudinal expected-capacity prior;
    3. LiDAR conveyance supplies *relative* geometry information;
    4. geometry-derived capacity is clipped to configurable bounds around the
       expected capacity;
    5. the gauge section is preserved exactly.

    The unconstrained geometry capacity remains in the dataframe as QA.
    """

    df = bank_df.copy().sort_values("station_m").reset_index(drop=True)
    k = pd.to_numeric(
        df.get("bankfull_conveyance_index"), errors="coerce"
    ).to_numpy(dtype=float, copy=True)
    station = df["station_m"].to_numpy(dtype=float, copy=True)
    local_area = pd.to_numeric(
        df["upstream_area_km2"], errors="coerce"
    ).to_numpy(dtype=float, copy=True)

    valid = np.isfinite(k) & (k > 0)
    if not valid.any():
        raise RuntimeError("No valid cross-section conveyance indices were produced.")

    if valid[0]:
        k_ref = float(k[0])
        calibration_section = int(df.iloc[0].get("section_id", 0))
    else:
        idx = np.where(valid)[0][:3]
        if len(idx) == 0:
            raise RuntimeError("Unable to establish a hydraulic calibration section.")
        k_ref = float(np.nanmedian(k[idx]))
        calibration_section = int(df.iloc[idx[0]].get("section_id", idx[0]))

    qbf_g = float(gauge_bankfull_flow_cfs)
    if qbf_g <= 0:
        raise RuntimeError("Gauge bankfull flow must be positive for V4.2 calibration.")

    factor = qbf_g / k_ref
    raw_capacity = k * factor

    # First remove abrupt geometry spikes/dips longitudinally.
    s = pd.Series(raw_capacity)
    med = s.rolling(window=5, center=True, min_periods=1).median().to_numpy(
        dtype=float, copy=True
    )
    ratio = np.full(raw_capacity.shape, np.nan, dtype=float)
    ok_med = np.isfinite(med) & (med > 0) & np.isfinite(raw_capacity) & (raw_capacity > 0)
    ratio[ok_med] = raw_capacity[ok_med] / med[ok_med]
    f = max(float(outlier_factor), 1.01)
    profile_outlier = ok_med & ((ratio > f) | (ratio < 1.0 / f))

    use = valid & ~profile_outlier
    if use.sum() < 2:
        use = valid
    if use.sum() < 2:
        geometry_smoothed = np.array(raw_capacity, dtype=float, copy=True)
    else:
        geometry_smoothed = np.interp(station, station[use], raw_capacity[use])
        geometry_smoothed = (
            pd.Series(geometry_smoothed)
            .rolling(window=3, center=True, min_periods=1)
            .median()
            .to_numpy(dtype=float, copy=True)
        )

    expected = expected_bankfull_capacity_profile(
        gauge_bankfull_flow_cfs=qbf_g,
        local_area_km2=local_area,
        gauge_area_km2=float(gauge_area_km2),
        exponent=float(area_exponent),
    )

    floor_factor = max(float(capacity_floor_factor), 0.05)
    ceiling_factor = max(float(capacity_ceiling_factor), floor_factor + 0.01)
    lower = expected * floor_factor
    upper = expected * ceiling_factor

    final_capacity = np.array(geometry_smoothed, dtype=float, copy=True)
    finite_expected = np.isfinite(expected) & (expected > 0)
    finite_geometry = np.isfinite(final_capacity) & (final_capacity > 0)

    # Where geometry is invalid, use the expected profile directly.
    final_capacity[finite_expected & ~finite_geometry] = expected[finite_expected & ~finite_geometry]
    constrained_low = finite_expected & np.isfinite(final_capacity) & (final_capacity < lower)
    constrained_high = finite_expected & np.isfinite(final_capacity) & (final_capacity > upper)
    final_capacity[constrained_low] = lower[constrained_low]
    final_capacity[constrained_high] = upper[constrained_high]

    # Preserve rating-curve capacity at the gauge exactly.
    final_capacity = np.array(final_capacity, dtype=float, copy=True)
    expected = np.array(expected, dtype=float, copy=True)
    final_capacity[0] = qbf_g
    expected[0] = qbf_g

    df["bankfull_capacity_raw_cfs"] = raw_capacity
    df["bankfull_capacity_geometry_smoothed_cfs"] = geometry_smoothed
    df["expected_bankfull_capacity_cfs"] = expected
    df["capacity_profile_outlier"] = profile_outlier
    df["capacity_constrained_low"] = constrained_low
    df["capacity_constrained_high"] = constrained_high
    df["bankfull_capacity_cfs"] = final_capacity
    df["hydraulic_calibration_factor"] = factor

    return df, {
        "calibration_factor": float(factor),
        "calibration_section_id": int(calibration_section),
        "capacity_outlier_sections": int(profile_outlier.sum()),
        "capacity_constrained_low_sections": int(constrained_low.sum()),
        "capacity_constrained_high_sections": int(constrained_high.sum()),
        "valid_conveyance_sections": int(valid.sum()),
        "capacity_floor_factor": float(floor_factor),
        "capacity_ceiling_factor": float(ceiling_factor),
        "capacity_area_exponent": float(area_exponent),
    }


def distribute_section_wse(
    bank_df,
    gauge_flow_cfs,
    gauge_stage_ft,
    gauge_bankfull_flow_cfs,
    gauge_bank_ft,
    gauge_channel_proxy_ft,
    gauge_area_km2,
    flow_area_exponent=0.85,
    capacity_area_exponent=0.70,
    contained_stage_exponent=0.60,
    overbank_excess_exponent=0.70,
    capacity_outlier_factor=3.0,
    capacity_floor_factor=1.00,
    capacity_ceiling_factor=1.50,
    max_local_overbank_ft=12.0,
    max_local_overbank_multiplier=1.5,
):
    """Build a stable section-specific flow/WSE screening profile.

    Important V4.1 corrections:
      * local capacities are stabilized around drainage-area-scaled expectations;
      * the unstable ``(Q/Qbf - 1)`` ratio-of-ratios amplification is removed;
      * local overbank depth is scaled by the *full* local/gauge capacity ratio;
      * local excess cannot exceed a configurable multiple of the gauge excess.
    """

    df, cal = robust_capacity_profile(
        bank_df,
        gauge_bankfull_flow_cfs=float(gauge_bankfull_flow_cfs),
        gauge_area_km2=float(gauge_area_km2),
        area_exponent=float(capacity_area_exponent),
        outlier_factor=float(capacity_outlier_factor),
        capacity_floor_factor=float(capacity_floor_factor),
        capacity_ceiling_factor=float(capacity_ceiling_factor),
    )

    local_area = pd.to_numeric(df["upstream_area_km2"], errors="coerce").to_numpy(
        dtype=float, copy=True
    )
    q_local = local_flow_profile(
        gauge_flow_cfs=float(gauge_flow_cfs),
        local_area_km2=local_area,
        gauge_area_km2=float(gauge_area_km2),
        exponent=float(flow_area_exponent),
    )
    q_local = np.array(q_local, dtype=float, copy=True)
    qbf = pd.to_numeric(df["bankfull_capacity_cfs"], errors="coerce").to_numpy(
        dtype=float, copy=True
    )
    bank_ft = pd.to_numeric(df["lower_bank_final_ft_navd88"], errors="coerce").to_numpy(
        dtype=float, copy=True
    )
    channel_ft = pd.to_numeric(
        df["routing_channel_elevation_ft_navd88"], errors="coerce"
    ).to_numpy(dtype=float, copy=True)

    qratio = np.divide(
        q_local,
        qbf,
        out=np.zeros_like(q_local),
        where=np.isfinite(qbf) & (qbf > 0),
    )

    qg = max(float(gauge_flow_cfs), 0.0)
    qbf_g = max(float(gauge_bankfull_flow_cfs), 1e-6)
    gauge_ratio = qg / qbf_g
    gauge_excess_ft = float(gauge_stage_ft) - float(gauge_bank_ft)

    contained_exp = float(contained_stage_exponent)
    gauge_freeboard = float(gauge_bank_ft) - float(gauge_channel_proxy_ft)
    if gauge_ratio > 0 and gauge_ratio < 1 and gauge_freeboard > 0:
        gauge_fraction = (
            float(gauge_stage_ft) - float(gauge_channel_proxy_ft)
        ) / gauge_freeboard
        if 0.01 < gauge_fraction < 0.99:
            try:
                p = math.log(gauge_fraction) / math.log(gauge_ratio)
                if np.isfinite(p):
                    contained_exp = float(np.clip(p, 0.15, 2.5))
            except Exception:
                pass

    wse_ft = np.full(len(df), np.nan, dtype=float)
    overtopped = np.zeros(len(df), dtype=bool)
    overtop_ft = np.zeros(len(df), dtype=float)

    max_multiplier = max(float(max_local_overbank_multiplier), 1.0)
    absolute_cap = max(float(max_local_overbank_ft), 0.0)
    if gauge_excess_ft > 0:
        scenario_cap = min(absolute_cap, gauge_excess_ft * max_multiplier)
    else:
        scenario_cap = 0.0

    for i in range(len(df)):
        if not (
            np.isfinite(bank_ft[i])
            and np.isfinite(channel_ft[i])
            and np.isfinite(qratio[i])
        ):
            continue

        if qratio[i] <= 1.0 or gauge_excess_ft <= 0.0 or gauge_ratio <= 1.0:
            frac = np.clip(qratio[i], 0.0, 1.0) ** contained_exp
            wse_ft[i] = channel_ft[i] + frac * max(bank_ft[i] - channel_ft[i], 0.0)
            overtop_ft[i] = 0.0
            overtopped[i] = False
        else:
            # Stable full-ratio scaling. A section at the same Q/Qbf ratio as the
            # gauge receives the same excess. Moderate relative differences only
            # modestly perturb the gauge excess.
            ratio_scale = max(qratio[i] / max(gauge_ratio, 1e-9), 0.0)
            excess = float(gauge_excess_ft) * (
                ratio_scale ** float(overbank_excess_exponent)
            )
            excess = float(np.clip(excess, 0.0, scenario_cap))
            wse_ft[i] = bank_ft[i] + excess
            overtop_ft[i] = excess
            overtopped[i] = excess > 0

    # Anchor section zero exactly to the gauge rating-curve/observed scenario.
    if len(df):
        wse_ft[0] = float(gauge_stage_ft)
        overtop_ft[0] = max(0.0, float(gauge_stage_ft) - float(gauge_bank_ft))
        overtopped[0] = overtop_ft[0] > 0
        q_local[0] = float(gauge_flow_cfs)
        qbf[0] = float(gauge_bankfull_flow_cfs)
        qratio[0] = float(gauge_flow_cfs) / max(float(gauge_bankfull_flow_cfs), 1e-9)

    df["local_flow_cfs"] = q_local
    df["bankfull_capacity_cfs"] = qbf
    df["flow_to_capacity_ratio"] = qratio
    df["scenario_wse_ft_navd88"] = wse_ft
    df["scenario_wse_m_navd88"] = wse_ft * FT_TO_M
    df["local_overtop_ft"] = overtop_ft
    df["local_overtop_m"] = overtop_ft * FT_TO_M
    df["overtopped"] = overtopped

    cal.update(
        {
            "flow_area_exponent": float(flow_area_exponent),
            "capacity_area_exponent_used": float(capacity_area_exponent),
            "contained_stage_exponent_used": float(contained_exp),
            "overbank_excess_exponent": float(overbank_excess_exponent),
            "gauge_flow_to_capacity_ratio": float(gauge_ratio),
            "gauge_overbank_excess_ft": float(max(gauge_excess_ft, 0.0)),
            "max_local_overbank_multiplier": float(max_multiplier),
            "scenario_local_overbank_cap_ft": float(scenario_cap),
        }
    )
    return df, cal
