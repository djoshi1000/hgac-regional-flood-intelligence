from __future__ import annotations

import numpy as np
import pandas as pd


def apply_stage_controls(section_df, controls):
    """Assimilate multiple stage controls into a section WSE profile.

    controls is an optional list of dictionaries with:
      station_m, stage_ft_navd88, and optionally gauge_id/network.

    The base v4 hydraulic screening profile is corrected by interpolating the
    model-minus-control residual along river station. With one pilot gauge this
    function is not needed; it exists so future USGS/HCFCD gauges can constrain
    the longitudinal WSE rather than requiring a redesign.
    """
    if not controls:
        return section_df.copy(), {"stage_controls_used": 0, "stage_control_ids": []}

    df = section_df.copy().sort_values("station_m").reset_index(drop=True)
    st = df["station_m"].to_numpy(float)
    wse = df["scenario_wse_ft_navd88"].to_numpy(float)

    c_st = []
    residual = []
    ids = []
    for i, c in enumerate(controls):
        try:
            cs = float(c["station_m"])
            ch = float(c["stage_ft_navd88"])
        except Exception:
            continue
        if not np.isfinite(cs) or not np.isfinite(ch):
            continue
        model_h = float(np.interp(cs, st, wse))
        c_st.append(cs)
        residual.append(ch - model_h)
        ids.append(str(c.get("gauge_id", f"control_{i+1}")))

    if not c_st:
        return df, {"stage_controls_used": 0, "stage_control_ids": []}

    order = np.argsort(c_st)
    c_st = np.asarray(c_st, dtype=float)[order]
    residual = np.asarray(residual, dtype=float)[order]
    ids = [ids[i] for i in order]

    if len(c_st) == 1:
        # One additional control is applied as a local/global residual. In a
        # multi-control network np.interp transitions smoothly between gauges.
        correction = np.full_like(st, residual[0], dtype=float)
    else:
        correction = np.interp(st, c_st, residual)

    df["pre_control_wse_ft_navd88"] = df["scenario_wse_ft_navd88"]
    df["stage_control_correction_ft"] = correction
    df["scenario_wse_ft_navd88"] = df["scenario_wse_ft_navd88"] + correction
    df["scenario_wse_m_navd88"] = df["scenario_wse_ft_navd88"] * 0.3048
    df["local_overtop_ft"] = np.maximum(
        df["scenario_wse_ft_navd88"].to_numpy(float)
        - df["lower_bank_final_ft_navd88"].to_numpy(float),
        0.0,
    )
    df["local_overtop_m"] = df["local_overtop_ft"] * 0.3048
    df["overtopped"] = df["local_overtop_ft"] > 0

    return df, {
        "stage_controls_used": int(len(c_st)),
        "stage_control_ids": ids,
        "max_abs_stage_control_correction_ft": float(np.max(np.abs(correction))),
    }
