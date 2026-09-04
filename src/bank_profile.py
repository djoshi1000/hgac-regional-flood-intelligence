from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyflwdir
import rasterio
from pyproj import Transformer
from rasterio.transform import xy
from scipy import ndimage
from scipy.signal import find_peaks, savgol_filter
from shapely.geometry import LineString, Point

from src.hydraulics import wetted_geometry, conveyance_index


FLOAT_NODATA = -9999.0
M_TO_FT = 3.280839895013123
FT_TO_M = 0.3048


def _cell_center(transform, row, col):
    x = transform.c + (col + 0.5) * transform.a + (row + 0.5) * transform.b
    y = transform.f + (col + 0.5) * transform.d + (row + 0.5) * transform.e
    return float(x), float(y)


def _write_raster(path, array, profile, dtype="float32", nodata=FLOAT_NODATA):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(
        dtype=dtype,
        count=1,
        nodata=nodata,
        compress="deflate",
        predictor=2 if np.issubdtype(np.dtype(dtype), np.floating) else 1,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array.astype(dtype), 1)
    return path


def _trace_ordered_mainstem(flw, mainstem, outlet_idx):
    """Return flat mainstem indices ordered from gauge outlet upstream."""
    nrows, ncols = mainstem.shape
    n = nrows * ncols
    main_flat = np.asarray(mainstem).ravel()
    idxs_us_main = np.asarray(flw.idxs_us_main)

    path = []
    idx = int(outlet_idx)
    seen = set()

    while 0 <= idx < n and idx not in seen and main_flat[idx]:
        path.append(idx)
        seen.add(idx)
        nxt = int(idxs_us_main[idx])
        if nxt < 0 or nxt >= n or nxt == idx or not main_flat[nxt]:
            break
        idx = nxt

    if len(path) < 10:
        raise RuntimeError(
            "Ordered mainstem trace is unexpectedly short. Re-run corrected terrain QA."
        )
    return np.asarray(path, dtype=np.int64)


def _path_xy_station(path_idxs, shape, transform):
    rr, cc = np.unravel_index(path_idxs, shape)
    xs = transform.c + (cc + 0.5) * transform.a + (rr + 0.5) * transform.b
    ys = transform.f + (cc + 0.5) * transform.d + (rr + 0.5) * transform.e
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    station = np.zeros(len(xs), dtype=float)
    if len(xs) > 1:
        station[1:] = np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))
    return rr, cc, xs, ys, station


def _tangent_for_index(xs, ys, station, i, window_m=30.0):
    s = station[i]
    i0 = int(np.searchsorted(station, max(0.0, s - window_m), side="left"))
    i1 = int(np.searchsorted(station, min(station[-1], s + window_m), side="right") - 1)
    i0 = max(0, min(i0, len(xs) - 1))
    i1 = max(0, min(i1, len(xs) - 1))

    if i1 == i0:
        i0 = max(0, i - 2)
        i1 = min(len(xs) - 1, i + 2)

    dx = float(xs[i1] - xs[i0])
    dy = float(ys[i1] - ys[i0])
    norm = float(np.hypot(dx, dy))
    if norm <= 0:
        raise RuntimeError("Could not determine local mainstem direction.")

    tangent = np.array([dx / norm, dy / norm], dtype=float)
    cross = np.array([-tangent[1], tangent[0]], dtype=float)
    return tangent, cross


def _smooth_profile(elevation_ft):
    filled = (
        pd.Series(elevation_ft)
        .interpolate(limit_direction="both")
        .to_numpy(dtype=float)
    )
    if len(filled) < 5:
        return filled
    window = min(9, len(filled) if len(filled) % 2 == 1 else len(filled) - 1)
    window = max(window, 5)
    if window >= len(filled):
        window = len(filled) - 1 if len(filled) % 2 == 0 else len(filled)
    if window < 5:
        return filled
    return savgol_filter(filled, window_length=window, polyorder=2)


def _find_first_bank(
    distances,
    smooth_ft,
    channel_index,
    side,
    bank_search_min_m=8.0,
    bank_search_max_m=65.0,
    min_relief_ft=6.5,
    peak_prominence_ft=0.35,
    sample_spacing_m=1.0,
):
    relative = distances - distances[channel_index]

    if side == "left":
        region = (relative <= -bank_search_min_m) & (relative >= -bank_search_max_m)
    else:
        region = (relative >= bank_search_min_m) & (relative <= bank_search_max_m)

    idx = np.where(region)[0]
    if len(idx) < 5:
        raise RuntimeError(f"Too few samples in {side} bank-search region.")

    idx_spatial = np.sort(idx)
    prof = smooth_ft[idx_spatial]
    peaks, _ = find_peaks(
        prof,
        prominence=float(peak_prominence_ft),
        distance=max(3, int(round(5.0 / max(sample_spacing_m, 0.1)))),
    )

    candidates = []
    for p in peaks:
        gi = int(idx_spatial[p])
        relief = float(smooth_ft[gi] - smooth_ft[channel_index])
        if relief >= float(min_relief_ft):
            candidates.append(gi)

    if candidates:
        return min(candidates, key=lambda j: abs(relative[j]))

    gradients = np.gradient(smooth_ft, distances)
    ordered = idx[np.argsort(np.abs(relative[idx]))]
    for gi in ordered:
        relief = float(smooth_ft[gi] - smooth_ft[channel_index])
        if relief >= float(min_relief_ft) and abs(float(gradients[gi])) <= 0.08:
            return int(gi)

    # Last-resort fallback is deliberately restricted to the near-bank window.
    return int(idx[np.argmax(smooth_ft[idx])])


def extract_cross_section(
    dem_src,
    center_x,
    center_y,
    cross_vector,
    routing_channel_elevation_m,
    station_m,
    half_width_m=150.0,
    sample_spacing_m=1.0,
    channel_search_half_width_m=35.0,
    bank_search_min_m=8.0,
    bank_search_max_m=65.0,
    min_relief_ft=6.5,
    peak_prominence_ft=0.35,
    seed_offset_m=4.0,
    min_channel_slope=1.0e-5,
    max_channel_slope=0.02,
):
    distances = np.arange(
        -float(half_width_m),
        float(half_width_m) + float(sample_spacing_m),
        float(sample_spacing_m),
    )
    sx = center_x + distances * cross_vector[0]
    sy = center_y + distances * cross_vector[1]
    coords = list(zip(sx, sy))

    vals = np.asarray([float(v[0]) for v in dem_src.sample(coords)], dtype=float)
    if dem_src.nodata is not None:
        vals[vals == dem_src.nodata] = np.nan
    elev_m = vals
    elev_ft = elev_m * M_TO_FT

    if np.isfinite(elev_ft).sum() < 15:
        raise RuntimeError("Too few valid LiDAR samples on cross section.")

    smooth_ft = _smooth_profile(elev_ft)
    central = (np.abs(distances) <= float(channel_search_half_width_m)) & np.isfinite(elev_ft)
    central_idx = np.where(central)[0]
    if len(central_idx) == 0:
        raise RuntimeError("No valid LiDAR samples in channel-search window.")

    channel_idx = int(central_idx[np.argmin(smooth_ft[central_idx])])
    channel_dist = float(distances[channel_idx])
    channel_surface_ft = float(elev_ft[channel_idx])

    left_idx = _find_first_bank(
        distances,
        smooth_ft,
        channel_idx,
        "left",
        bank_search_min_m,
        bank_search_max_m,
        min_relief_ft,
        peak_prominence_ft,
        sample_spacing_m,
    )
    right_idx = _find_first_bank(
        distances,
        smooth_ft,
        channel_idx,
        "right",
        bank_search_min_m,
        bank_search_max_m,
        min_relief_ft,
        peak_prominence_ft,
        sample_spacing_m,
    )

    left_ft = float(elev_ft[left_idx])
    right_ft = float(elev_ft[right_idx])
    left_dist = float(distances[left_idx])
    right_dist = float(distances[right_idx])

    if left_ft <= right_ft:
        lower_side = "left"
        lower_sign = -1.0
        lower_idx = left_idx
    else:
        lower_side = "right"
        lower_sign = 1.0
        lower_idx = right_idx

    lower_raw_ft = float(elev_ft[lower_idx])
    lower_dist = float(distances[lower_idx])
    lower_x = float(sx[lower_idx])
    lower_y = float(sy[lower_idx])

    seed_x = float(lower_x + lower_sign * float(seed_offset_m) * cross_vector[0])
    seed_y = float(lower_y + lower_sign * float(seed_offset_m) * cross_vector[1])

    routing_ft = float(routing_channel_elevation_m) * M_TO_FT
    freeboard_raw_ft = lower_raw_ft - routing_ft

    flags = []
    if abs(channel_dist) > 15.0:
        flags.append("channel_low_far_from_mainstem_center")
    if abs(left_dist - channel_dist) > 70.0:
        flags.append("left_bank_far_from_channel")
    if abs(right_dist - channel_dist) > 70.0:
        flags.append("right_bank_far_from_channel")
    if abs(left_ft - right_ft) > 8.0:
        flags.append("bank_elevation_asymmetry_gt_8ft")
    if freeboard_raw_ft < 4.0:
        flags.append("lower_bank_relief_lt_4ft")
    if freeboard_raw_ft > 30.0:
        flags.append("lower_bank_relief_gt_30ft")

    return {
        "station_m": float(station_m),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "cross_dx": float(cross_vector[0]),
        "cross_dy": float(cross_vector[1]),
        "routing_channel_elevation_m_navd88": float(routing_channel_elevation_m),
        "routing_channel_elevation_ft_navd88": routing_ft,
        "channel_surface_elevation_m_navd88": channel_surface_ft * FT_TO_M,
        "channel_surface_elevation_ft_navd88": channel_surface_ft,
        "channel_distance_m": channel_dist,
        "left_bank_ft_navd88": left_ft,
        "left_bank_distance_m": left_dist,
        "right_bank_ft_navd88": right_ft,
        "right_bank_distance_m": right_dist,
        "lower_bank_raw_ft_navd88": lower_raw_ft,
        "lower_bank_raw_m_navd88": lower_raw_ft * FT_TO_M,
        "lower_bank_side": lower_side,
        "lower_bank_side_sign": lower_sign,
        "lower_bank_distance_m": lower_dist,
        "lower_bank_x": lower_x,
        "lower_bank_y": lower_y,
        "seed_x": seed_x,
        "seed_y": seed_y,
        "freeboard_raw_ft": freeboard_raw_ft,
        "qa_flags": flags,
        "raw_valid": len(flags) == 0,
        "line_x0": float(sx[0]),
        "line_y0": float(sy[0]),
        "line_x1": float(sx[-1]),
        "line_y1": float(sy[-1]),
        "distances": distances,
        "sample_x": sx,
        "sample_y": sy,
        "elevation_m": elev_m,
        "elevation_ft": elev_ft,
        "smooth_ft": smooth_ft,
    }


def _robust_finalize_profile(df, outlier_threshold_ft=4.0):
    df = df.copy().sort_values("station_m").reset_index(drop=True)
    raw = df["freeboard_raw_ft"].to_numpy(dtype=float)
    base_valid = df["raw_valid"].to_numpy(dtype=bool) & np.isfinite(raw)

    series = pd.Series(raw)
    rolling = series.rolling(window=5, center=True, min_periods=1).median().to_numpy()
    outlier = np.isfinite(raw) & np.isfinite(rolling) & (
        np.abs(raw - rolling) > float(outlier_threshold_ft)
    )

    # The gauge cross section is the anchor. Keep it unless it failed basic QA.
    if len(outlier) > 0 and base_valid[0]:
        outlier[0] = False

    use = base_valid & ~outlier
    if use.sum() < 2:
        # Fall back to all finite freeboards rather than inventing a profile.
        use = np.isfinite(raw)
    if use.sum() < 2:
        raise RuntimeError("Too few valid bank sections to create a longitudinal profile.")

    station = df["station_m"].to_numpy(dtype=float)
    interp = np.interp(station, station[use], raw[use])
    final_fb = ndimage.median_filter(interp, size=3, mode="nearest")
    if base_valid[0]:
        final_fb[0] = raw[0]

    routing_ft = df["routing_channel_elevation_ft_navd88"].to_numpy(dtype=float)
    final_bank_ft = routing_ft + final_fb

    df["profile_outlier"] = outlier
    df["profile_used_raw"] = use
    df["freeboard_final_ft"] = final_fb
    df["lower_bank_final_ft_navd88"] = final_bank_ft
    df["lower_bank_final_m_navd88"] = final_bank_ft * FT_TO_M
    return df


def build_bank_profile(
    hydraulic_dem_path,
    conditioned_dem_path,
    flowdir_path,
    mainstem_path,
    basin_mask_path,
    terrain_qa_path,
    upstream_area_path,
    processed_dir,
    section_spacing_m=150.0,
    half_width_m=150.0,
    sample_spacing_m=1.0,
    tangent_window_m=30.0,
    bank_search_min_m=8.0,
    bank_search_max_m=65.0,
    min_relief_ft=6.5,
    peak_prominence_ft=0.35,
    outlier_threshold_ft=4.0,
    seed_offset_m=4.0,
    min_channel_slope=1.0e-5,
    max_channel_slope=0.02,
):
    """Build local bank elevations and relative hydraulic geometry along the mainstem."""
    hydraulic_dem_path = Path(hydraulic_dem_path)
    conditioned_dem_path = Path(conditioned_dem_path)
    flowdir_path = Path(flowdir_path)
    mainstem_path = Path(mainstem_path)
    basin_mask_path = Path(basin_mask_path)
    terrain_qa_path = Path(terrain_qa_path)
    upstream_area_path = Path(upstream_area_path)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    for path in [
        hydraulic_dem_path,
        conditioned_dem_path,
        flowdir_path,
        mainstem_path,
        basin_mask_path,
        terrain_qa_path,
        upstream_area_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required bank-profile input missing: {path}")

    qa = json.loads(terrain_qa_path.read_text(encoding="utf-8"))

    with rasterio.open(flowdir_path) as d8_src, rasterio.open(
        mainstem_path
    ) as main_src, rasterio.open(basin_mask_path) as basin_src, rasterio.open(
        conditioned_dem_path
    ) as cond_src, rasterio.open(upstream_area_path) as upa_src, rasterio.open(hydraulic_dem_path) as hydro_src:
        d8 = d8_src.read(1)
        mainstem = main_src.read(1) > 0
        basin = basin_src.read(1) > 0
        transform = d8_src.transform
        crs = d8_src.crs
        profile = d8_src.profile.copy()
        conditioned = cond_src.read(1).astype(float)
        upstream_area = upa_src.read(1).astype(float)
        if upa_src.nodata is not None:
            upstream_area[upstream_area == upa_src.nodata] = np.nan

        flw = pyflwdir.from_array(
            d8,
            ftype="d8",
            mask=basin,
            transform=transform,
            latlon=bool(crs.is_geographic),
            cache=True,
        )

        outlet_row = int(qa["snapped_outlet_row"])
        outlet_col = int(qa["snapped_outlet_col"])
        outlet_idx = outlet_row * d8.shape[1] + outlet_col

        path_idxs = _trace_ordered_mainstem(flw, mainstem, outlet_idx)
        rr, cc, xs, ys, station = _path_xy_station(path_idxs, d8.shape, transform)

        targets = np.arange(0.0, station[-1] + float(section_spacing_m), float(section_spacing_m))
        targets = targets[targets <= station[-1] + 1e-6]
        if targets[-1] < station[-1] - 0.5 * float(section_spacing_m):
            targets = np.append(targets, station[-1])

        chosen = []
        for s in targets:
            i = int(np.argmin(np.abs(station - s)))
            if not chosen or i != chosen[-1]:
                chosen.append(i)

        records = []
        section_details = []

        for section_id, i in enumerate(chosen):
            tangent, cross = _tangent_for_index(
                xs, ys, station, i, window_m=float(tangent_window_m)
            )
            cx, cy = float(xs[i]), float(ys[i])
            routing_m = float(conditioned[rr[i], cc[i]])
            local_uparea_km2 = float(upstream_area[rr[i], cc[i]]) if np.isfinite(upstream_area[rr[i], cc[i]]) else np.nan

            try:
                rec = extract_cross_section(
                    hydro_src,
                    cx,
                    cy,
                    cross,
                    routing_m,
                    float(station[i]),
                    half_width_m=half_width_m,
                    sample_spacing_m=sample_spacing_m,
                    bank_search_min_m=bank_search_min_m,
                    bank_search_max_m=bank_search_max_m,
                    min_relief_ft=min_relief_ft,
                    peak_prominence_ft=peak_prominence_ft,
                    seed_offset_m=seed_offset_m,
                )
            except Exception as exc:
                # Keep a row so longitudinal interpolation can bridge a bad section.
                rec = {
                    "station_m": float(station[i]),
                    "center_x": cx,
                    "center_y": cy,
                    "cross_dx": float(cross[0]),
                    "cross_dy": float(cross[1]),
                    "routing_channel_elevation_m_navd88": routing_m,
                    "routing_channel_elevation_ft_navd88": routing_m * M_TO_FT,
                    "channel_surface_elevation_m_navd88": np.nan,
                    "channel_surface_elevation_ft_navd88": np.nan,
                    "channel_distance_m": np.nan,
                    "left_bank_ft_navd88": np.nan,
                    "left_bank_distance_m": np.nan,
                    "right_bank_ft_navd88": np.nan,
                    "right_bank_distance_m": np.nan,
                    "lower_bank_raw_ft_navd88": np.nan,
                    "lower_bank_raw_m_navd88": np.nan,
                    "lower_bank_side": "unknown",
                    "lower_bank_side_sign": np.nan,
                    "lower_bank_distance_m": np.nan,
                    "lower_bank_x": np.nan,
                    "lower_bank_y": np.nan,
                    "seed_x": np.nan,
                    "seed_y": np.nan,
                    "freeboard_raw_ft": np.nan,
                    "qa_flags": [f"cross_section_failed:{exc}"],
                    "raw_valid": False,
                    "line_x0": cx - half_width_m * cross[0],
                    "line_y0": cy - half_width_m * cross[1],
                    "line_x1": cx + half_width_m * cross[0],
                    "line_y1": cy + half_width_m * cross[1],
                }

            rec["section_id"] = int(section_id)
            rec["upstream_area_km2"] = local_uparea_km2
            rec["path_index"] = int(i)
            rec["tangent_dx"] = float(tangent[0])
            rec["tangent_dy"] = float(tangent[1])
            detail = rec.copy()
            for key in ["distances", "sample_x", "sample_y", "elevation_m", "elevation_ft", "smooth_ft"]:
                detail.pop(key, None)
            detail["qa_flags"] = ";".join(rec.get("qa_flags", []))
            records.append(detail)
            section_details.append(rec)

        df = pd.DataFrame(records)
        df = _robust_finalize_profile(df, outlier_threshold_ft=outlier_threshold_ft)

        # ----------------------------------------------------
        # V4 hydraulic screening attributes
        # ----------------------------------------------------
        # Local channel slope is derived from the conditioned mainstem profile.
        # Station increases upstream, so a positive dZ/dStation is expected.
        st = df["station_m"].to_numpy(float)
        ch = df["routing_channel_elevation_m_navd88"].to_numpy(float)
        if len(df) > 1:
            slope = np.gradient(ch, st, edge_order=1)
        else:
            slope = np.full(len(df), float(min_channel_slope), dtype=float)
        slope = np.abs(slope)
        slope = pd.Series(slope).rolling(window=5, center=True, min_periods=1).median().to_numpy(float)
        slope = np.clip(slope, float(min_channel_slope), float(max_channel_slope))
        df["local_channel_slope"] = slope

        sample_rows = []
        bankfull_area = []
        bankfull_perimeter = []
        bankfull_radius = []
        bankfull_width = []
        bankfull_k = []

        for j, rec in enumerate(section_details):
            if "distances" not in rec or "elevation_m" not in rec:
                bankfull_area.append(np.nan)
                bankfull_perimeter.append(np.nan)
                bankfull_radius.append(np.nan)
                bankfull_width.append(np.nan)
                bankfull_k.append(np.nan)
                continue

            dist = np.asarray(rec["distances"], dtype=float)
            elev = np.asarray(rec["elevation_m"], dtype=float)
            for dval, zval in zip(dist, elev):
                if np.isfinite(zval):
                    sample_rows.append({
                        "section_id": int(df.iloc[j]["section_id"]),
                        "station_m": float(df.iloc[j]["station_m"]),
                        "distance_m": float(dval),
                        "elevation_m_navd88": float(zval),
                        "elevation_ft_navd88": float(zval * M_TO_FT),
                    })

            left_lim = df.iloc[j].get("left_bank_distance_m", np.nan)
            right_lim = df.iloc[j].get("right_bank_distance_m", np.nan)
            if not np.isfinite(left_lim) or not np.isfinite(right_lim):
                left_lim = -float(bank_search_max_m)
                right_lim = float(bank_search_max_m)
            lo = min(float(left_lim), float(right_lim))
            hi = max(float(left_lim), float(right_lim))
            stage_m = float(df.iloc[j]["lower_bank_final_m_navd88"])
            geom = wetted_geometry(dist, elev, stage_m, lo, hi)
            k = conveyance_index(
                geom["area_m2"],
                geom["hydraulic_radius_m"],
                float(slope[j]),
            )
            bankfull_area.append(geom["area_m2"])
            bankfull_perimeter.append(geom["wetted_perimeter_m"])
            bankfull_radius.append(geom["hydraulic_radius_m"])
            bankfull_width.append(geom["top_width_m"])
            bankfull_k.append(k)

        df["bankfull_area_proxy_m2"] = bankfull_area
        df["bankfull_wetted_perimeter_proxy_m"] = bankfull_perimeter
        df["bankfull_hydraulic_radius_proxy_m"] = bankfull_radius
        df["bankfull_top_width_proxy_m"] = bankfull_width
        df["bankfull_conveyance_index"] = bankfull_k

        # Project section centers and bank points to WGS84 for easy dashboard use.
        to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        center_lon, center_lat = to_wgs.transform(
            df["center_x"].to_numpy(float), df["center_y"].to_numpy(float)
        )
        df["center_lon"] = center_lon
        df["center_lat"] = center_lat

        # Interpolate final local-bank profile to every mainstem cell.
        final_bank_m_sections = df["lower_bank_final_m_navd88"].to_numpy(float)
        main_bank_m = np.interp(station, df["station_m"], final_bank_m_sections)
        main_station = np.full(d8.shape, FLOAT_NODATA, dtype=float)
        main_bank = np.full(d8.shape, FLOAT_NODATA, dtype=float)
        main_channel = np.full(d8.shape, FLOAT_NODATA, dtype=float)
        for j, flat_idx in enumerate(path_idxs):
            r, c = np.unravel_index(int(flat_idx), d8.shape)
            main_station[r, c] = station[j]
            main_bank[r, c] = main_bank_m[j]
            main_channel[r, c] = conditioned[r, c]

        # Vector outputs.
        lines = []
        line_rows = []
        bank_points = []
        point_rows = []
        for _, row in df.iterrows():
            lines.append(
                LineString(
                    [
                        (float(row["line_x0"]), float(row["line_y0"])),
                        (float(row["line_x1"]), float(row["line_y1"])),
                    ]
                )
            )
            line_rows.append(
                {
                    "section_id": int(row["section_id"]),
                    "station_m": float(row["station_m"]),
                    "bank_ft": float(row["lower_bank_final_ft_navd88"]),
                    "raw_bank_ft": float(row["lower_bank_raw_ft_navd88"])
                    if np.isfinite(row["lower_bank_raw_ft_navd88"])
                    else None,
                    "side": str(row["lower_bank_side"]),
                    "used_raw": bool(row["profile_used_raw"]),
                }
            )
            if np.isfinite(row.get("lower_bank_x", np.nan)) and np.isfinite(
                row.get("lower_bank_y", np.nan)
            ):
                bank_points.append(Point(float(row["lower_bank_x"]), float(row["lower_bank_y"])))
                point_rows.append(
                    {
                        "section_id": int(row["section_id"]),
                        "station_m": float(row["station_m"]),
                        "bank_ft": float(row["lower_bank_final_ft_navd88"]),
                        "raw_bank_ft": float(row["lower_bank_raw_ft_navd88"]),
                        "side": str(row["lower_bank_side"]),
                    }
                )

        mainstem_line = LineString(list(zip(xs, ys)))

    # Write outside raster context.
    csv_path = processed_dir / "bank_profile.csv"
    sections_path = processed_dir / "bank_cross_sections.geojson"
    bank_points_path = processed_dir / "bank_profile_points.geojson"
    mainstem_geojson = processed_dir / "mainstem.geojson"
    bank_raster_path = processed_dir / "mainstem_bank_elevation_2m.tif"
    station_raster_path = processed_dir / "mainstem_station_2m.tif"
    channel_raster_path = processed_dir / "mainstem_channel_elevation_2m.tif"
    summary_path = processed_dir / "bank_profile_summary.json"
    samples_path = processed_dir / "bank_cross_section_samples.csv"

    df.to_csv(csv_path, index=False)
    pd.DataFrame(sample_rows).to_csv(samples_path, index=False)

    gpd.GeoDataFrame(line_rows, geometry=lines, crs=crs).to_crs(4326).to_file(
        sections_path, driver="GeoJSON"
    )
    if bank_points:
        gpd.GeoDataFrame(point_rows, geometry=bank_points, crs=crs).to_crs(4326).to_file(
            bank_points_path, driver="GeoJSON"
        )
    else:
        bank_points_path.write_text(
            json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8"
        )
    gpd.GeoDataFrame(
        {"name": ["Corrected configured-basin mainstem"]}, geometry=[mainstem_line], crs=crs
    ).to_crs(4326).to_file(mainstem_geojson, driver="GeoJSON")

    _write_raster(bank_raster_path, main_bank, profile)
    _write_raster(station_raster_path, main_station, profile)
    _write_raster(channel_raster_path, main_channel, profile)

    valid_raw = int(df["raw_valid"].sum())
    used_raw = int(df["profile_used_raw"].sum())
    outliers = int(df["profile_outlier"].sum())
    gauge_bank_ft = float(df.iloc[0]["lower_bank_final_ft_navd88"])

    summary = {
        "section_spacing_m": float(section_spacing_m),
        "section_count": int(len(df)),
        "raw_valid_sections": valid_raw,
        "raw_sections_used_after_outlier_filter": used_raw,
        "profile_outlier_sections": outliers,
        "mainstem_length_m": float(station[-1]),
        "gauge_bank_stage_ft_navd88": gauge_bank_ft,
        "gauge_routing_channel_elevation_ft_navd88": float(
            df.iloc[0]["routing_channel_elevation_ft_navd88"]
        ),
        "median_final_freeboard_ft": float(np.nanmedian(df["freeboard_final_ft"])),
        "min_final_bank_ft_navd88": float(np.nanmin(df["lower_bank_final_ft_navd88"])),
        "max_final_bank_ft_navd88": float(np.nanmax(df["lower_bank_final_ft_navd88"])),
        "hydraulic_valid_sections": int(np.isfinite(df["bankfull_conveyance_index"]).sum()),
        "median_local_channel_slope": float(np.nanmedian(df["local_channel_slope"])),
        "method": "v4_multi_cross_section_local_bank_plus_relative_conveyance",
        "note": (
            "Automated LiDAR bank profile for screening only. Review sections near roads, bridges, "
            "detention structures, and unusual terrain before engineering use."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        **summary,
        "paths": {
            "bank_profile_csv": str(csv_path),
            "bank_cross_sections_geojson": str(sections_path),
            "bank_profile_points_geojson": str(bank_points_path),
            "mainstem_geojson": str(mainstem_geojson),
            "mainstem_bank_elevation": str(bank_raster_path),
            "mainstem_station": str(station_raster_path),
            "mainstem_channel_elevation": str(channel_raster_path),
            "bank_profile_summary": str(summary_path),
            "bank_cross_section_samples_csv": str(samples_path),
        },
    }


def write_gauge_cross_section_products(
    hydraulic_dem_path,
    bank_profile_csv,
    processed_dir,
    outputs_dir,
):
    """Regenerate the familiar gauge cross-section CSV/JSON/PNG from section 0."""
    import matplotlib.pyplot as plt

    hydraulic_dem_path = Path(hydraulic_dem_path)
    bank_profile_csv = Path(bank_profile_csv)
    processed_dir = Path(processed_dir)
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(bank_profile_csv).sort_values("station_m").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Bank profile is empty.")
    row = df.iloc[0]
    cross = np.array([float(row["cross_dx"]), float(row["cross_dy"])], dtype=float)

    with rasterio.open(hydraulic_dem_path) as dem_src:
        sec = extract_cross_section(
            dem_src,
            float(row["center_x"]),
            float(row["center_y"]),
            cross,
            float(row["routing_channel_elevation_m_navd88"]),
            float(row["station_m"]),
            half_width_m=150.0,
            sample_spacing_m=1.0,
            seed_offset_m=4.0,
        )
        crs = dem_src.crs

    profile_df = pd.DataFrame(
        {
            "distance_m": sec["distances"],
            "x": sec["sample_x"],
            "y": sec["sample_y"],
            "elevation_m_navd88": sec["elevation_m"],
            "elevation_ft_navd88": sec["elevation_ft"],
            "smoothed_elevation_ft_navd88": sec["smooth_ft"],
        }
    )
    csv_path = processed_dir / "gauge_cross_section.csv"
    profile_df.to_csv(csv_path, index=False)

    line = LineString(
        [
            (float(row["line_x0"]), float(row["line_y0"])),
            (float(row["line_x1"]), float(row["line_y1"])),
        ]
    )
    line_path = processed_dir / "gauge_cross_section.geojson"
    gpd.GeoDataFrame(
        {"name": ["Configured-gauge cross section"]}, geometry=[line], crs=crs
    ).to_crs(4326).to_file(line_path, driver="GeoJSON")

    summary = {
        "channel_elevation_ft_navd88": float(sec["channel_surface_elevation_ft_navd88"]),
        "channel_surface_elevation_ft_navd88": float(sec["channel_surface_elevation_ft_navd88"]),
        "channel_distance_m": float(sec["channel_distance_m"]),
        "left_bank_ft_navd88": float(sec["left_bank_ft_navd88"]),
        "left_bank_distance_m": float(sec["left_bank_distance_m"]),
        "right_bank_ft_navd88": float(sec["right_bank_ft_navd88"]),
        "right_bank_distance_m": float(sec["right_bank_distance_m"]),
        "candidate_overbank_stage_ft": float(row["lower_bank_final_ft_navd88"]),
        "raw_candidate_overbank_stage_ft": float(sec["lower_bank_raw_ft_navd88"]),
        "bank_detection_method": "first_local_crest_from_channel; longitudinal_profile_anchor",
        "qa_flags": sec.get("qa_flags", []),
    }
    summary_path = processed_dir / "gauge_cross_section_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sec["distances"], sec["elevation_ft"], linewidth=1.7, label="LiDAR terrain")
    ax.plot(
        sec["distances"], sec["smooth_ft"], linewidth=1.0, alpha=0.7,
        label="Smoothed profile used for bank detection"
    )
    ax.scatter(
        [sec["channel_distance_m"]], [sec["channel_surface_elevation_ft_navd88"]],
        s=75, label="DEM channel-surface low point"
    )
    ax.scatter(
        [sec["left_bank_distance_m"]], [sec["left_bank_ft_navd88"]],
        s=75, label="Left local-bank crest"
    )
    ax.scatter(
        [sec["right_bank_distance_m"]], [sec["right_bank_ft_navd88"]],
        s=75, label="Right local-bank crest"
    )
    ax.axhline(
        float(row["lower_bank_final_ft_navd88"]), linestyle="--", linewidth=1.4,
        label=f"Gauge bank-profile elevation ({float(row['lower_bank_final_ft_navd88']):.2f} ft)"
    )
    ax.set_xlabel("Cross-section distance (m)")
    ax.set_ylabel("Elevation (ft NAVD88)")
    ax.set_title("LiDAR Cross Section near configured gauge")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    plot_path = outputs_dir / "gauge_cross_section.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "gauge_cross_section_csv": str(csv_path),
        "gauge_cross_section_geojson": str(line_path),
        "gauge_cross_section_summary": str(summary_path),
        "gauge_cross_section_png": str(plot_path),
    }
