from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import shapes, rasterize
from rasterio.transform import rowcol
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon, shape
from shapely.ops import unary_union

from src.gauge_network import apply_stage_controls
from src.hydraulics import distribute_section_wse
from src.hydrograph import integrate_excess_hydrograph

FLOAT_NODATA = -9999.0
FT_TO_M = 0.3048
M_TO_FT = 3.280839895013123
M2_PER_SQMI = 2_589_988.110336
CFS_TO_CMS = 0.028316846592


def _assert_same_grid(reference, other, name):
    if reference.shape != other.shape:
        raise RuntimeError(f"Raster grid mismatch for {name}: {other.shape} vs {reference.shape}.")
    if reference.crs != other.crs:
        raise RuntimeError(f"Raster CRS mismatch for {name}: {other.crs} vs {reference.crs}.")
    if not reference.transform.almost_equals(other.transform):
        raise RuntimeError(f"Raster transform mismatch for {name}.")


def _write_float_raster(path, array, profile, valid=None):
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.full(array.shape, FLOAT_NODATA, dtype="float32")
    if valid is None:
        valid = np.isfinite(array)
    out[valid] = array[valid].astype("float32")
    out_profile = profile.copy()
    out_profile.update(
        dtype="float32",
        count=1,
        nodata=FLOAT_NODATA,
        compress="deflate",
        predictor=2,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(out, 1)
    return path


def _write_mask_raster(path, mask, profile):
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = profile.copy()
    p.update(dtype="uint8", count=1, nodata=0, compress="deflate", tiled=True, BIGTIFF="IF_SAFER")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(mask.astype("uint8"), 1)
    return path


def _write_extent_geojson(mask, transform, crs, polygon_out):
    polygon_out = Path(polygon_out)
    polygon_out.parent.mkdir(parents=True, exist_ok=True)
    if not mask.any():
        polygon_out.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
        return
    geoms = []
    for geom, value in shapes(mask.astype("uint8"), mask=mask, transform=transform, connectivity=4):
        if int(value) == 1:
            geoms.append(shape(geom))
    if not geoms:
        polygon_out.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
        return
    merged = unary_union(geoms)
    gdf = gpd.GeoDataFrame(
        {"class": ["v4_2_volume_limited_floodplain_screening"]},
        geometry=[merged],
        crs=crs,
    )
    gdf.to_crs(4326).to_file(polygon_out, driver="GeoJSON")


def _empty_geojson(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")


def _filter_small_seeded_components(mask, seed_mask, connectivity, min_pixels):
    if not mask.any():
        return mask
    structure = ndimage.generate_binary_structure(2, 1 if int(connectivity) == 4 else 2)
    labels, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return mask
    counts = np.bincount(labels.ravel())
    seed_labels = set(np.unique(labels[seed_mask]).tolist())
    seed_labels.discard(0)
    keep = np.zeros(n + 1, dtype=bool)
    for lab in seed_labels:
        if lab < len(counts) and counts[lab] >= int(min_pixels):
            keep[lab] = True
    return keep[labels]


def _qa_point(path, crs, x, y, attributes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame([attributes], geometry=[Point(float(x), float(y))], crs=crs)
    gdf.to_crs(4326).to_file(path, driver="GeoJSON")


def _section_wse_raster(bank, shape, transform, max_distance_m):
    """Nearest-cross-section WSE field with a finite influence distance."""
    rows = []
    cols = []
    vals = []
    for _, r in bank.iterrows():
        h = float(r.get("scenario_wse_m_navd88", np.nan))
        if not np.isfinite(h):
            continue
        rr, cc = rowcol(transform, float(r["center_x"]), float(r["center_y"]))
        rr, cc = int(rr), int(cc)
        if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
            rows.append(rr)
            cols.append(cc)
            vals.append(h)
    if not rows:
        return np.full(shape, np.nan, dtype="float64"), np.full(shape, np.inf, dtype="float64")

    point_mask = np.zeros(shape, dtype=bool)
    id_grid = np.full(shape, -1, dtype="int32")
    for i, (r, c) in enumerate(zip(rows, cols)):
        point_mask[r, c] = True
        id_grid[r, c] = i

    # Distance transform returns nearest point indices efficiently for the full grid.
    res_y = abs(float(transform.e))
    res_x = abs(float(transform.a))
    distance_m, inds = ndimage.distance_transform_edt(
        ~point_mask,
        sampling=(res_y, res_x),
        return_distances=True,
        return_indices=True,
    )
    nearest_id = id_grid[inds[0], inds[1]]
    val_arr = np.asarray(vals, dtype=float)
    wse = np.full(shape, np.nan, dtype="float64")
    good = (nearest_id >= 0) & (distance_m <= float(max_distance_m))
    wse[good] = val_arr[nearest_id[good]]
    return wse, distance_m




def _build_bank_to_bank_channel_corridor(
    bank_df,
    raster_shape,
    transform,
    crs,
    exclusion_buffer_m=2.0,
    polygon_out=None,
    mask_out=None,
    profile=None,
):
    """Build a LiDAR-derived bank-to-bank channel corridor from cross sections.

    Left/right bank locations are taken from the bank-profile cross sections.
    Missing bank distances are interpolated longitudinally before adjacent
    section quadrilaterals are unioned. A small raster-scale safety buffer may
    be applied outside the derived bank lines so channel pixels do not leak into
    floodplain statistics.

    This is an automated screening corridor, not surveyed channel geometry.
    """
    df = bank_df.copy().sort_values("station_m").reset_index(drop=True)
    required = [
        "station_m", "center_x", "center_y", "cross_dx", "cross_dy",
        "left_bank_distance_m", "right_bank_distance_m",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Bank profile is missing channel-corridor columns: " + ", ".join(missing)
        )

    station = pd.to_numeric(df["station_m"], errors="coerce").to_numpy(dtype=float, copy=True)
    left_raw = pd.to_numeric(df["left_bank_distance_m"], errors="coerce").to_numpy(dtype=float, copy=True)
    right_raw = pd.to_numeric(df["right_bank_distance_m"], errors="coerce").to_numpy(dtype=float, copy=True)

    valid_pair = np.isfinite(station) & np.isfinite(left_raw) & np.isfinite(right_raw)
    # Require a plausible non-zero cross-section width. Signs are normalized
    # below, so this works even if an individual section's cross-vector flips.
    valid_pair &= np.abs(right_raw - left_raw) >= 2.0
    if int(valid_pair.sum()) < 4:
        raise RuntimeError(
            "Too few valid left/right bank pairs to build the V4.2 channel corridor. "
            "Re-run scripts\\08_extract_gauge_cross_section.py and inspect bank-profile QA."
        )

    left_sorted = np.minimum(left_raw, right_raw)
    right_sorted = np.maximum(left_raw, right_raw)
    valid_left = valid_pair & np.isfinite(left_sorted)
    valid_right = valid_pair & np.isfinite(right_sorted)

    left = np.interp(station, station[valid_left], left_sorted[valid_left])
    right = np.interp(station, station[valid_right], right_sorted[valid_right])

    cx = pd.to_numeric(df["center_x"], errors="coerce").to_numpy(dtype=float, copy=True)
    cy = pd.to_numeric(df["center_y"], errors="coerce").to_numpy(dtype=float, copy=True)
    dx = pd.to_numeric(df["cross_dx"], errors="coerce").to_numpy(dtype=float, copy=True)
    dy = pd.to_numeric(df["cross_dy"], errors="coerce").to_numpy(dtype=float, copy=True)

    norm = np.hypot(dx, dy)
    ok_norm = np.isfinite(norm) & (norm > 0)
    if not ok_norm.all():
        raise RuntimeError("Invalid cross-section direction vector in bank_profile.csv.")
    dx = dx / norm
    dy = dy / norm

    lx = cx + left * dx
    ly = cy + left * dy
    rx = cx + right * dx
    ry = cy + right * dy

    quads = []
    for i in range(len(df) - 1):
        coords = [
            (float(lx[i]), float(ly[i])),
            (float(rx[i]), float(ry[i])),
            (float(rx[i + 1]), float(ry[i + 1])),
            (float(lx[i + 1]), float(ly[i + 1])),
        ]
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty and poly.area > 0:
            quads.append(poly)

    if not quads:
        raise RuntimeError("V4.2 could not construct any bank-to-bank channel polygons.")

    corridor = unary_union(quads)
    if not corridor.is_valid:
        corridor = corridor.buffer(0)
    buffer_m = max(float(exclusion_buffer_m), 0.0)
    exclusion_geom = corridor.buffer(buffer_m) if buffer_m > 0 else corridor
    if exclusion_geom.is_empty:
        raise RuntimeError("V4.2 channel corridor geometry is empty after construction.")

    mask = rasterize(
        [(exclusion_geom, 1)],
        out_shape=raster_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)

    if polygon_out:
        polygon_out = Path(polygon_out)
        polygon_out.parent.mkdir(parents=True, exist_ok=True)
        gdf = gpd.GeoDataFrame(
            {
                "class": ["v4_2_lidar_bank_to_bank_channel_exclusion"],
                "raw_bank_pairs": [int(valid_pair.sum())],
                "interpolated_sections": [int(len(df) - valid_pair.sum())],
                "exclusion_buffer_m": [buffer_m],
            },
            geometry=[exclusion_geom],
            crs=crs,
        )
        gdf.to_crs(4326).to_file(polygon_out, driver="GeoJSON")

    if mask_out and profile is not None:
        _write_mask_raster(mask_out, mask, profile)

    pixel_area = abs(
        float(transform.a) * float(transform.e)
        - float(transform.b) * float(transform.d)
    )
    qa = {
        "raw_valid_bank_pairs": int(valid_pair.sum()),
        "interpolated_bank_sections": int(len(df) - valid_pair.sum()),
        "channel_corridor_pixels": int(mask.sum()),
        "channel_corridor_area_m2": float(mask.sum() * pixel_area),
        "channel_corridor_area_sqmi": float(mask.sum() * pixel_area / M2_PER_SQMI),
        "channel_exclusion_buffer_m": buffer_m,
    }
    return mask, exclusion_geom, qa



def _volume_limited_propagation(
    candidate_mask,
    seed_mask,
    raw_depth,
    dem,
    transform,
    volume_budget_m3,
    connectivity=4,
    uphill_penalty=8.0,
):
    """Fill a connected floodplain progressively until the volume budget is used.

    This is deliberately a *screening* approximation, not a dynamic hydraulic
    solver. Expansion begins at floodplain-entry seeds and proceeds through the
    connected candidate terrain using a least-cost path. Uphill steps are
    penalized so the fill prefers hydraulically accessible low corridors rather
    than instantly occupying every terrain-connected cell.

    The final cell may receive a partial depth so total stored volume does not
    exceed the supplied budget (within floating-point tolerance).
    """
    import heapq

    candidate = np.asarray(candidate_mask, dtype=bool)
    seeds = np.asarray(seed_mask, dtype=bool) & candidate
    depth_req = np.maximum(np.asarray(raw_depth, dtype=float), 0.0)
    ground = np.asarray(dem, dtype=float)

    out_depth = np.zeros(candidate.shape, dtype="float64")
    accepted = np.zeros(candidate.shape, dtype=bool)
    budget = max(float(volume_budget_m3), 0.0)
    if budget <= 0.0 or not seeds.any() or not candidate.any():
        return out_depth, accepted, 0.0, budget

    pixel_area = abs(
        float(transform.a) * float(transform.e)
        - float(transform.b) * float(transform.d)
    )
    step_y = abs(float(transform.e))
    step_x = abs(float(transform.a))

    if int(connectivity) == 8:
        neighbors = [
            (-1, 0, step_y), (1, 0, step_y), (0, -1, step_x), (0, 1, step_x),
            (-1, -1, (step_x**2 + step_y**2) ** 0.5),
            (-1, 1, (step_x**2 + step_y**2) ** 0.5),
            (1, -1, (step_x**2 + step_y**2) ** 0.5),
            (1, 1, (step_x**2 + step_y**2) ** 0.5),
        ]
    else:
        neighbors = [(-1, 0, step_y), (1, 0, step_y), (0, -1, step_x), (0, 1, step_x)]

    nrows, ncols = candidate.shape
    best = np.full(candidate.shape, np.inf, dtype="float64")
    heap = []
    sr, sc = np.where(seeds)
    for r, c in zip(sr.tolist(), sc.tolist()):
        best[r, c] = 0.0
        heapq.heappush(heap, (0.0, int(r), int(c)))

    used = 0.0
    remaining = budget
    penalty = max(float(uphill_penalty), 0.0)

    while heap and remaining > 0.0:
        cost, r, c = heapq.heappop(heap)
        if cost != best[r, c] or accepted[r, c] or not candidate[r, c]:
            continue

        d = float(depth_req[r, c])
        if not np.isfinite(d) or d <= 0.0:
            continue
        full_cell_volume = d * pixel_area

        if full_cell_volume <= remaining + 1e-12:
            out_depth[r, c] = d
            accepted[r, c] = True
            used += full_cell_volume
            remaining -= full_cell_volume
        else:
            partial_depth = remaining / pixel_area
            if partial_depth > 0:
                out_depth[r, c] = min(d, partial_depth)
                accepted[r, c] = True
                used += out_depth[r, c] * pixel_area
            remaining = 0.0
            break

        current_ground = ground[r, c]
        for dr, dc, step in neighbors:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < nrows and 0 <= nc < ncols):
                continue
            if not candidate[nr, nc] or accepted[nr, nc]:
                continue
            rise = max(0.0, float(ground[nr, nc]) - float(current_ground))
            inc = float(step) * (1.0 + penalty * rise)
            new_cost = cost + inc
            if new_cost < best[nr, nc]:
                best[nr, nc] = new_cost
                heapq.heappush(heap, (new_cost, nr, nc))

    return out_depth, accepted, float(used), float(remaining)


def make_v4_discharge_capacity_inundation(
    dem_path,
    hand_path,
    mainstem_path,
    bank_profile_csv,
    scenario_stage_ft,
    scenario_flow_cfs,
    gauge_bankfull_flow_cfs,
    gauge_channel_elevation_m,
    gauge_upstream_area_km2,
    depth_out,
    polygon_out,
    min_depth_m=0.05,
    max_hand_m=12.0,
    fill_depth_path=None,
    wse_out=None,
    overtopped_sections_out=None,
    seed_points_out=None,
    max_depth_point_out=None,
    scenario_profile_out=None,
    seed_mask_out=None,
    potential_depth_out=None,
    potential_polygon_out=None,
    channel_corridor_out=None,
    channel_mask_out=None,
    channel_exclusion_buffer_m=2.0,
    hydrograph=None,
    hydrograph_flow_unit="cfs",
    hydrograph_max_interval_hours=3.0,
    connectivity=4,
    min_component_pixels=20,
    min_overtop_ft=0.02,
    landward_seed_offset_m=4.0,
    landward_seed_search_m=20.0,
    landward_seed_step_m=2.0,
    fill_depth_warning_m=1.0,
    flow_area_exponent=0.85,
    capacity_area_exponent=0.70,
    contained_stage_exponent=0.60,
    overbank_excess_exponent=0.70,
    capacity_outlier_factor=3.0,
    capacity_floor_factor=1.00,
    capacity_ceiling_factor=1.50,
    max_local_overbank_ft=12.0,
    max_local_overbank_multiplier=1.5,
    max_section_influence_m=1200.0,
    screening_duration_hours=6.0,
    floodplain_volume_fraction=1.0,
    volume_uphill_penalty=8.0,
    volume_warning_ratio=1.5,
    deep_depth_threshold_m=1.5,
    additional_stage_controls=None,
):
    """V4.3 channel-excluded, hydrograph-aware riverine terrain screening.

    Compared with V4.1, this version adds a LiDAR bank-to-bank channel exclusion
    corridor so channel water columns are not counted as floodplain depth/storage,
    and can integrate the actual forecast hydrograph above bankfull for the volume
    budget. It retains V4.1 stable local capacity and volume-limited propagation.

    V4.1 foundations retained:
      * stabilizes LiDAR-derived bankfull capacity around a drainage-area prior;
      * removes the unstable ratio-of-excess-ratios WSE amplification;
      * caps local overbank amplification relative to the gauge scenario;
      * writes the unconstrained terrain-connected extent as an *upper bound*;
      * makes the primary depth/extent volume-limited using the gauge excess-flow
        volume proxy and a seeded least-cost floodplain expansion.

    For live NWM scenarios, ``hydrograph`` should contain time/streamflow rows;
    the volume proxy is then the piecewise-trapezoidal integral of Q-Qbankfull.
    Manual scenarios fall back to scenario excess flow times screening duration.

    This remains a screening approximation and is not a substitute for HEC-RAS
    2D, surveyed channel geometry, structures, or a dynamic hydrograph model.
    """
    dem_path = Path(dem_path)
    hand_path = Path(hand_path)
    mainstem_path = Path(mainstem_path)
    bank_profile_csv = Path(bank_profile_csv)
    for p in [dem_path, hand_path, mainstem_path, bank_profile_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Required V4.2 inundation input missing: {p}")

    with rasterio.open(dem_path) as dem_src, rasterio.open(hand_path) as hand_src, rasterio.open(mainstem_path) as main_src:
        _assert_same_grid(dem_src, hand_src, "HAND")
        _assert_same_grid(dem_src, main_src, "mainstem")
        dem = dem_src.read(1).astype("float64")
        hand = hand_src.read(1).astype("float64")
        mainstem = main_src.read(1) > 0
        profile = dem_src.profile.copy()
        transform = dem_src.transform
        crs = dem_src.crs
        valid = np.isfinite(dem)
        if dem_src.nodata is not None:
            valid &= dem != dem_src.nodata
        valid &= np.isfinite(hand)
        if hand_src.nodata is not None:
            valid &= hand != hand_src.nodata

        fill_depth = None
        if fill_depth_path and Path(fill_depth_path).exists():
            with rasterio.open(fill_depth_path) as fill_src:
                _assert_same_grid(dem_src, fill_src, "DEM fill depth")
                fill_depth = fill_src.read(1).astype("float64")
                if fill_src.nodata is not None:
                    fill_depth[fill_depth == fill_src.nodata] = np.nan

    bank = pd.read_csv(bank_profile_csv).sort_values("station_m").reset_index(drop=True)
    required_cols = [
        "station_m", "center_x", "center_y", "cross_dx", "cross_dy",
        "upstream_area_km2", "routing_channel_elevation_ft_navd88",
        "lower_bank_final_ft_navd88", "lower_bank_distance_m",
        "lower_bank_side_sign", "bankfull_conveyance_index",
        "left_bank_distance_m", "right_bank_distance_m",
    ]
    missing = [c for c in required_cols if c not in bank.columns]
    if missing:
        raise RuntimeError(
            "bank_profile.csv is missing V4.2-required hydraulic/channel columns: " + ", ".join(missing) +
            ". Re-run scripts\\08_extract_gauge_cross_section.py with V4 code."
        )

    channel_mask, channel_geom, channel_qa = _build_bank_to_bank_channel_corridor(
        bank,
        raster_shape=dem.shape,
        transform=transform,
        crs=crs,
        exclusion_buffer_m=float(channel_exclusion_buffer_m),
        polygon_out=channel_corridor_out,
        mask_out=channel_mask_out,
        profile=profile,
    )

    gauge_bank_ft = float(bank.iloc[0]["lower_bank_final_ft_navd88"])
    section_profile, hyd_qa = distribute_section_wse(
        bank,
        gauge_flow_cfs=float(scenario_flow_cfs),
        gauge_stage_ft=float(scenario_stage_ft),
        gauge_bankfull_flow_cfs=float(gauge_bankfull_flow_cfs),
        gauge_bank_ft=gauge_bank_ft,
        gauge_channel_proxy_ft=float(gauge_channel_elevation_m) * M_TO_FT,
        gauge_area_km2=float(gauge_upstream_area_km2),
        flow_area_exponent=float(flow_area_exponent),
        capacity_area_exponent=float(capacity_area_exponent),
        contained_stage_exponent=float(contained_stage_exponent),
        overbank_excess_exponent=float(overbank_excess_exponent),
        capacity_outlier_factor=float(capacity_outlier_factor),
        capacity_floor_factor=float(capacity_floor_factor),
        capacity_ceiling_factor=float(capacity_ceiling_factor),
        max_local_overbank_ft=float(max_local_overbank_ft),
        max_local_overbank_multiplier=float(max_local_overbank_multiplier),
    )

    section_profile, control_qa = apply_stage_controls(section_profile, additional_stage_controls)
    section_profile["overtopped"] = (
        section_profile["local_overtop_ft"].to_numpy(dtype=float, copy=True)
        >= float(min_overtop_ft)
    )

    if scenario_profile_out:
        Path(scenario_profile_out).parent.mkdir(parents=True, exist_ok=True)
        section_profile.to_csv(scenario_profile_out, index=False)

    wse, distance_to_section = _section_wse_raster(
        section_profile,
        dem.shape,
        transform,
        max_distance_m=float(max_section_influence_m),
    )
    raw_depth = wse - dem
    base_mask = (
        valid
        & (hand >= -1e-6)
        & (hand <= float(max_hand_m))
        & np.isfinite(wse)
        & (raw_depth >= float(min_depth_m))
        & (~channel_mask)
        & (~mainstem)
        & (distance_to_section <= float(max_section_influence_m))
    )

    seed_mask = np.zeros(dem.shape, dtype=bool)
    seed_rows = []
    overtopped_rows = []
    nrows, ncols = dem.shape

    for _, row in section_profile[section_profile["overtopped"]].iterrows():
        overtopped_rows.append(row)
        cx = float(row["center_x"])
        cy = float(row["center_y"])
        dx = float(row["cross_dx"])
        dy = float(row["cross_dy"])
        sign = float(row["lower_bank_side_sign"])
        bank_dist = float(row["lower_bank_distance_m"])
        local_wse_m = float(row["scenario_wse_m_navd88"])

        found = None
        for offset in np.arange(
            float(landward_seed_offset_m),
            float(landward_seed_search_m) + float(landward_seed_step_m),
            float(landward_seed_step_m),
        ):
            d = bank_dist + sign * float(offset)
            x = cx + d * dx
            y = cy + d * dy
            rr, cc = rowcol(transform, x, y)
            rr, cc = int(rr), int(cc)
            if not (0 <= rr < nrows and 0 <= cc < ncols):
                continue
            if not valid[rr, cc] or mainstem[rr, cc] or channel_mask[rr, cc]:
                continue
            if hand[rr, cc] < -1e-6 or hand[rr, cc] > float(max_hand_m):
                continue
            local_depth = local_wse_m - dem[rr, cc]
            if local_depth >= float(min_depth_m):
                found = (rr, cc, x, y, offset, local_depth)
                break

        if found is None:
            continue

        rr, cc, x, y, offset, local_depth = found
        seed_mask[rr, cc] = True
        # Keep the seed footprint very small; V4's 3x3 dilation could create
        # multiple artificial entry paths through narrow barriers.
        seed_rows.append(
            {
                "section_id": int(row.get("section_id", len(seed_rows))),
                "station_m": float(row["station_m"]),
                "local_flow_cfs": float(row["local_flow_cfs"]),
                "bankfull_capacity_cfs": float(row["bankfull_capacity_cfs"]),
                "expected_bankfull_capacity_cfs": float(row.get("expected_bankfull_capacity_cfs", np.nan)),
                "flow_capacity_ratio": float(row["flow_to_capacity_ratio"]),
                "local_bank_ft_navd88": float(row["lower_bank_final_ft_navd88"]),
                "local_wse_ft_navd88": float(row["scenario_wse_ft_navd88"]),
                "local_overtop_ft": float(row["local_overtop_ft"]),
                "seed_search_offset_m": float(offset),
                "x": float(x),
                "y": float(y),
            }
        )

    structure = ndimage.generate_binary_structure(2, 1 if int(connectivity) == 4 else 2)
    if seed_mask.any():
        potential_connected = ndimage.binary_propagation(seed_mask, structure=structure, mask=base_mask)
        potential_connected = _filter_small_seeded_components(
            potential_connected, seed_mask, int(connectivity), int(min_component_pixels)
        )
    else:
        potential_connected = np.zeros(dem.shape, dtype=bool)

    potential_depth = np.zeros(dem.shape, dtype="float64")
    potential_depth[potential_connected] = np.maximum(raw_depth[potential_connected], 0.0)
    potential_depth[potential_depth < float(min_depth_m)] = 0.0
    potential_inundated = valid & (potential_depth > 0)

    pixel_area_m2 = abs(transform.a * transform.e - transform.b * transform.d)
    potential_area_m2 = float(potential_inundated.sum() * pixel_area_m2)
    potential_volume_m3 = (
        float(np.sum(potential_depth[potential_inundated]) * pixel_area_m2)
        if potential_inundated.any() else 0.0
    )

    gauge_excess_q_cfs = max(float(scenario_flow_cfs) - float(gauge_bankfull_flow_cfs), 0.0)

    hydrograph_qa = None
    if hydrograph is not None:
        hydrograph_qa = integrate_excess_hydrograph(
            hydrograph,
            bankfull_flow_cfs=float(gauge_bankfull_flow_cfs),
            flow_unit=hydrograph_flow_unit,
            max_interval_hours=float(hydrograph_max_interval_hours),
        )
        excess_volume_proxy_m3 = float(hydrograph_qa.get("excess_volume_m3", 0.0))
        volume_source = "integrated_hydrograph"
    else:
        excess_volume_proxy_m3 = (
            gauge_excess_q_cfs * CFS_TO_CMS * 3600.0 * max(float(screening_duration_hours), 0.0)
        )
        volume_source = "constant_excess_flow_x_duration"
        hydrograph_qa = {
            "source": volume_source,
            "point_count": 0,
            "horizon_hours": float(screening_duration_hours),
            "excess_volume_m3": float(excess_volume_proxy_m3),
            "peak_flow_cfs": float(scenario_flow_cfs),
            "bankfull_flow_cfs": float(gauge_bankfull_flow_cfs),
            "intervals_used": 0,
            "intervals_skipped_large_gap": 0,
            "exceedance_interval_hours": float(screening_duration_hours) if gauge_excess_q_cfs > 0 else 0.0,
            "max_interval_hours": float(hydrograph_max_interval_hours),
            "start_time_utc": None,
            "end_time_utc": None,
        }

    volume_fraction = float(np.clip(float(floodplain_volume_fraction), 0.0, 1.0))
    volume_budget_m3 = excess_volume_proxy_m3 * volume_fraction

    depth, inundated, volume_used_m3, volume_remaining_m3 = _volume_limited_propagation(
        candidate_mask=potential_connected,
        seed_mask=seed_mask,
        raw_depth=raw_depth,
        dem=dem,
        transform=transform,
        volume_budget_m3=volume_budget_m3,
        connectivity=int(connectivity),
        uphill_penalty=float(volume_uphill_penalty),
    )
    depth[depth < float(min_depth_m)] = 0.0
    inundated = valid & (depth > 0)

    area_m2 = float(inundated.sum() * pixel_area_m2)
    flood_volume_m3 = float(np.sum(depth[inundated]) * pixel_area_m2) if inundated.any() else 0.0
    positive = depth[inundated]
    potential_positive = potential_depth[potential_inundated]

    _write_float_raster(depth_out, depth, profile, valid=valid)
    _write_extent_geojson(inundated, transform, crs, polygon_out)
    if potential_depth_out:
        _write_float_raster(potential_depth_out, potential_depth, profile, valid=valid)
    if potential_polygon_out:
        _write_extent_geojson(potential_inundated, transform, crs, potential_polygon_out)
    if wse_out:
        _write_float_raster(wse_out, wse, profile, valid=np.isfinite(wse) & valid)
    if seed_mask_out:
        _write_mask_raster(seed_mask_out, seed_mask, profile)

    if overtopped_sections_out:
        if overtopped_rows:
            odf = pd.DataFrame(overtopped_rows)
            geoms = [Point(float(x), float(y)) for x, y in zip(odf["center_x"], odf["center_y"])]
            cols = [
                "section_id", "station_m", "upstream_area_km2", "local_flow_cfs",
                "expected_bankfull_capacity_cfs", "bankfull_capacity_cfs",
                "flow_to_capacity_ratio", "lower_bank_final_ft_navd88",
                "scenario_wse_ft_navd88", "local_overtop_ft",
            ]
            cols = [c for c in cols if c in odf.columns]
            gpd.GeoDataFrame(odf[cols].copy(), geometry=geoms, crs=crs).to_crs(4326).to_file(
                overtopped_sections_out, driver="GeoJSON"
            )
        else:
            _empty_geojson(overtopped_sections_out)

    if seed_points_out:
        if seed_rows:
            sdf = pd.DataFrame(seed_rows)
            geoms = [Point(float(x), float(y)) for x, y in zip(sdf["x"], sdf["y"])]
            gpd.GeoDataFrame(sdf.drop(columns=["x", "y"]), geometry=geoms, crs=crs).to_crs(4326).to_file(
                seed_points_out, driver="GeoJSON"
            )
        else:
            _empty_geojson(seed_points_out)

    # QA flags distinguish the unconstrained upper bound from the preferred
    # volume-limited result.
    potential_storage_ratio = (
        potential_volume_m3 / excess_volume_proxy_m3 if excess_volume_proxy_m3 > 0 else None
    )
    storage_to_budget_ratio = (
        flood_volume_m3 / volume_budget_m3 if volume_budget_m3 > 0 else None
    )
    qa_flags = []
    if potential_storage_ratio is not None and potential_storage_ratio > float(volume_warning_ratio):
        qa_flags.append("potential_connected_storage_exceeds_excess_flow_volume_proxy")
    if volume_budget_m3 > 0 and flood_volume_m3 > volume_budget_m3 * 1.001:
        qa_flags.append("volume_limited_storage_exceeds_budget_unexpected")

    channel_distance_m = ndimage.distance_transform_edt(
        ~channel_mask,
        sampling=(abs(float(transform.e)), abs(float(transform.a))),
    )

    max_info = None
    if positive.size:
        flat_idx = int(np.nanargmax(depth))
        rmax, cmax = np.unravel_index(flat_idx, depth.shape)
        xmax = transform.c + (cmax + 0.5) * transform.a + (rmax + 0.5) * transform.b
        ymax = transform.f + (cmax + 0.5) * transform.d + (rmax + 0.5) * transform.e
        mr, mc = np.where(mainstem)
        mx = transform.c + (mc + 0.5) * transform.a + (mr + 0.5) * transform.b
        my = transform.f + (mc + 0.5) * transform.d + (mr + 0.5) * transform.e
        tree = cKDTree(np.column_stack([mx, my])) if len(mx) else None
        dist_to_mainstem = float(tree.query([[xmax, ymax]], k=1)[0][0]) if tree is not None else np.nan
        fill_at_max = (
            float(fill_depth[rmax, cmax])
            if fill_depth is not None and np.isfinite(fill_depth[rmax, cmax]) else None
        )
        to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = to_wgs.transform(float(xmax), float(ymax))
        max_info = {
            "depth_m": float(depth[rmax, cmax]),
            "ground_elevation_ft_navd88": float(dem[rmax, cmax] * M_TO_FT),
            "wse_ft_navd88": float(wse[rmax, cmax] * M_TO_FT),
            "hand_m": float(hand[rmax, cmax]),
            "fill_depth_m": fill_at_max,
            "distance_to_mainstem_m": dist_to_mainstem,
            "distance_to_channel_corridor_m": float(channel_distance_m[rmax, cmax]),
            "longitude": float(lon),
            "latitude": float(lat),
            "row": int(rmax),
            "col": int(cmax),
        }

        # V4.3: make deep-pocket QA interpretable by tying the maximum-depth
        # pixel back to the nearest hydraulic cross section.
        if len(section_profile):
            sec_xy = section_profile[["center_x", "center_y"]].to_numpy(dtype=float, copy=True)
            sec_tree = cKDTree(sec_xy)
            sec_dist, sec_idx = sec_tree.query([[xmax, ymax]], k=1)
            sec = section_profile.iloc[int(sec_idx[0])]
            max_info.update({
                "nearest_section_id": int(sec.get("section_id", int(sec_idx[0]))),
                "nearest_section_distance_m": float(sec_dist[0]),
                "nearest_section_station_m": float(sec.get("station_m", np.nan)),
                "nearest_section_bank_ft_navd88": float(sec.get("lower_bank_final_ft_navd88", np.nan)),
                "nearest_section_wse_ft_navd88": float(sec.get("scenario_wse_ft_navd88", np.nan)),
                "nearest_section_overtop_ft": float(sec.get("local_overtop_ft", np.nan)),
                "bank_to_ground_drop_ft": float(sec.get("lower_bank_final_ft_navd88", np.nan)) - float(dem[rmax, cmax] * M_TO_FT),
            })
        if fill_at_max is not None and fill_at_max >= float(fill_depth_warning_m):
            qa_flags.append("maximum_depth_location_has_large_dem_fill_depth")
        if float(depth[rmax, cmax]) >= float(deep_depth_threshold_m):
            qa_flags.append("maximum_depth_exceeds_deep_depth_qa_threshold")
        if float(channel_distance_m[rmax, cmax]) <= max(abs(float(transform.a)), abs(float(transform.e))) * 1.5:
            qa_flags.append("maximum_depth_immediately_outside_channel_corridor")
        if max_depth_point_out:
            _qa_point(
                max_depth_point_out, crs, xmax, ymax,
                {
                    "depth_m": max_info["depth_m"],
                    "ground_ft": max_info["ground_elevation_ft_navd88"],
                    "wse_ft": max_info["wse_ft_navd88"],
                    "hand_m": max_info["hand_m"],
                    "fill_depth_m": fill_at_max,
                    "distance_to_mainstem_m": dist_to_mainstem,
                    "distance_to_channel_corridor_m": float(channel_distance_m[rmax, cmax]),
                },
            )
    elif max_depth_point_out:
        _empty_geojson(max_depth_point_out)

    deep_mask = inundated & (depth >= float(deep_depth_threshold_m))
    deep_area_m2 = float(deep_mask.sum() * pixel_area_m2)

    result = {
        "method": "v4_3_validation_aoi_channel_excluded_volume_limited_screening",
        "scenario_stage_ft_navd88": float(scenario_stage_ft),
        "scenario_flow_cfs": float(scenario_flow_cfs),
        "gauge_bank_stage_ft_navd88": gauge_bank_ft,
        "gauge_bankfull_flow_cfs": float(gauge_bankfull_flow_cfs),
        "gauge_overbank_excess_ft": float(float(scenario_stage_ft) - gauge_bank_ft),
        "gauge_channel_elevation_m_navd88": float(gauge_channel_elevation_m),
        "bank_sections": int(len(section_profile)),
        "hydraulic_valid_sections": int(hyd_qa.get("valid_conveyance_sections", 0)),
        "capacity_profile_outliers": int(hyd_qa.get("capacity_outlier_sections", 0)),
        "capacity_constrained_low_sections": int(hyd_qa.get("capacity_constrained_low_sections", 0)),
        "capacity_constrained_high_sections": int(hyd_qa.get("capacity_constrained_high_sections", 0)),
        "overtopped_sections": int(section_profile["overtopped"].sum()),
        "seeded_sections": int(len(seed_rows)),
        "channel_corridor_raw_bank_pairs": int(channel_qa["raw_valid_bank_pairs"]),
        "channel_corridor_interpolated_sections": int(channel_qa["interpolated_bank_sections"]),
        "channel_corridor_pixels": int(channel_qa["channel_corridor_pixels"]),
        "channel_corridor_area_m2": float(channel_qa["channel_corridor_area_m2"]),
        "channel_corridor_area_sqmi": float(channel_qa["channel_corridor_area_sqmi"]),
        "channel_exclusion_buffer_m": float(channel_qa["channel_exclusion_buffer_m"]),
        "candidate_pixels_before_connectivity": int(base_mask.sum()),
        "seed_pixels": int(seed_mask.sum()),
        "potential_connected_pixels": int(potential_inundated.sum()),
        "potential_inundated_area_m2": potential_area_m2,
        "potential_inundated_area_sqmi": potential_area_m2 / M2_PER_SQMI,
        "potential_storage_volume_m3": potential_volume_m3,
        "potential_max_depth_m": float(potential_positive.max()) if potential_positive.size else 0.0,
        "volume_limited_pixels": int(inundated.sum()),
        "connected_pixels": int(inundated.sum()),
        "inundated_area_m2": area_m2,
        "inundated_area_sqmi": area_m2 / M2_PER_SQMI,
        "flood_storage_volume_m3": flood_volume_m3,
        "screening_duration_hours": float(screening_duration_hours),
        "volume_budget_source": volume_source,
        "hydrograph_volume_qa": hydrograph_qa,
        "gauge_excess_flow_cfs": gauge_excess_q_cfs,
        "excess_flow_volume_proxy_m3": excess_volume_proxy_m3,
        "floodplain_volume_fraction": volume_fraction,
        "volume_budget_m3": volume_budget_m3,
        "volume_used_m3": float(volume_used_m3),
        "volume_remaining_m3": float(volume_remaining_m3),
        "storage_to_volume_budget_ratio": storage_to_budget_ratio,
        "potential_storage_to_excess_volume_ratio": potential_storage_ratio,
        # Backward-compatible key now refers to the preferred volume-limited map.
        "storage_to_excess_volume_ratio": (
            flood_volume_m3 / excess_volume_proxy_m3 if excess_volume_proxy_m3 > 0 else None
        ),
        "max_depth_m": float(positive.max()) if positive.size else 0.0,
        "mean_depth_m": float(positive.mean()) if positive.size else 0.0,
        "deep_area_sqmi": deep_area_m2 / M2_PER_SQMI,
        "deep_depth_threshold_m": float(deep_depth_threshold_m),
        "max_depth_location": max_info,
        "hydraulic_profile_qa": hyd_qa,
        "gauge_control_qa": control_qa,
        "qa_flags": sorted(set(qa_flags)),
        "depth_raster": str(depth_out),
        "wse_raster": str(wse_out) if wse_out else None,
        "extent_geojson": str(polygon_out),
        "potential_depth_raster": str(potential_depth_out) if potential_depth_out else None,
        "potential_extent_geojson": str(potential_polygon_out) if potential_polygon_out else None,
        "channel_corridor_geojson": str(channel_corridor_out) if channel_corridor_out else None,
        "channel_corridor_mask_raster": str(channel_mask_out) if channel_mask_out else None,
        "overtopped_sections_geojson": str(overtopped_sections_out) if overtopped_sections_out else None,
        "seed_points_geojson": str(seed_points_out) if seed_points_out else None,
        "max_depth_point_geojson": str(max_depth_point_out) if max_depth_point_out else None,
        "scenario_profile_csv": str(scenario_profile_out) if scenario_profile_out else None,
        "seed_mask_raster": str(seed_mask_out) if seed_mask_out else None,
    }
    return result


# Keep the V4 function name for script/app compatibility; behavior is V4.2.


def make_local_bank_connected_inundation(*args, **kwargs):
    raise RuntimeError(
        "The V3 inundation engine is disabled. Use "
        "make_v4_discharge_capacity_inundation (V4.2 behavior) instead."
    )
