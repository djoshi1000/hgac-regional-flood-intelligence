from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union
from pyproj import Transformer
import pyflwdir
from pyflwdir import dem as pyflwdir_dem


FLOAT_NODATA = -9999.0
SQMI_TO_KM2 = 2.589988110336


def _write_raster(path, array, profile, dtype, nodata):
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


def _project_gauge_to_raster(src, lon, lat):
    transformer = Transformer.from_crs(
        "EPSG:4326",
        src.crs,
        always_xy=True,
    )
    x, y = transformer.transform(float(lon), float(lat))
    row, col = src.index(x, y)
    return float(x), float(y), int(row), int(col)


def _cell_center(transform, row, col):
    x = transform.c + (col + 0.5) * transform.a + (row + 0.5) * transform.b
    y = transform.f + (col + 0.5) * transform.d + (row + 0.5) * transform.e
    return float(x), float(y)


def _nearest_true_cell(mask, row, col, transform, x, y, max_radius_m=250.0):
    """Spatial fallback: nearest True raster cell to a requested point."""
    nrows, ncols = mask.shape
    cell_size = max(abs(transform.a), abs(transform.e))
    radius_cells = max(1, int(np.ceil(max_radius_m / cell_size)))

    r0 = max(0, row - radius_cells)
    r1 = min(nrows, row + radius_cells + 1)
    c0 = max(0, col - radius_cells)
    c1 = min(ncols, col + radius_cells + 1)

    rr, cc = np.where(mask[r0:r1, c0:c1])
    if rr.size == 0:
        raise RuntimeError(
            f"Could not find a qualifying stream cell within {max_radius_m:.0f} m of the gauge."
        )

    rr = rr + r0
    cc = cc + c0

    xs = transform.c + (cc + 0.5) * transform.a + (rr + 0.5) * transform.b
    ys = transform.f + (cc + 0.5) * transform.d + (rr + 0.5) * transform.e

    dist2 = (xs - x) ** 2 + (ys - y) ** 2
    j = int(np.argmin(dist2))

    return int(rr[j]), int(cc[j]), float(np.sqrt(dist2[j]))


def _trace_mainstem(flw, outlet_idx, uparea, basin_mask, min_uparea_km2):
    """Trace the single dominant upstream path from the snapped gauge outlet."""
    nrows, ncols = uparea.shape
    n = nrows * ncols
    mainstem = np.zeros(n, dtype=bool)

    idxs_us_main = np.asarray(flw.idxs_us_main)
    up_flat = np.asarray(uparea).ravel()
    basin_flat = np.asarray(basin_mask).ravel()

    idx = int(outlet_idx)
    seen = set()

    while 0 <= idx < n and idx not in seen and basin_flat[idx]:
        seen.add(idx)
        mainstem[idx] = True

        next_idx = int(idxs_us_main[idx])
        if next_idx < 0 or next_idx >= n or next_idx == idx:
            break
        if not basin_flat[next_idx]:
            break

        if np.isfinite(up_flat[next_idx]) and up_flat[next_idx] < min_uparea_km2:
            break

        idx = next_idx

    return mainstem.reshape((nrows, ncols))




def _trace_selected_branch_mainstem(
    flw,
    outlet_idx,
    uparea,
    basin_mask,
    stream_mask,
    transform,
    crs,
    selected_lon,
    selected_lat,
    min_uparea_km2,
    selected_snap_max_m=2000.0,
):
    """Trace the branch the user actually selected, then connect it to the gauge.

    The regional app must honor the clicked bayou rather than always choosing the
    basin's largest upstream tributary.  We snap the selected point to the DEM
    stream network, follow that cell downstream until the configured gauge outlet,
    then extend upstream using the locally dominant upstream continuation.

    Returns
    -------
    mainstem_mask, snap_distance_m, selected_stream_idx
    """
    nrows, ncols = uparea.shape
    n = nrows * ncols
    basin_flat = np.asarray(basin_mask).ravel()
    stream_flat = np.asarray(stream_mask).ravel()
    up_flat = np.asarray(uparea).ravel()

    class _Src:
        pass
    src = _Src()
    src.crs = crs
    src.transform = transform
    src.index = lambda x, y: rasterio.transform.rowcol(transform, x, y)

    transformer = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    sx, sy = transformer.transform(float(selected_lon), float(selected_lat))
    srow, scol = rasterio.transform.rowcol(transform, sx, sy)
    rr, cc, snap_dist = _nearest_true_cell(
        stream_mask & basin_mask, int(srow), int(scol), transform, sx, sy,
        max_radius_m=float(selected_snap_max_m),
    )
    selected_idx = int(rr * ncols + cc)

    idxs_ds = np.asarray(getattr(flw, 'idxs_ds', []))
    if idxs_ds.size != n:
        raise RuntimeError('Flow-direction object does not expose downstream indices needed for selected-branch tracing.')

    # Follow the selected branch downstream to the gauge outlet.
    downstream = []
    idx = selected_idx
    seen = set()
    reached = False
    while 0 <= idx < n and idx not in seen and basin_flat[idx]:
        downstream.append(idx)
        if idx == int(outlet_idx):
            reached = True
            break
        seen.add(idx)
        nxt = int(idxs_ds[idx])
        if nxt < 0 or nxt >= n or nxt == idx:
            break
        idx = nxt

    if not reached:
        raise RuntimeError(
            'The selected bayou branch does not drain to the automatically chosen control gauge in the DEM routing network. '
            'The app will not silently model another branch.'
        )

    mainstem = np.zeros(n, dtype=bool)
    for i in downstream:
        mainstem[int(i)] = True

    # Extend upstream from the selected point along the locally dominant
    # continuation. This keeps the model centered on the clicked bayou while
    # still using DEM routing upstream of the click.
    idxs_us_main = np.asarray(flw.idxs_us_main)
    idx = selected_idx
    seen = set(downstream)
    while 0 <= idx < n and basin_flat[idx]:
        nxt = int(idxs_us_main[idx])
        if nxt < 0 or nxt >= n or nxt == idx or nxt in seen or not basin_flat[nxt]:
            break
        if np.isfinite(up_flat[nxt]) and up_flat[nxt] < float(min_uparea_km2):
            break
        mainstem[nxt] = True
        seen.add(nxt)
        idx = nxt

    return mainstem.reshape((nrows, ncols)), float(snap_dist), int(selected_idx)


def _write_basin_geojson(mask, transform, crs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    geoms = []
    raster = mask.astype("uint8")
    for geom, value in shapes(raster, mask=mask, transform=transform, connectivity=8):
        if int(value) == 1:
            geoms.append(shape(geom))

    if not geoms:
        path.write_text(
            json.dumps({"type": "FeatureCollection", "features": []}),
            encoding="utf-8",
        )
        return

    merged = unary_union(geoms)
    gdf = gpd.GeoDataFrame(
        {"name": ["DEM-derived upstream basin to USGS gauge"]},
        geometry=[merged],
        crs=crs,
    )
    gdf.to_crs(4326).to_file(path, driver="GeoJSON")


def build_corrected_terrain(
    dem_path,
    gauge_lon,
    gauge_lat,
    processed_dir,
    stream_threshold_km2=5.0,
    mainstem_threshold_km2=8.0,
    expected_drainage_area_sqmi=None,
    gauge_snap_max_m=250.0,
    selected_lon=None,
    selected_lat=None,
    selected_snap_max_m=2000.0,
):
    """
    Build an upstream-basin, mainstem-referenced HAND workflow.

    Important corrections:
      1. Do NOT force every DEM cell to drain to the gauge. Flow directions are
         first derived using normal valid-grid edge outlets.
      2. Snap the gauge to a real DEM-derived stream cell.
      3. Delineate only the cells upstream of that snapped outlet.
      4. Compare the DEM-derived upstream area with the published gauge drainage
         area as a QA check.
      5. Compute HAND only within that upstream basin and reference it to the
         dominant configured-basin mainstem.
      6. Keep the original LiDAR DEM untouched for final depth calculations.
    """
    dem_path = Path(dem_path)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not dem_path.exists():
        raise FileNotFoundError(f"Analysis DEM not found: {dem_path}")

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float32")
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        src_nodata = src.nodata
        gauge_x, gauge_y, gauge_row_input, gauge_col_input = _project_gauge_to_raster(
            src, gauge_lon, gauge_lat
        )

    valid = np.isfinite(dem)
    if src_nodata is not None:
        valid &= dem != src_nodata

    if not valid.any():
        raise RuntimeError("DEM contains no valid terrain cells.")

    if not (0 <= gauge_row_input < dem.shape[0] and 0 <= gauge_col_input < dem.shape[1]):
        raise RuntimeError("Gauge coordinate falls outside the DEM.")

    input_gauge_dem_m = float(dem[gauge_row_input, gauge_col_input])

    routing_dem = dem.copy()
    routing_dem[~valid] = FLOAT_NODATA

    print("Conditioning DEM for routing using normal edge outlets ...")
    conditioned, d8 = pyflwdir_dem.fill_depressions(
        routing_dem,
        outlets="edge",
        idxs_pit=None,
        nodata=FLOAT_NODATA,
        connectivity=8,
    )

    flw = pyflwdir.from_array(
        d8,
        ftype="d8",
        mask=valid,
        transform=transform,
        latlon=bool(crs.is_geographic),
        cache=True,
    )

    print("Computing preliminary upstream area and stream network ...")
    uparea_all = np.asarray(flw.upstream_area("km2"), dtype="float32")
    stream_all = valid & (uparea_all >= float(stream_threshold_km2))

    # First try a flow-path snap. If the coordinate is on a road/bridge or is
    # slightly offset, the downstream trace generally reaches the configured mainstem.
    outlet_idx = None
    snap_path_m = None
    try:
        idxs, dists = flw.snap(
            xy=(np.array([gauge_x]), np.array([gauge_y])),
            mask=stream_all,
            max_length=float(gauge_snap_max_m),
            unit="m",
            direction="down",
        )
        candidate_idx = int(np.asarray(idxs).ravel()[0])
        candidate_dist = float(np.asarray(dists).ravel()[0])
        if 0 <= candidate_idx < dem.size and stream_all.ravel()[candidate_idx]:
            outlet_idx = candidate_idx
            snap_path_m = candidate_dist
    except Exception:
        outlet_idx = None

    # Robust spatial fallback if flow-path snapping does not find a stream.
    if outlet_idx is None:
        rr, cc, spatial_dist = _nearest_true_cell(
            stream_all,
            gauge_row_input,
            gauge_col_input,
            transform,
            gauge_x,
            gauge_y,
            max_radius_m=float(gauge_snap_max_m),
        )
        outlet_idx = rr * dem.shape[1] + cc
        snap_path_m = spatial_dist

    outlet_row, outlet_col = np.unravel_index(outlet_idx, dem.shape)
    outlet_x, outlet_y = _cell_center(transform, outlet_row, outlet_col)
    outlet_euclidean_snap_m = float(
        np.hypot(outlet_x - gauge_x, outlet_y - gauge_y)
    )

    print("Delineating the DEM-derived upstream basin to the snapped gauge ...")
    basin_ids = flw.basins(
        idxs=np.array([outlet_idx], dtype=np.int64),
        ids=np.array([1], dtype=np.uint32),
    )
    basin = valid & (np.asarray(basin_ids) == 1)

    if not basin.any():
        raise RuntimeError("DEM-derived upstream basin is empty.")

    pixel_area_m2 = abs(
        transform.a * transform.e - transform.b * transform.d
    )
    basin_area_km2 = float(basin.sum() * pixel_area_m2 / 1_000_000.0)
    outlet_uparea_km2 = float(uparea_all[outlet_row, outlet_col])

    expected_km2 = None
    area_ratio_to_usgs = None
    if expected_drainage_area_sqmi is not None:
        expected_km2 = float(expected_drainage_area_sqmi) * SQMI_TO_KM2
        if expected_km2 > 0:
            area_ratio_to_usgs = basin_area_km2 / expected_km2
            if area_ratio_to_usgs < 0.50 or area_ratio_to_usgs > 1.50:
                raise RuntimeError(
                    "DEM-derived upstream basin differs drastically from the published "
                    f"USGS drainage area: {basin_area_km2:.2f} km² vs "
                    f"{expected_km2:.2f} km² ({area_ratio_to_usgs:.1%}). "
                    "Check gauge coordinates, stream snapping, and the DEM clip."
                )

    stream_network = basin & stream_all

    mainstem_threshold_km2 = min(
        float(mainstem_threshold_km2),
        max(0.1, outlet_uparea_km2 * 0.80),
    )

    selected_stream_idx = None
    selected_click_to_mainstem_m = None
    if selected_lon is not None and selected_lat is not None:
        print("Tracing the user-selected bayou branch to the control gauge ...")
        mainstem, selected_click_to_mainstem_m, selected_stream_idx = _trace_selected_branch_mainstem(
            flw=flw,
            outlet_idx=outlet_idx,
            uparea=uparea_all,
            basin_mask=basin,
            stream_mask=stream_network,
            transform=transform,
            crs=crs,
            selected_lon=float(selected_lon),
            selected_lat=float(selected_lat),
            min_uparea_km2=mainstem_threshold_km2,
            selected_snap_max_m=float(selected_snap_max_m),
        )
    else:
        print("Tracing the dominant configured-basin mainstem ...")
        mainstem = _trace_mainstem(
            flw,
            outlet_idx,
            uparea_all,
            basin,
            mainstem_threshold_km2,
        )
    mainstem[outlet_row, outlet_col] = True

    if int(mainstem.sum()) < 10:
        raise RuntimeError(
            "Mainstem trace is unexpectedly short. Check gauge snapping and routing."
        )

    print("Computing mainstem-referenced HAND inside the upstream basin ...")
    hand_all = np.asarray(
        flw.hand(
            drain=mainstem,
            elevtn=conditioned,
        ),
        dtype="float32",
    )

    hand_valid = basin & np.isfinite(hand_all) & (hand_all > FLOAT_NODATA + 1)
    materially_negative = hand_valid & (hand_all < -0.01)
    tiny_negative = hand_valid & (hand_all < 0.0) & ~materially_negative

    if materially_negative.any():
        min_negative = float(hand_all[materially_negative].min())
        raise RuntimeError(
            "Corrected HAND still contains materially negative values "
            f"(minimum {min_negative:.3f} m). Stop here rather than mapping them."
        )

    hand_all[tiny_negative] = 0.0

    hand = np.full(dem.shape, FLOAT_NODATA, dtype="float32")
    hand[hand_valid] = hand_all[hand_valid]

    drain_elevation = np.full(dem.shape, FLOAT_NODATA, dtype="float32")
    drain_elevation[hand_valid] = conditioned[hand_valid] - hand_all[hand_valid]

    fill_depth = np.full(dem.shape, FLOAT_NODATA, dtype="float32")
    fill_depth[basin] = np.maximum(conditioned[basin] - dem[basin], 0.0)

    conditioned_out = np.full(dem.shape, FLOAT_NODATA, dtype="float32")
    conditioned_out[basin] = conditioned[basin]

    uparea = np.full(dem.shape, FLOAT_NODATA, dtype="float32")
    uparea[basin] = uparea_all[basin]

    d8_out = np.zeros(dem.shape, dtype="uint8")
    d8_out[basin] = np.asarray(d8)[basin]

    outlet_raw_elev_m = float(dem[outlet_row, outlet_col])
    outlet_conditioned_elev_m = float(conditioned[outlet_row, outlet_col])

    qa = {
        "dem": str(dem_path),
        "crs": str(crs),
        "working_resolution_m": float(max(abs(transform.a), abs(transform.e))),
        "working_raster_width": int(dem.shape[1]),
        "working_raster_height": int(dem.shape[0]),
        "working_raster_cells": int(dem.size),
        "input_gauge_lon": float(gauge_lon),
        "input_gauge_lat": float(gauge_lat),
        "input_gauge_projected_x": gauge_x,
        "input_gauge_projected_y": gauge_y,
        "input_gauge_row": int(gauge_row_input),
        "input_gauge_col": int(gauge_col_input),
        "input_gauge_dem_elevation_m_navd88": input_gauge_dem_m,
        "input_gauge_dem_elevation_ft_navd88": input_gauge_dem_m * 3.280839895,
        "snapped_outlet_row": int(outlet_row),
        "snapped_outlet_col": int(outlet_col),
        "snapped_outlet_projected_x": outlet_x,
        "snapped_outlet_projected_y": outlet_y,
        "gauge_snap_path_distance_m": float(snap_path_m),
        "gauge_snap_euclidean_distance_m": outlet_euclidean_snap_m,
        "gauge_raw_dem_elevation_m_navd88": outlet_raw_elev_m,
        "gauge_channel_elevation_m_navd88": outlet_conditioned_elev_m,
        "gauge_channel_elevation_ft_navd88": outlet_conditioned_elev_m * 3.280839895,
        "mainstem_mode": "selected_branch" if selected_stream_idx is not None else "dominant_branch",
        "selected_click_to_mainstem_m": selected_click_to_mainstem_m,
        "selected_stream_idx": selected_stream_idx,
        "dem_upstream_basin_area_km2": basin_area_km2,
        "outlet_upstream_area_km2": outlet_uparea_km2,
        "published_drainage_area_sqmi": (
            float(expected_drainage_area_sqmi)
            if expected_drainage_area_sqmi is not None
            else None
        ),
        "published_drainage_area_km2": expected_km2,
        "dem_to_published_area_ratio": area_ratio_to_usgs,
        "stream_threshold_km2": float(stream_threshold_km2),
        "mainstem_threshold_km2": float(mainstem_threshold_km2),
        "stream_network_cells": int(stream_network.sum()),
        "mainstem_cells": int(mainstem.sum()),
        "negative_hand_cells_before_roundoff_fix": int(materially_negative.sum()),
        "hand_min_m": float(np.nanmin(hand_all[hand_valid])),
        "hand_median_m": float(np.nanmedian(hand_all[hand_valid])),
        "hand_p95_m": float(np.nanpercentile(hand_all[hand_valid], 95)),
        "fill_depth_median_m": float(np.nanmedian(fill_depth[basin])),
        "fill_depth_p95_m": float(np.nanpercentile(fill_depth[basin], 95)),
        "fill_depth_p99_m": float(np.nanpercentile(fill_depth[basin], 99)),
        "fill_depth_max_m": float(np.nanmax(fill_depth[basin])),
    }

    outputs = {
        "conditioned_dem": processed_dir / "dem_conditioned_2m.tif",
        "fill_depth": processed_dir / "dem_fill_depth_2m.tif",
        "flowdir_d8": processed_dir / "flowdir_d8.tif",
        "upstream_area": processed_dir / "upstream_area_km2.tif",
        "stream_network": processed_dir / "stream_mask.tif",
        "mainstem": processed_dir / "mainstem_mask.tif",
        "hand": processed_dir / "hand_2m.tif",
        "drain_elevation": processed_dir / "drain_elevation_2m.tif",
        "basin_mask": processed_dir / "dem_upstream_basin_mask.tif",
        "basin_geojson": processed_dir / "dem_upstream_basin.geojson",
        "qa": processed_dir / "terrain_qa.json",
    }

    _write_raster(outputs["conditioned_dem"], conditioned_out, profile, "float32", FLOAT_NODATA)
    _write_raster(outputs["fill_depth"], fill_depth, profile, "float32", FLOAT_NODATA)
    _write_raster(outputs["flowdir_d8"], d8_out, profile, "uint8", 0)
    _write_raster(outputs["upstream_area"], uparea, profile, "float32", FLOAT_NODATA)
    _write_raster(outputs["stream_network"], stream_network.astype("uint8"), profile, "uint8", 0)
    _write_raster(outputs["mainstem"], mainstem.astype("uint8"), profile, "uint8", 0)
    _write_raster(outputs["hand"], hand, profile, "float32", FLOAT_NODATA)
    _write_raster(outputs["drain_elevation"], drain_elevation, profile, "float32", FLOAT_NODATA)
    _write_raster(outputs["basin_mask"], basin.astype("uint8"), profile, "uint8", 0)
    _write_basin_geojson(basin, transform, crs, outputs["basin_geojson"])

    outputs["qa"].write_text(json.dumps(qa, indent=2), encoding="utf-8")

    return {
        **qa,
        "paths": {k: str(v) for k, v in outputs.items()},
    }
