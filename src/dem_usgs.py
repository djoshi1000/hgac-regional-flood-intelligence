from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
import xml.etree.ElementTree as ET

import geopandas as gpd
import requests
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import box

S3_ROOT = "https://prd-tnm.s3.amazonaws.com"


def houston_dem_prefix(project, work_unit):
    return f"StagedProducts/Elevation/OPR/Projects/{project}/{work_unit}/TIFF/"


def list_s3_keys(prefix):
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": 1000}
        if token:
            params["continuation-token"] = token
        r = requests.get(S3_ROOT, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for n in root.findall("s3:Contents/s3:Key", ns):
            if n.text and n.text.lower().endswith((".tif", ".tiff")):
                keys.append(n.text)
        truncated = root.findtext("s3:IsTruncated", "false", ns).lower() == "true"
        if not truncated:
            break
        token = root.findtext("s3:NextContinuationToken", None, ns)
        if not token:
            break
    return keys


def remote_url(key):
    return f"{S3_ROOT}/{quote(key, safe='/')}"


def tile_intersects(url, aoi_wgs84):
    env = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    }
    with rasterio.Env(**env):
        with rasterio.open(url) as src:
            aoi = aoi_wgs84.to_crs(src.crs)
            return box(*src.bounds).intersects(box(*aoi.total_bounds))


def select_tiles(keys, aoi_wgs84):
    selected = []
    for i, key in enumerate(keys, 1):
        try:
            if tile_intersects(remote_url(key), aoi_wgs84):
                selected.append(key)
        except Exception as exc:
            print(f"[WARN] Could not inspect {Path(key).name}: {exc}")
        if i % 50 == 0:
            print(f"Inspected {i}/{len(keys)}; selected {len(selected)}")
    return selected


def _download_one(key, dest):
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / Path(key).name
    if out.exists() and out.stat().st_size > 0:
        return out
    with requests.get(remote_url(key), stream=True, timeout=180) as r:
        r.raise_for_status()
        with out.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    return out


def download_tiles(keys, dest_dir, workers=4):
    dest = Path(dest_dir)
    paths = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(_download_one, k, dest): k for k in keys}
        for fut in as_completed(futures):
            try:
                p = fut.result()
                paths.append(p)
                print("Downloaded", p.name)
            except Exception as exc:
                print("[ERROR]", futures[fut], exc)
    return sorted(paths)


def mosaic_and_clip(tile_paths, aoi_wgs84, out_path, target_resolution_m):
    if not tile_paths:
        raise RuntimeError("No DEM tiles were downloaded.")

    datasets = [rasterio.open(str(p)) for p in tile_paths]
    try:
        crss = {str(ds.crs) for ds in datasets}
        if len(crss) != 1:
            raise RuntimeError(
                "Selected tiles span multiple CRSs. Reproject the small set of edge tiles "
                "to a common CRS before merging. Found: " + ", ".join(sorted(crss))
            )

        crs = datasets[0].crs
        aoi = aoi_wgs84.to_crs(crs)
        mosaic, transform = merge(
            datasets,
            bounds=tuple(aoi.total_bounds),
            res=(float(target_resolution_m), float(target_resolution_m)),
            nodata=datasets[0].nodata,
        )
        meta = datasets[0].meta.copy()
        meta.update(
            height=mosaic.shape[1], width=mosaic.shape[2],
            transform=transform, count=mosaic.shape[0],
            compress="deflate", tiled=True, BIGTIFF="IF_SAFER",
        )
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        temp = out.with_suffix(".bbox.tif")
        with rasterio.open(temp, "w", **meta) as dst:
            dst.write(mosaic)

        with rasterio.open(temp) as src:
            clipped, tr = mask(
                src,
                [g.__geo_interface__ for g in aoi.geometry],
                crop=True,
            )
            meta2 = src.meta.copy()
            meta2.update(
                height=clipped.shape[1], width=clipped.shape[2],
                transform=tr, compress="deflate", tiled=True, BIGTIFF="IF_SAFER"
            )
        with rasterio.open(out, "w", **meta2) as dst:
            dst.write(clipped)
        temp.unlink(missing_ok=True)
        return out
    finally:
        for ds in datasets:
            ds.close()
