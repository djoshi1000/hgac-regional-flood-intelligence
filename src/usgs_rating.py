from __future__ import annotations

from io import StringIO
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import requests

STAC_SEARCH = "https://api.waterdata.usgs.gov/stac/v0/search"
HEADERS = {"User-Agent": "HGAC-Flood-Intelligence-V5/1.0"}


def fetch_usgs_rating(site_no: str, out_csv: str | Path | None = None, file_type="exsa"):
    """Fetch a current USGS stage-discharge rating through the modern STAC API."""
    mlid = f"USGS-{site_no}"
    filt = f"monitoring_location_id='{mlid}' AND file_type='{file_type}'"
    r = requests.get(
        STAC_SEARCH,
        params={"collection": "ratings", "filter": filt, "limit": 10, "f": "json"},
        headers=HEADERS,
        timeout=90,
    )
    r.raise_for_status()
    feats = r.json().get("features", [])
    if not feats and file_type != "base":
        return fetch_usgs_rating(site_no, out_csv=out_csv, file_type="base")
    if not feats:
        raise RuntimeError(f"No USGS rating file is published for site {site_no}.")
    feat = feats[0]
    href = feat.get("assets", {}).get("data", {}).get("href")
    if not href:
        raise RuntimeError(f"USGS rating STAC item for {site_no} has no data asset.")
    rr = requests.get(href, headers=HEADERS, timeout=90)
    rr.raise_for_status()
    text = rr.text

    lines = text.splitlines()
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith("INDEP\t") or line.strip().split("\t")[0] == "INDEP":
            data_start = i
            break
    if data_start is None:
        raise RuntimeError("Could not locate INDEP/DEP columns in USGS rating file.")
    # RDB has a field-width/type row immediately under the header; pandas will
    # read it as data, so numeric coercion naturally removes it.
    tab = "\n".join(lines[data_start:])
    df = pd.read_csv(StringIO(tab), sep="\t", dtype=str)
    if "INDEP" not in df.columns or "DEP" not in df.columns:
        raise RuntimeError("USGS rating file does not contain INDEP and DEP columns.")
    stage = pd.to_numeric(df["INDEP"], errors="coerce")
    flow = pd.to_numeric(df["DEP"], errors="coerce")
    shift = pd.to_numeric(df["SHIFT"], errors="coerce") if "SHIFT" in df.columns else 0.0
    out = pd.DataFrame({
        "gage_height_ft": stage,
        "shift_ft": shift,
        "flow_cfs": flow,
    }).dropna(subset=["gage_height_ft", "flow_cfs"])
    out = out[out["flow_cfs"] >= 0].sort_values("gage_height_ft").drop_duplicates("gage_height_ft")
    out.attrs["asset_url"] = href
    out.attrs["file_type"] = feat.get("properties", {}).get("file_type", file_type)
    if out_csv:
        p = Path(out_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p, index=False)
        (p.with_suffix(p.suffix + ".json")).write_text(json.dumps({
            "site_no": site_no,
            "asset_url": href,
            "file_type": out.attrs["file_type"],
            "note": "INDEP is USGS gage height; V5 converts it to NAVD88 only after a datum offset passes QA.",
        }, indent=2), encoding="utf-8")
    return out


def make_absolute_rating(rating_df: pd.DataFrame, gage_datum_navd88_ft: float, out_csv=None):
    out = rating_df.copy()
    out["stage_ft"] = out["gage_height_ft"].astype(float) + float(gage_datum_navd88_ft)
    out = out[["flow_cfs", "stage_ft", "gage_height_ft"]].sort_values("stage_ft")
    if out_csv:
        p = Path(out_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p, index=False)
    return out


def datum_candidate_from_monitoring_metadata(gauge: dict):
    """Return a provisional gage-datum offset and a QA status.

    USGS gage height is relative to a site-specific gage datum. For automated
    LiDAR inundation, that datum must be tied to NAVD88. V5 only auto-accepts
    monitoring-location altitude when the published vertical datum is NAVD88;
    even then the pipeline performs an independent DEM/current-stage consistency
    check before allowing inundation.
    """
    alt = gauge.get("altitude_ft")
    vd = (gauge.get("vertical_datum") or gauge.get("vertical_datum_name") or "").upper()
    if alt is None or not np.isfinite(float(alt)):
        return None, "missing_gage_datum"
    if "NAVD" in vd and "88" in vd:
        return float(alt), "provisional_navd88_metadata"
    if "NGVD" in vd or "29" in vd:
        return None, "ngvd29_requires_conversion_or_manual_override"
    return None, "vertical_datum_not_navd88"
