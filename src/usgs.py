from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

# V4.3.1: use the modern USGS Water Data API first.  The legacy
# waterservices endpoint is retained only as a temporary fallback because USGS
# is decommissioning WaterServices in early 2027 and may intentionally degrade
# it during the second half of 2026.
MODERN_CONTINUOUS = (
    "https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous/items"
)
LEGACY_IV = "https://waterservices.usgs.gov/nwis/iv/"

_RETRYABLE = {429, 500, 502, 503, 504}


def _session():
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "HGAC-Flood-Prototype/4.3.1",
            "Accept": "application/json",
        }
    )
    return s


def _get_json(url, *, params=None, timeout=45, retries=3):
    last_exc = None
    with _session() as s:
        for attempt in range(max(int(retries), 1)):
            try:
                r = s.get(url, params=params, timeout=timeout)
                if r.status_code in _RETRYABLE and attempt + 1 < retries:
                    time.sleep(min(2 ** attempt, 4))
                    continue
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                if attempt + 1 < retries:
                    time.sleep(min(2 ** attempt, 4))
                    continue
                raise
    raise RuntimeError(str(last_exc) if last_exc else "USGS request failed")


def _modern_page_to_long(payload):
    rows = []
    for feature in payload.get("features", []) or []:
        p = feature.get("properties", {}) or {}
        code = p.get("parameter_code")
        when = p.get("time")
        value = p.get("value")
        if code is None or when is None or value is None:
            continue
        rows.append(
            {
                "time": when,
                "value": value,
                "parameter": str(code),
                "unit": p.get("unit_of_measure"),
                "approval_status": p.get("approval_status"),
            }
        )
    return rows


def _next_link(payload):
    for link in payload.get("links", []) or []:
        if str(link.get("rel", "")).lower() == "next" and link.get("href"):
            return link["href"]
    return None


def _fetch_modern(site_no, period, parameter_codes):
    params = {
        "f": "json",
        "monitoring_location_id": f"USGS-{site_no}",
        "parameter_code": ",".join(parameter_codes),
        "time": period,
        "skipGeometry": "true",
        "limit": 50000,
    }

    rows = []
    url = MODERN_CONTINUOUS
    page_params = params
    page_count = 0
    while url and page_count < 20:
        payload = _get_json(url, params=page_params)
        rows.extend(_modern_page_to_long(payload))
        url = _next_link(payload)
        page_params = None  # next href already contains its query string
        page_count += 1

    if not rows:
        return pd.DataFrame(columns=["time", "00060", "00065"])

    long = pd.DataFrame(rows)
    long["time"] = pd.to_datetime(long["time"], utc=True, errors="coerce")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["time", "value", "parameter"])
    wide = long.pivot_table(
        index="time", columns="parameter", values="value", aggfunc="last"
    )
    wide = wide.sort_index().reset_index()
    wide.columns.name = None
    return wide


def _fetch_legacy(site_no, period, parameter_codes):
    params = {
        "format": "json",
        "sites": site_no,
        "period": period,
        "parameterCd": ",".join(parameter_codes),
        "siteStatus": "all",
    }
    payload = _get_json(LEGACY_IV, params=params)

    frames = []
    for series in payload.get("value", {}).get("timeSeries", []):
        variable = series.get("variable", {})
        code = variable.get("variableCode", [{}])[0].get("value")
        values = []
        for block in series.get("values", []):
            values.extend(block.get("value", []))
        if not values:
            continue
        df = pd.DataFrame(values)
        if "dateTime" not in df or "value" not in df:
            continue
        df["time"] = pd.to_datetime(df["dateTime"], utc=True, errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["time", "value"]].dropna()
        df["parameter"] = code
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["time", "00060", "00065"])
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(
        index="time", columns="parameter", values="value", aggfunc="last"
    )
    wide = wide.sort_index().reset_index()
    wide.columns.name = None
    return wide


def _save_cache(df, cache_path, source):
    if df is None or df.empty or not cache_path:
        return
    cache = Path(cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    meta = {
        "saved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source": source,
    }
    cache.with_suffix(cache.suffix + ".json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def _load_cache(cache_path):
    if not cache_path:
        return None
    cache = Path(cache_path)
    if not cache.exists():
        return None
    try:
        df = pd.read_csv(cache)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        for c in ("00060", "00065"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        meta_path = cache.with_suffix(cache.suffix + ".json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        df.attrs["usgs_source"] = "cache"
        df.attrs["usgs_cache_saved_at_utc"] = meta.get("saved_at_utc")
        df.attrs["usgs_cache_original_source"] = meta.get("source")
        return df
    except Exception:
        return None


def fetch_iv(
    site_no,
    period="P7D",
    parameter_codes=("00060", "00065"),
    cache_path="data/processed/usgs_iv_cache.csv",
    allow_legacy_fallback=True,
):
    """Fetch USGS continuous values with modern-API primary + resilient fallback.

    Returns the same wide DataFrame interface used by earlier prototype versions:
    columns ``time``, ``00060`` (discharge), and/or ``00065`` (gage height).

    DataFrame attrs:
      usgs_source = modern_api | legacy_api | cache
      usgs_primary_error / usgs_legacy_error when fallback was required
    """

    modern_error = None
    legacy_error = None

    try:
        df = _fetch_modern(str(site_no), str(period), tuple(parameter_codes))
        if not df.empty:
            df.attrs["usgs_source"] = "modern_api"
            _save_cache(df, cache_path, "modern_api")
            return df
        modern_error = "Modern USGS API returned no continuous values."
    except Exception as exc:
        modern_error = str(exc)

    if allow_legacy_fallback:
        try:
            df = _fetch_legacy(str(site_no), str(period), tuple(parameter_codes))
            if not df.empty:
                df.attrs["usgs_source"] = "legacy_api"
                df.attrs["usgs_primary_error"] = modern_error
                _save_cache(df, cache_path, "legacy_api")
                return df
            legacy_error = "Legacy USGS WaterServices returned no values."
        except Exception as exc:
            legacy_error = str(exc)

    cached = _load_cache(cache_path)
    if cached is not None and not cached.empty:
        cached.attrs["usgs_primary_error"] = modern_error
        cached.attrs["usgs_legacy_error"] = legacy_error
        return cached

    pieces = ["USGS continuous data unavailable."]
    if modern_error:
        pieces.append(f"Modern API: {modern_error}")
    if allow_legacy_fallback and legacy_error:
        pieces.append(f"Legacy fallback: {legacy_error}")
    raise RuntimeError(" ".join(pieces))


def latest_observation(df):
    if df is None or df.empty:
        return {}
    row = df.sort_values("time").iloc[-1]
    return {
        "time": row.get("time"),
        "discharge_cfs": row.get("00060"),
        "stage_ft": row.get("00065"),
    }
