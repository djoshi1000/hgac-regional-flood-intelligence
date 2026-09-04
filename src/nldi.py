import time
import geopandas as gpd
import requests

BASE = "https://api.water.usgs.gov/nldi/linked-data"
HEADERS = {"User-Agent": "HGAC-Hunting-Bayou-Flood-Prototype/0.2"}


def _get(url, params=None, retries=4):
    last_error = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=90)

            # Retry temporary server/gateway failures.
            if r.status_code in (500, 502, 503, 504):
                last_error = requests.HTTPError(
                    f"{r.status_code} server error for {r.url}"
                )
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue

            r.raise_for_status()
            return r.json()

        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise

    raise last_error


def site_identifier(site_no):
    return f"USGS-{site_no}"


def get_feature(site_no):
    return _get(
        f"{BASE}/nwissite/{site_identifier(site_no)}",
        {"f": "json"},
    )


def get_basin(site_no, simplified=False, split_catchment=True):
    """
    Try the precise point-specific basin first.

    If USGS NLDI returns a temporary 5xx error, progressively fall back to:
      1) unsplit full-resolution upstream basin
      2) simplified unsplit upstream basin

    Those fallbacks are fully adequate for selecting LiDAR tiles for this prototype.
    """
    url = f"{BASE}/nwissite/{site_identifier(site_no)}/basin"

    attempts = [
        {
            "f": "json",
            "simplified": str(bool(simplified)).lower(),
            "splitCatchment": str(bool(split_catchment)).lower(),
        },
        {
            "f": "json",
            "simplified": "false",
            "splitCatchment": "false",
        },
        {
            "f": "json",
            "simplified": "true",
            "splitCatchment": "false",
        },
    ]

    errors = []
    seen = set()

    for params in attempts:
        key = tuple(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)

        try:
            print(
                "Trying NLDI basin:",
                f"simplified={params['simplified']}, "
                f"splitCatchment={params['splitCatchment']}"
            )
            payload = _get(url, params=params, retries=3)
            features = payload.get("features", [])
            if features:
                if params["splitCatchment"] == "false":
                    print(
                        "[INFO] Using an unsplit NHDPlus upstream basin fallback. "
                        "This is sufficient for DEM tile selection and the first prototype."
                    )
                return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        except Exception as exc:
            errors.append(str(exc))
            print("[WARN] NLDI attempt failed:", exc)

    raise RuntimeError(
        "USGS NLDI basin service failed for all basin variants. "
        "This is most likely a temporary upstream service problem.\n\n"
        + "\n".join(errors)
    )


def get_comid(site_no):
    try:
        payload = get_feature(site_no)
        features = payload.get("features", [])
        if not features:
            return None
        props = features[0].get("properties", {})
        val = props.get("comid") or props.get("nhdplus_comid")
        return str(val) if val not in (None, "") else None
    except Exception as exc:
        print("[WARN] Could not retrieve COMID from NLDI:", exc)
        return None
