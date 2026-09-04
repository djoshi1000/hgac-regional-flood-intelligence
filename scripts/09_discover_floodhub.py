from pathlib import Path
import sys
import math

sys.path.insert(
    0,
    str(
        Path(
            __file__
        ).resolve().parents[1]
    )
)

from src.config import load_config
from src.floodhub import FloodHubClient


# ============================================================
# DISTANCE
# ============================================================

def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2,
):

    R = 6371.0088

    p1 = math.radians(
        lat1
    )

    p2 = math.radians(
        lat2
    )

    dp = math.radians(
        lat2
        -
        lat1
    )

    dl = math.radians(
        lon2
        -
        lon1
    )


    a = (

        math.sin(
            dp / 2
        ) ** 2

        +

        math.cos(
            p1
        )

        *

        math.cos(
            p2
        )

        *

        math.sin(
            dl / 2
        ) ** 2

    )


    return (

        2
        *
        R
        *
        math.atan2(
            math.sqrt(
                a
            ),
            math.sqrt(
                1 - a
            ),
        )

    )


# ============================================================
# CONFIG
# ============================================================

cfg = load_config()


api_key = cfg[
    "secrets"
][
    "floodhub_api_key"
]


if not api_key:

    raise RuntimeError(

        "Flood Hub API key is missing. "
        "Set FLOODHUB_API_KEY in .env."

    )


client = FloodHubClient(
    api_key
)


# ============================================================
# HUNTING BAYOU GAUGE
# ============================================================

lat = float(
    cfg["pilot"][
        "gauge_lat"
    ]
)

lon = float(
    cfg["pilot"][
        "gauge_lon"
    ]
)


print(
    "\nSearching Google Flood Forecasting gauges around:"
)

print(
    f"USGS 08075770"
)

print(
    f"{lat:.6f}, {lon:.6f}"
)


# ============================================================
# SEARCH BOX
#
# ~16 km × 16 km around Hunting Bayou
# ============================================================

dlat = 0.075
dlon = 0.085


vertices = [

    (
        lat - dlat,
        lon - dlon
    ),

    (
        lat - dlat,
        lon + dlon
    ),

    (
        lat + dlat,
        lon + dlon
    ),

    (
        lat + dlat,
        lon - dlon
    ),

]


result = client.search_gauges_by_loop(

    vertices,

    include_non_quality_verified=True,

)


gauges = result.get(
    "gauges",
    []
)


print(
    f"\nFound {len(gauges)} Google gauges."
)


# ============================================================
# CALCULATE DISTANCE
# ============================================================

candidates = []


for gauge in gauges:

    location = gauge.get(
        "location",
        {}
    )


    glat = location.get(
        "latitude"
    )

    glon = location.get(
        "longitude"
    )


    if (
        glat is None
        or
        glon is None
    ):

        continue


    distance = haversine_km(

        lat,
        lon,

        float(
            glat
        ),

        float(
            glon
        ),

    )


    candidates.append(

        (
            distance,
            gauge,
        )

    )


candidates.sort(
    key=lambda x: x[0]
)


# ============================================================
# PRINT TOP CANDIDATES
# ============================================================

print(
    "\n============================================="
)

print(
    "NEAREST GOOGLE FLOOD FORECASTING GAUGES"
)

print(
    "============================================="
)


for i, (
    distance,
    gauge,
) in enumerate(
    candidates[:12],
    start=1,
):

    gauge_id = gauge.get(
        "gaugeId"
    )


    print(
        f"\n#{i}"
    )

    print(
        "Gauge ID:",
        gauge_id
    )

    print(
        "Distance:",
        f"{distance:.2f} km"
    )

    print(
        "Site:",
        gauge.get(
            "siteName"
        )
    )

    print(
        "River:",
        gauge.get(
            "river"
        )
    )

    print(
        "Source:",
        gauge.get(
            "source"
        )
    )

    print(
        "Quality verified:",
        gauge.get(
            "qualityVerified"
        )
    )

    print(
        "Has model:",
        gauge.get(
            "hasModel"
        )
    )

    print(
        "Location:",
        gauge.get(
            "location"
        )
    )


    # --------------------------------------------------------
    # GET GOOGLE MODEL METADATA
    # --------------------------------------------------------

    if (
        gauge_id
        and
        gauge.get(
            "hasModel"
        )
    ):

        try:

            model = (
                client.get_gauge_model(
                    gauge_id
                )
            )


            print(
                "Google model unit:",
                model.get(
                    "gaugeValueUnit"
                )
            )

            print(
                "Thresholds:",
                model.get(
                    "thresholds"
                )
            )

            print(
                "Model quality verified:",
                model.get(
                    "qualityVerified"
                )
            )


        except Exception as exc:

            print(
                "Model metadata error:",
                exc
            )


print(
    "\n============================================="
)

print(
    "DO NOT select a gauge only because it is nearest."
)

print(
    "We will check river/source/location/model unit "
    "before connecting it to Hunting Bayou."
)