from pathlib import Path
import sys
import zipfile
import re
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd

from src.config import load_config, ensure_directories
from src.dem_usgs import (
    houston_dem_prefix,
    list_s3_keys,
    download_tiles,
    mosaic_and_clip,
)

# -------------------------------------------------------
# HGAC LiDAR 2024 Grid Merged
# -------------------------------------------------------

HGAC_DATASET_ID = "5e9698743aa0425bae0fc97653b9bbc1_0"

HGAC_SHAPEFILE_URL = (
    "https://hub.arcgis.com/api/v3/datasets/"
    f"{HGAC_DATASET_ID}/downloads/data"
    "?format=shp&spatialRefId=4326&where=1%3D1"
)


# -------------------------------------------------------
# Download HGAC LiDAR grid shapefile
# -------------------------------------------------------

def download_hgac_grid(raw_dir: Path) -> Path:

    out_dir = raw_dir / "LiDAR_2024_Grid_Merged"
    zip_path = raw_dir / "LiDAR_2024_Grid_Merged.zip"

    # Reuse if already downloaded
    if out_dir.exists():
        shp_files = list(out_dir.rglob("*.shp"))

        if shp_files:
            print("\nUsing existing HGAC LiDAR grid:")
            print(shp_files[0])

            return shp_files[0]

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n========================================")
    print("DOWNLOADING HGAC LIDAR TILE GRID")
    print("========================================")

    print(HGAC_SHAPEFILE_URL)

    with requests.get(
        HGAC_SHAPEFILE_URL,
        stream=True,
        timeout=180
    ) as r:

        r.raise_for_status()

        with open(zip_path, "wb") as f:

            for chunk in r.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:
                    f.write(chunk)

    if not zipfile.is_zipfile(zip_path):

        raise RuntimeError(
            "Downloaded HGAC file is not a ZIP file."
        )

    print("\nExtracting shapefile...")

    with zipfile.ZipFile(zip_path, "r") as zf:

        zf.extractall(out_dir)

    shp_files = list(out_dir.rglob("*.shp"))

    if not shp_files:

        raise RuntimeError(
            "No SHP file found after extraction."
        )

    print("\nHGAC grid shapefile:")
    print(shp_files[0])

    return shp_files[0]


# -------------------------------------------------------
# Intersect Hunting Bayou basin with HGAC tile grid
# -------------------------------------------------------

def find_intersecting_tiles(
    basin,
    tile_index_path
):

    print("\n========================================")
    print("READING HGAC TILE GRID")
    print("========================================")

    tiles = gpd.read_file(tile_index_path)

    print("Grid CRS:", tiles.crs)

    print("Number of grid cells:", len(tiles))

    print("\nFields:")
    print(list(tiles.columns))

    # Project basin into same CRS
    basin_projected = basin.to_crs(
        tiles.crs
    )

    basin_geom = (
        basin_projected
        .geometry
        .union_all()
    )

    selected = tiles[
        tiles.geometry.intersects(
            basin_geom
        )
    ].copy()

    if selected.empty:

        raise RuntimeError(
            "No LiDAR grid polygons intersect Hunting Bayou."
        )

    print("\n========================================")
    print(
        "INTERSECTING HGAC GRID CELLS:",
        len(selected)
    )
    print("========================================")

    print(
        selected
        .drop(columns="geometry", errors="ignore")
        .to_string(index=False)
    )

    return selected


# -------------------------------------------------------
# Normalize names
# -------------------------------------------------------
def normalize_name(value):

    value = str(value).strip()

    value = Path(value).name

    # ---------------------------------------------------
    # HGAC grid names contain "_Preliminary"
    # Example:
    #
    # 15RTP273300_Preliminary
    #
    # USGS raster name uses the actual tile ID:
    #
    # 15RTP273300
    #
    # Remove Preliminary before matching.
    # ---------------------------------------------------

    value = re.sub(
        r"(?i)[_\-\s]*preliminary",
        "",
        value
    )

    # Also remove common status suffixes if present
    value = re.sub(
        r"(?i)[_\-\s]*(final|draft)$",
        "",
        value
    )

    value = value.lower()

    # Remove file extensions
    value = re.sub(
        r"\.(tif|tiff|las|laz|shp)$",
        "",
        value
    )

    # Keep only letters and numbers
    value = re.sub(
        r"[^a-z0-9]+",
        "",
        value
    )

    return value

# -------------------------------------------------------
# Get USGS TIFF filename listings
#
# This DOES NOT open every raster.
# It only gets filename listings from S3.
# -------------------------------------------------------

def build_s3_lookup(cfg):

    basename_lookup = {}

    normalized_lookup = {}

    print("\n========================================")
    print("READING USGS TIFF FILE LISTS")
    print("========================================")

    for work_unit in cfg["dem"]["work_units"]:

        print("\nWork unit:")
        print(work_unit)

        prefix = houston_dem_prefix(
            cfg["dem"]["project"],
            work_unit
        )

        keys = list_s3_keys(
            prefix
        )

        print(
            "Number of TIFF filenames:",
            len(keys)
        )

        for key in keys:

            basename = (
                Path(key)
                .name
                .lower()
            )

            basename_lookup[
                basename
            ] = key

            normalized = normalize_name(
                basename
            )

            normalized_lookup.setdefault(
                normalized,
                []
            ).append(
                key
            )

    return (
        basename_lookup,
        normalized_lookup
    )


# -------------------------------------------------------
# Extract usable attributes from each HGAC grid row
# -------------------------------------------------------

def get_row_values(row):

    values = []

    for col, val in row.items():

        if col == "geometry":

            continue

        if val is None:

            continue

        text = str(val).strip()

        if (
            text == ""
            or text.lower()
            in {
                "nan",
                "none",
                "null"
            }
        ):

            continue

        values.append(
            (
                col,
                text
            )
        )

    return values


# -------------------------------------------------------
# Match HGAC grid cells to USGS TIFFs
# -------------------------------------------------------

def match_tiles_to_s3(
    selected,
    basename_lookup,
    normalized_lookup
):

    matched = []

    unmatched = []

    all_s3 = list(
        basename_lookup.items()
    )

    print("\n========================================")
    print("MATCHING GRID CELLS TO DEM TIFFS")
    print("========================================")

    for idx, row in selected.iterrows():

        values = get_row_values(
            row
        )

        found = None

        found_by = None

        # -------------------------------------------
        # 1. Exact filename
        # -------------------------------------------

        for col, value in values:

            name = (
                Path(value)
                .name
                .lower()
            )

            possible = [

                name,

                name + ".tif",

                name + ".tiff",

            ]

            for candidate in possible:

                if candidate in basename_lookup:

                    found = basename_lookup[
                        candidate
                    ]

                    found_by = (
                        f"{col} = {value}"
                    )

                    break

            if found:

                break

        # -------------------------------------------
        # 2. Exact normalized name
        # -------------------------------------------

        if not found:

            for col, value in values:

                token = normalize_name(
                    value
                )

                if len(token) < 5:

                    continue

                possible = normalized_lookup.get(
                    token,
                    []
                )

                if len(possible) == 1:

                    found = possible[0]

                    found_by = (
                        f"{col} = {value}"
                    )

                    break

        # -------------------------------------------
        # 3. Unique substring match
        # -------------------------------------------

        if not found:

            for col, value in values:

                token = normalize_name(
                    value
                )

                # Ignore meaningless numbers such as OBJECTID
                if len(token) < 6:

                    continue

                candidates = []

                for basename, key in all_s3:

                    s3_token = normalize_name(
                        basename
                    )

                    if (
                        token in s3_token
                        or
                        s3_token in token
                    ):

                        candidates.append(
                            key
                        )

                candidates = sorted(
                    set(candidates)
                )

                if len(candidates) == 1:

                    found = candidates[0]

                    found_by = (
                        f"{col} = {value}"
                    )

                    break

        # -------------------------------------------
        # Save result
        # -------------------------------------------

        if found:

            print(
                "\nMATCH:"
            )

            print(
                "Grid row:",
                idx
            )

            print(
                "DEM:",
                Path(found).name
            )

            print(
                "Matched using:",
                found_by
            )

            matched.append(
                found
            )

        else:

            unmatched.append(
                (
                    idx,
                    values
                )
            )

    matched = sorted(
        set(matched)
    )

    print("\n========================================")
    print(
        "TOTAL MATCHED DEM FILES:",
        len(matched)
    )
    print("========================================")

    for key in matched:

        print(
            Path(key).name
        )

    # -----------------------------------------------
    # Print unmatched records
    # -----------------------------------------------

    if unmatched:

        print("\n========================================")
        print(
            "UNMATCHED GRID ROWS:",
            len(unmatched)
        )
        print("========================================")

        for idx, values in unmatched:

            print(
                "\nGrid row:",
                idx
            )

            for col, value in values:

                print(
                    f"{col}: {value}"
                )

    if not matched:

        raise RuntimeError(
            "\nNo DEM TIFF files could be matched.\n"
            "Copy the printed HGAC intersecting tile attributes "
            "and send them to ChatGPT."
        )

    return matched


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():

    cfg = load_config()

    ensure_directories(
        cfg
    )

    # ---------------------------------------------------
    # Load Hunting Bayou basin
    # ---------------------------------------------------

    basin_path = Path(
        cfg["paths"][
            "basin_geojson"
        ]
    )

    if not basin_path.exists():

        raise FileNotFoundError(
            "\nHunting Bayou basin not found.\n"
            "Run first:\n"
            "python scripts\\00_get_basin.py"
        )

    basin = (
        gpd.read_file(
            basin_path
        )
        .to_crs(
            4326
        )
    )

    print("\n========================================")
    print("HUNTING BAYOU BASIN")
    print("========================================")

    print(
        "CRS:",
        basin.crs
    )

    print(
        "Bounds:",
        basin.total_bounds
    )

    # ---------------------------------------------------
    # Download HGAC LiDAR tile grid
    # ---------------------------------------------------

    raw_dir = Path(
        cfg["paths"]["raw"]
    )

    tile_index_path = (
        download_hgac_grid(
            raw_dir
        )
    )

    # ---------------------------------------------------
    # Select only grid polygons intersecting basin
    # ---------------------------------------------------

    selected_grid = (
        find_intersecting_tiles(
            basin,
            tile_index_path
        )
    )

    # Save selected tile-grid polygons
    selected_grid_output = (
        Path(
            cfg["paths"][
                "processed"
            ]
        )
        /
        "hunting_bayou_lidar_grid_tiles.geojson"
    )

    selected_grid \
        .to_crs(4326) \
        .to_file(
            selected_grid_output,
            driver="GeoJSON"
        )

    print(
        "\nSelected grid polygons saved:"
    )

    print(
        selected_grid_output
    )

    # ---------------------------------------------------
    # Read only USGS filenames
    # ---------------------------------------------------

    (
        basename_lookup,
        normalized_lookup

    ) = build_s3_lookup(
        cfg
    )

    # ---------------------------------------------------
    # Match grid to DEM filenames
    # ---------------------------------------------------

    matched_keys = (
        match_tiles_to_s3(
            selected_grid,
            basename_lookup,
            normalized_lookup
        )
    )

    # ---------------------------------------------------
    # Download only selected DEMs
    # ---------------------------------------------------

    dem_tile_dir = (
        raw_dir
        /
        "usgs_dem_tiles"
    )

    print("\n========================================")
    print("DOWNLOADING SELECTED DEM FILES")
    print("========================================")

    local_tiles = download_tiles(

        matched_keys,

        dem_tile_dir,

        workers=cfg["dem"][
            "download_workers"
        ],

    )

    print(
        "\nDownloaded/reused tiles:",
        len(local_tiles)
    )

    # ---------------------------------------------------
    # Create 1-m DEM
    # ---------------------------------------------------

    print("\n========================================")
    print("CREATING 1-M HYDRAULIC DEM")
    print("========================================")

    mosaic_and_clip(

        local_tiles,

        basin,

        cfg["paths"][
            "dem_hydraulic"
        ],

        cfg["dem"][
            "hydraulic_resolution_m"
        ],

    )

    print(
        "\nCreated:"
    )

    print(
        cfg["paths"][
            "dem_hydraulic"
        ]
    )

    # ---------------------------------------------------
    # Create 2-m analysis DEM
    # ---------------------------------------------------

    print("\n========================================")
    print("CREATING 2-M ANALYSIS DEM")
    print("========================================")

    mosaic_and_clip(

        local_tiles,

        basin,

        cfg["paths"][
            "dem_analysis"
        ],

        cfg["dem"][
            "analysis_resolution_m"
        ],

    )

    print(
        "\nCreated:"
    )

    print(
        cfg["paths"][
            "dem_analysis"
        ]
    )

    # ---------------------------------------------------
    # Finished
    # ---------------------------------------------------

    print("\n========================================")
    print("DEM PROCESSING COMPLETE")
    print("========================================")

    print(
        "\n1-m hydraulic DEM:"
    )

    print(
        cfg["paths"][
            "dem_hydraulic"
        ]
    )

    print(
        "\n2-m analysis DEM:"
    )

    print(
        cfg["paths"][
            "dem_analysis"
        ]
    )

    print(
        "\nSelected HGAC tile polygons:"
    )

    print(
        selected_grid_output
    )

    print(
        "\nNEXT STEP:"
    )

    print(
        "python scripts\\02_build_terrain.py"
    )


if __name__ == "__main__":

    main()