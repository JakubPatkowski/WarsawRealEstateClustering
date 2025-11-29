"""
Warsaw District Boundaries Fetcher.

Fetches real administrative boundaries of Warsaw's 18 districts from
OpenStreetMap via Overpass API. Uses the Manual Stitching algorithm (alg5)
which produces the highest quality polygons.

Based on P3 project - warsaw_districts_production.py

Example:
    from src.boundaries import DistrictFetcher

    fetcher = DistrictFetcher()
    gdf = fetcher.fetch_and_process()
    fetcher.save_cache()
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


def stitch_line_segments(
    lines: List[LineString], max_iterations: int = 1000, verbose: bool = False
) -> Optional[Polygon]:
    """
    Manual stitching algorithm for connecting line segments into a polygon.

    This algorithm is the BEST for Warsaw district boundaries because:
    1. Preserves ORIGINAL points from OpenStreetMap (precision!)
    2. Does not introduce artifacts like buffer/simplify
    3. Does not change shape like concave/convex hull
    4. Creates a SINGLE Polygon instead of MultiPolygon

    Algorithm:
    - Iteratively finds matching segment endpoints
    - Joins them into longer paths
    - Continues until polygon is closed

    WARNING: DO NOT MODIFY THIS FUNCTION - it works perfectly!

    Args:
        lines: List of LineString segments to connect
        max_iterations: Maximum iterations (safety limit)
        verbose: Whether to show detailed logs

    Returns:
        Polygon if successful, None otherwise

    Example:
        >>> lines = [LineString([(0,0), (1,0)]), LineString([(1,0), (1,1)])]
        >>> polygon = stitch_line_segments(lines)
        >>> print(polygon.geom_type)
        'Polygon'
    """
    if not lines or len(lines) < 2:
        return None

    try:
        # Convert to coordinate lists for faster operations
        segments: List[List[Tuple[float, float]]] = [
            list(line.coords) for line in lines
        ]

        iteration = 0
        initial_count = len(segments)

        while len(segments) > 1 and iteration < max_iterations:
            iteration += 1
            found_connection = False

            # Check all segment pairs
            for i in range(len(segments)):
                for j in range(i + 1, len(segments)):
                    seg1, seg2 = segments[i], segments[j]

                    # CASE 1: end of seg1 == start of seg2
                    # [A→B] + [B→C] = [A→B→C]
                    if seg1[-1] == seg2[0]:
                        segments[i] = seg1 + seg2[1:]  # Remove duplicate point B
                        segments.pop(j)
                        found_connection = True
                        break

                    # CASE 2: end of seg1 == end of seg2
                    # [A→B] + [C→B] = [A→B→C] (seg2 reversed)
                    elif seg1[-1] == seg2[-1]:
                        segments[i] = seg1 + seg2[-2::-1]
                        segments.pop(j)
                        found_connection = True
                        break

                    # CASE 3: start of seg1 == start of seg2
                    # [B→A] + [B→C] = [A→B→C] (seg1 reversed)
                    elif seg1[0] == seg2[0]:
                        segments[i] = seg1[::-1] + seg2[1:]
                        segments.pop(j)
                        found_connection = True
                        break

                    # CASE 4: start of seg1 == end of seg2
                    # [B→A] + [C→B] = [C→B→A]
                    elif seg1[0] == seg2[-1]:
                        segments[i] = seg2 + seg1[1:]
                        segments.pop(j)
                        found_connection = True
                        break

                if found_connection:
                    break

            # If nothing can be connected, break
            if not found_connection:
                if verbose:
                    logger.warning(
                        f"Cannot connect more segments. "
                        f"Remaining: {len(segments)} of {initial_count}"
                    )
                break

        # Select longest segment as result
        if segments:
            longest = max(segments, key=len)

            # Validation: enough points for polygon?
            if len(longest) >= 4:
                # Close polygon if not closed
                if longest[0] != longest[-1]:
                    longest.append(longest[0])

                polygon = Polygon(longest)

                # Additional geometry validation
                if polygon.is_valid and not polygon.is_empty:
                    return polygon
                else:
                    # Attempt repair
                    polygon = polygon.buffer(0)
                    if polygon.is_valid:
                        if isinstance(polygon, Polygon):
                            return polygon
                        elif isinstance(polygon, MultiPolygon):
                            # Return largest part
                            return max(polygon.geoms, key=lambda p: p.area)

    except Exception as e:
        if verbose:
            logger.error(f"Error in stitch_line_segments: {e}")

    return None


def clean_multipolygon(
    geom: MultiPolygon, area_threshold: float = 0.05, min_ratio: float = 10.0
) -> Polygon:
    """
    Clean MultiPolygon by selecting the largest part.

    The stitch algorithm usually produces Polygon, but this function
    serves as a fallback for edge cases.

    Args:
        geom: MultiPolygon to clean
        area_threshold: Area threshold (% of largest part)
        min_ratio: Minimum area ratio of main to other parts

    Returns:
        Single Polygon (largest part)
    """
    if isinstance(geom, Polygon):
        return geom

    if not isinstance(geom, MultiPolygon):
        raise TypeError(f"Expected Polygon or MultiPolygon, got {type(geom)}")

    parts = list(geom.geoms)
    areas = [part.area for part in parts]
    max_area = max(areas)

    # Check if one part is dominant
    other_areas = [a for a in areas if a != max_area]
    if other_areas:
        max_other = max(other_areas)
        if max_area / max_other >= min_ratio:
            # Main part is >10x larger - select it
            return parts[areas.index(max_area)]

    # Keep only significant parts
    threshold_area = max_area * area_threshold
    significant_parts = [
        part for part, area in zip(parts, areas) if area >= threshold_area
    ]

    if len(significant_parts) == 1:
        return significant_parts[0]
    elif len(significant_parts) > 1:
        result = unary_union(significant_parts)
        if isinstance(result, Polygon):
            return result
        elif isinstance(result, MultiPolygon):
            return max(result.geoms, key=lambda p: p.area)

    return parts[areas.index(max_area)]


class DistrictFetcher:
    """
    Fetches and processes Warsaw district boundaries from OpenStreetMap.

    Uses the Overpass API with simplified query (no recursion) and
    the Manual Stitching algorithm for polygon construction.

    Example:
        >>> fetcher = DistrictFetcher()
        >>> gdf = fetcher.fetch_and_process()
        >>> print(len(gdf))  # 18 districts
        18
        >>> fetcher.save_cache()
    """

    # Polish district names with diacritics
    DISTRICT_NAMES = [
        "Bemowo",
        "Białołęka",
        "Bielany",
        "Mokotów",
        "Ochota",
        "Praga-Południe",
        "Praga-Północ",
        "Rembertów",
        "Targówek",
        "Ursus",
        "Ursynów",
        "Wawer",
        "Wesoła",
        "Wilanów",
        "Wola",
        "Włochy",
        "Śródmieście",
        "Żoliborz",
    ]

    # ASCII versions for compatibility
    DISTRICT_NAMES_ASCII = [
        "Bemowo",
        "Bialoleka",
        "Bielany",
        "Mokotow",
        "Ochota",
        "Praga-Poludnie",
        "Praga-Polnoc",
        "Rembertow",
        "Targowek",
        "Ursus",
        "Ursynow",
        "Wawer",
        "Wesola",
        "Wilanow",
        "Wola",
        "Wlochy",
        "Srodmiescie",
        "Zoliborz",
    ]

    # Mapping from Polish to ASCII
    NAME_MAPPING = dict(zip(DISTRICT_NAMES, DISTRICT_NAMES_ASCII))

    def __init__(
        self,
        api_timeout: int = 60,
        rate_limit_delay: float = 1.5,
        crs_metric: str = "EPSG:32634",
        crs_geographic: str = "EPSG:4326",
    ):
        """
        Initialize the fetcher.

        Args:
            api_timeout: Timeout for API requests in seconds
            rate_limit_delay: Delay between API requests
            crs_metric: Metric CRS for area calculations (UTM 34N for Poland)
            crs_geographic: Geographic CRS (WGS84)
        """
        self.api_timeout = api_timeout
        self.rate_limit_delay = rate_limit_delay
        self.crs_metric = crs_metric
        self.crs_geographic = crs_geographic

        self.raw_data: Optional[Dict[str, Any]] = None
        self.gdf: Optional[gpd.GeoDataFrame] = None

        logger.debug(f"Initialized DistrictFetcher with timeout={api_timeout}s")

    def fetch_from_api(self) -> Optional[Dict[str, Any]]:
        """
        Fetch district data from Overpass API.
        ...
        """
        logger.info("Fetching district boundaries from Overpass API...")

        query = """
        [out:json][timeout:60];
        area["name"="Warszawa"]->.a;
        relation["boundary"="administrative"]["admin_level"="9"](area.a);
        out geom;
        """

        try:
            response = requests.post(
                settings.api.overpass_url,
                data={"data": query},
                timeout=self.api_timeout,
            )

            if response.status_code == 200:
                data = response.json()
                n_elements = len(data.get("elements", []))
                logger.info(f"Retrieved {n_elements} elements from API")
                self.raw_data = data

                # FIX: Explicitly cast the JSON result to the expected Dict type
                return cast(Dict[str, Any], data)
            else:
                logger.error(f"API returned HTTP {response.status_code}")
                return None

        except requests.Timeout:
            logger.error(f"API request timed out after {self.api_timeout}s")
            return None
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {e}")
            return None

    def _extract_lines_from_relation(
        self, relation: Dict[str, Any]
    ) -> List[LineString]:
        """
        Extract line segments from an Overpass relation.

        Args:
            relation: Relation dictionary from API response

        Returns:
            List of LineString representing boundaries
        """
        lines: List[LineString] = []

        for member in relation.get("members", []):
            # Only "outer" ways (external boundaries)
            if member["type"] == "way" and "geometry" in member:
                if member.get("role") in ["outer", ""]:
                    coords = [(p["lon"], p["lat"]) for p in member["geometry"]]
                    if len(coords) >= 2:
                        try:
                            lines.append(LineString(coords))
                        except Exception:
                            pass

        return lines

    def _normalize_name(self, name: str) -> str:
        """
        Normalize district name to ASCII version.

        Args:
            name: District name (possibly with Polish diacritics)

        Returns:
            ASCII version of the name
        """
        return self.NAME_MAPPING.get(name, name)

    def process_raw_data(
        self, data: Optional[Dict[str, Any]] = None, verbose: bool = True
    ) -> gpd.GeoDataFrame:
        """
        Process raw API data using the stitch algorithm.

        Args:
            data: Optional data (uses self.raw_data if None)
            verbose: Whether to show progress

        Returns:
            GeoDataFrame with processed districts

        Raises:
            ValueError: If no data available
        """
        if data is None:
            if self.raw_data is None:
                raise ValueError("No data to process. Call fetch_from_api() first.")
            data = self.raw_data

        if verbose:
            logger.info("Processing districts (algorithm: manual stitching)...")

        districts: List[Dict[str, Any]] = []
        elements = data.get("elements", [])

        for element in elements:
            if element.get("type") != "relation":
                continue
            if element.get("tags", {}).get("admin_level") != "9":
                continue

            name_pl = element.get("tags", {}).get("name", "UNKNOWN")
            name = self._normalize_name(name_pl)

            # Extract lines
            lines = self._extract_lines_from_relation(element)
            if not lines:
                if verbose:
                    logger.warning(f"{name}: no line segments found")
                continue

            # STITCH ALGORITHM
            geom = stitch_line_segments(lines, verbose=verbose)

            if geom is None:
                if verbose:
                    logger.warning(f"{name}: failed to create geometry")
                continue

            # Validation
            if not geom.is_valid:
                geom = geom.buffer(0)

            # Clean MultiPolygon (fallback)
            if isinstance(geom, MultiPolygon):
                geom = clean_multipolygon(geom)

            districts.append(
                {
                    "name": name,
                    "name_pl": name_pl,
                    "geometry": geom,
                    "num_segments": len(lines),
                    "geom_type": geom.geom_type,
                }
            )

            if verbose:
                logger.debug(f"  ✓ {name}: {geom.geom_type} ({len(lines)} segments)")

        # Create GeoDataFrame
        self.gdf = gpd.GeoDataFrame(districts, crs=self.crs_geographic)

        # Calculate areas in metric CRS
        gdf_metric = self.gdf.to_crs(self.crs_metric)
        self.gdf["area_km2"] = (gdf_metric.geometry.area / 1_000_000).round(2)

        # Calculate centroids
        self.gdf["centroid_lat"] = self.gdf.geometry.centroid.y
        self.gdf["centroid_lon"] = self.gdf.geometry.centroid.x

        # Calculate area errors vs reference
        errors = []
        for _, row in self.gdf.iterrows():
            ref_area = settings.warsaw.reference_areas.get(row["name"])
            if ref_area:
                error = abs(row["area_km2"] - ref_area) / ref_area * 100
                errors.append(round(error, 1))
            else:
                errors.append(None)
        self.gdf["area_error_pct"] = errors

        logger.info(f"Processed {len(self.gdf)} districts successfully")

        return self.gdf

    def fetch_and_process(self, verbose: bool = True) -> gpd.GeoDataFrame:
        """
        Convenience method to fetch and process in one call.

        Args:
            verbose: Whether to show progress

        Returns:
            GeoDataFrame with processed districts
        """
        self.fetch_from_api()
        return self.process_raw_data(verbose=verbose)

    def load_from_cache(self, cache_path: Optional[Path] = None) -> gpd.GeoDataFrame:
        """
        Load district boundaries from cached GeoJSON file.

        Args:
            cache_path: Path to cache file (uses default if None)

        Returns:
            GeoDataFrame with districts

        Raises:
            FileNotFoundError: If cache file doesn't exist
        """
        path = cache_path or settings.districts_cache_path

        if not path.exists():
            raise FileNotFoundError(f"Cache file not found: {path}")

        logger.info(f"Loading districts from cache: {path}")
        self.gdf = gpd.read_file(path)

        return self.gdf

    def save_cache(self, cache_path: Optional[Path] = None) -> Path:
        """
        Save district boundaries to GeoJSON cache file.

        Args:
            cache_path: Path to save (uses default if None)

        Returns:
            Path to saved file

        Raises:
            ValueError: If no data to save
        """
        if self.gdf is None:
            raise ValueError("No data to save. Fetch and process first.")

        path = cache_path or settings.districts_cache_path
        path.parent.mkdir(parents=True, exist_ok=True)

        self.gdf.to_file(path, driver="GeoJSON")
        logger.info(f"Saved district cache to: {path}")

        return path

    def get_or_fetch(self, force_refresh: bool = False) -> gpd.GeoDataFrame:
        """
        Get districts from cache or fetch from API if not available.

        This is the recommended method for most use cases.

        Args:
            force_refresh: If True, always fetch from API

        Returns:
            GeoDataFrame with districts
        """
        cache_path = settings.districts_cache_path

        if not force_refresh and cache_path.exists():
            try:
                return self.load_from_cache(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        # Fetch from API
        gdf = self.fetch_and_process()

        # Save to cache
        try:
            self.save_cache(cache_path)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

        return gdf

    def get_district_by_point(self, lat: float, lon: float) -> Optional[str]:
        """
        Find which district contains the given point.
        ...
        """
        if self.gdf is None:
            raise ValueError("No data loaded. Call fetch_and_process() first.")

        point = Point(lon, lat)

        for _, row in self.gdf.iterrows():
            if row.geometry.contains(point):
                # FIX: Explicitly cast the row value to string
                return cast(str, row["name"])

        return None

    def get_polygon(self, district_name: str) -> Optional[Polygon]:
        """
        Get the polygon geometry for a specific district.

        Args:
            district_name: District name (ASCII or Polish)

        Returns:
            Polygon geometry or None if not found
        """
        if self.gdf is None:
            raise ValueError("No data loaded. Call fetch_and_process() first.")

        # Try ASCII name
        matches = self.gdf[self.gdf["name"] == district_name]
        if len(matches) == 0:
            # Try Polish name
            matches = self.gdf[self.gdf["name_pl"] == district_name]

        if len(matches) > 0:
            return matches.iloc[0].geometry

        return None

    def get_summary(self) -> str:
        """
        Get a text summary of loaded districts.

        Returns:
            Formatted summary string
        """
        if self.gdf is None:
            return "No data loaded."

        lines = [
            "=" * 60,
            "WARSAW DISTRICTS SUMMARY",
            "=" * 60,
            f"Total districts: {len(self.gdf)}",
            f"Total area: {self.gdf['area_km2'].sum():.1f} km²",
            f"Polygon: {len(self.gdf[self.gdf['geom_type'] == 'Polygon'])}",
            f"MultiPolygon: {len(self.gdf[self.gdf['geom_type'] == 'MultiPolygon'])}",
            "",
            "Districts by area:",
            "-" * 40,
        ]

        for _, row in self.gdf.sort_values("area_km2", ascending=False).iterrows():
            error_str = (
                f"{row['area_error_pct']:.1f}%"
                if pd.notna(row["area_error_pct"])
                else "N/A"
            )
            lines.append(
                f"  {row['name']:<20} {row['area_km2']:>6.2f} km² (error: {error_str})"
            )

        return "\n".join(lines)
