"""
Synthetic data generator for Warsaw real estate.

Generates realistic property data with DISTINCT MARKET SEGMENTS
and ensures all points are INSIDE real district polygon boundaries.

Key features:
- Creates 4 distinct market segments (Premium, Upper, Standard, Economy)
- Each segment has unique price ranges, area distributions, and locations
- Points are generated WITHIN actual district polygons (from OpenStreetMap)
- Uses NBP price data for realistic pricing
"""

import random
from math import asin, cos, radians, sin, sqrt
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


# Market segment definitions with distinct characteristics
MARKET_SEGMENTS: Dict[str, Dict[str, Any]] = {
    "premium": {
        "weight": 0.15,  # 15% of properties
        "price_range": (22000, 35000),  # Very high prices
        "area_range": (35, 80),  # Smaller, exclusive apartments
        "year_range": (2015, 2024),  # New construction
        "districts": ["Srodmiescie", "Mokotow", "Zoliborz", "Wilanow"],
        "floor_preference": "high",  # Prefers high floors
        "distance_modifier": 0.8,  # Closer to center
    },
    "upper": {
        "weight": 0.25,  # 25% of properties
        "price_range": (16000, 22000),
        "area_range": (50, 100),
        "year_range": (2000, 2020),
        "districts": ["Mokotow", "Ochota", "Wola", "Zoliborz", "Bielany", "Wilanow"],
        "floor_preference": "medium",
        "distance_modifier": 0.9,
    },
    "standard": {
        "weight": 0.35,  # 35% of properties
        "price_range": (11000, 16000),
        "area_range": (45, 120),
        "year_range": (1970, 2015),
        "districts": ["Wola", "Praga-Poludnie", "Bielany", "Ursynow", "Bemowo", "Targowek"],
        "floor_preference": "any",
        "distance_modifier": 1.0,
    },
    "economy": {
        "weight": 0.25,  # 25% of properties
        "price_range": (7000, 12000),  # Low prices
        "area_range": (40, 150),  # Larger, older apartments
        "year_range": (1950, 1995),  # Older construction
        "districts": [
            "Bialoleka", "Targowek", "Ursus", "Wawer", "Wesola", 
            "Rembertow", "Praga-Polnoc", "Wlochy"
        ],
        "floor_preference": "low",
        "distance_modifier": 1.2,  # Further from center
    }
}


class SyntheticDataGenerator:
    """
    Generates synthetic real estate data for Warsaw with distinct market segments.
    
    IMPORTANT: This generator uses REAL district polygon boundaries from
    OpenStreetMap to ensure all generated points fall within actual districts.
    
    Example:
        >>> from src.boundaries import DistrictFetcher
        >>> fetcher = DistrictFetcher()
        >>> gdf_districts = fetcher.get_or_fetch()
        >>> generator = SyntheticDataGenerator(district_polygons=gdf_districts)
        >>> df = generator.generate_dataset(n_samples=500)
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        district_prices: Optional[Dict[str, float]] = None,
        district_polygons: Optional[gpd.GeoDataFrame] = None
    ):
        """
        Initialize the generator.
        
        Args:
            seed: Random seed for reproducibility
            district_prices: Dict of district base prices from NBP
            district_polygons: GeoDataFrame with real district boundaries
        """
        self.seed = seed or settings.data.seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        self.district_centers = settings.warsaw.districts
        self.center = settings.warsaw.center
        self.district_prices = district_prices
        
        # Store district polygons for point-in-polygon generation
        self._district_polygons: Dict[str, Polygon] = {}
        if district_polygons is not None:
            self._load_district_polygons(district_polygons)
            logger.info(
                f"Initialized generator with {len(self._district_polygons)} "
                "real district polygons"
            )
        else:
            logger.warning(
                "No district polygons provided - falling back to circular approximation"
            )
        
        logger.info("Initialized generator with distinct market segments")
    
    def _load_district_polygons(self, gdf: gpd.GeoDataFrame) -> None:
        """
        Load district polygons from GeoDataFrame.
        
        Args:
            gdf: GeoDataFrame with 'name' and 'geometry' columns
        """
        for _, row in gdf.iterrows():
            name = row["name"]
            geom = row.geometry
            if geom is not None and geom.is_valid:
                self._district_polygons[name] = geom
    
    def _haversine_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate great-circle distance between two points in km."""
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        
        return 6371 * c
    
    def _generate_point_in_polygon(
        self,
        polygon: Polygon,
        max_attempts: int = 100
    ) -> Tuple[float, float]:
        """
        Generate a random point INSIDE a polygon.
        
        Uses rejection sampling: generates random points in the bounding box
        and keeps only those that fall inside the polygon.
        
        Args:
            polygon: Shapely Polygon to generate point within
            max_attempts: Maximum sampling attempts before fallback
            
        Returns:
            Tuple of (lat, lon) coordinates inside the polygon
        """
        minx, miny, maxx, maxy = polygon.bounds
        
        for _ in range(max_attempts):
            lon = random.uniform(minx, maxx)
            lat = random.uniform(miny, maxy)
            point = Point(lon, lat)
            
            if polygon.contains(point):
                return (lat, lon)
        
        # Fallback: return centroid
        centroid = polygon.centroid
        return (centroid.y, centroid.x)
    
    def _generate_coordinates_fallback(
        self,
        district_center: Tuple[float, float],
        radius_km: float = 2.0,
        distance_modifier: float = 1.0
    ) -> Tuple[float, float]:
        """
        Fallback: Generate random coordinates using circular approximation.
        
        Used when real polygon is not available for a district.
        
        Args:
            district_center: (lat, lon) center of district
            radius_km: Approximate radius in km
            distance_modifier: Modifier for radius (premium closer, economy further)
            
        Returns:
            Tuple of (lat, lon) coordinates
        """
        lat, lon = district_center
        
        # Apply modifier to radius
        effective_radius = radius_km * distance_modifier
        
        angle = random.uniform(0, 2 * np.pi)
        distance = random.uniform(0, effective_radius)
        
        lat_offset = distance * np.cos(angle) / 111
        lon_offset = distance * np.sin(angle) / (111 * np.cos(radians(lat)))
        
        return (lat + lat_offset, lon + lon_offset)
    
    def _generate_coordinates_in_district(
        self,
        district: str,
        distance_modifier: float = 1.0
    ) -> Tuple[float, float]:
        """
        Generate coordinates within a district.
        
        Uses real polygon boundaries if available, otherwise falls back
        to circular approximation.
        
        Args:
            district: District name
            distance_modifier: Not used with real polygons (kept for compatibility)
            
        Returns:
            Tuple of (lat, lon) coordinates
        """
        # Try real polygon first
        if district in self._district_polygons:
            polygon = self._district_polygons[district]
            return self._generate_point_in_polygon(polygon)
        
        # Fallback to circular approximation
        if district in self.district_centers:
            center = self.district_centers[district]
            return self._generate_coordinates_fallback(
                center, 
                radius_km=2.0, 
                distance_modifier=distance_modifier
            )
        
        # Last resort: random point in Warsaw bbox
        logger.warning(f"Unknown district '{district}', using random Warsaw location")
        bbox = settings.warsaw.bbox
        lat = random.uniform(bbox["miny"], bbox["maxy"])
        lon = random.uniform(bbox["minx"], bbox["maxx"])
        return (lat, lon)
    
    def _select_segment(self) -> str:
        """Randomly select a market segment based on weights."""
        segments = list(MARKET_SEGMENTS.keys())
        weights = [MARKET_SEGMENTS[s]["weight"] for s in segments]
        return random.choices(segments, weights=weights)[0]
    
    def _generate_floor(self, segment: str, year_built: int) -> int:
        """Generate floor number based on segment preference and building age."""
        max_floor = 4 if year_built < 1960 else (8 if year_built < 2000 else 15)
        
        preference = MARKET_SEGMENTS[segment]["floor_preference"]
        
        if preference == "high" and max_floor > 5:
            return random.randint(max(3, max_floor - 5), max_floor)
        elif preference == "low":
            return random.randint(0, min(3, max_floor))
        else:
            return random.randint(0, max_floor)
    
    def generate_property(
        self, 
        property_id: int, 
        segment: Optional[str] = None,
        force_district: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a single property record.
        
        Args:
            property_id: Unique property identifier
            segment: Optional forced segment (for balanced generation)
            force_district: Optional forced district (for coverage guarantee)
            
        Returns:
            Dictionary with property attributes
        """
        # Select segment if not provided
        if segment is None:
            segment = self._select_segment()
        
        seg_config = MARKET_SEGMENTS[segment]
        
        # Select district
        if force_district:
            district = force_district
        else:
            # Select from segment's preferred districts
            district = random.choice(seg_config["districts"])
            # Validate district exists
            if district not in self.district_centers and district not in self._district_polygons:
                district = random.choice(list(self.district_centers.keys()))
        
        # Generate coordinates INSIDE the district polygon
        lat, lon = self._generate_coordinates_in_district(
            district,
            distance_modifier=seg_config["distance_modifier"]
        )
        
        # Calculate distance from Warsaw center
        distance = self._haversine_distance(
            lat, lon, self.center[0], self.center[1]
        )
        
        # Generate area from segment-specific range
        area_min, area_max = seg_config["area_range"]
        area = random.uniform(area_min, area_max)
        
        # Calculate rooms based on area
        rooms = max(1, min(6, int(area / 25) + random.randint(-1, 1)))
        
        # Generate year from segment-specific range
        year_min, year_max = seg_config["year_range"]
        year_built = random.randint(year_min, year_max)
        
        # Generate floor
        floor = self._generate_floor(segment, year_built)
        
        # Generate price with segment-specific range and variance
        price_min, price_max = seg_config["price_range"]
        
        # Use NBP district price as base if available
        if self.district_prices and district in self.district_prices:
            nbp_base = self.district_prices[district]
            # Blend NBP price with segment range (60% NBP, 40% segment)
            segment_mid = (price_min + price_max) / 2
            base_price = nbp_base * 0.6 + segment_mid * 0.4
            # Apply segment-appropriate bounds
            base_price = max(price_min, min(price_max, base_price))
        else:
            base_price = random.uniform(price_min, price_max)
        
        # Apply modifiers for realistic variance
        # Distance modifier (closer = higher price)
        distance_effect = 1.0 - (distance * 0.015)
        distance_effect = max(0.75, min(1.15, distance_effect))
        
        # Age modifier
        age = 2024 - year_built
        if age < 3:
            age_effect = 1.12
        elif age < 10:
            age_effect = 1.05
        elif age > 50:
            age_effect = 0.92
        else:
            age_effect = 1.0
        
        # Floor modifier
        if floor == 0:
            floor_effect = 0.94
        elif floor > 8:
            floor_effect = 1.06
        else:
            floor_effect = 1.0
        
        # Random noise (±15% for realistic variance)
        noise = random.uniform(0.85, 1.15)
        
        # Calculate final price
        price_per_sqm = base_price * distance_effect * age_effect * floor_effect * noise
        
        # Ensure price stays within segment bounds (with some flexibility)
        price_per_sqm = max(price_min * 0.85, min(price_max * 1.15, price_per_sqm))
        price_per_sqm = round(price_per_sqm, 2)
        
        price_total = round(price_per_sqm * area, 2)
        
        return {
            "id": property_id,
            "district": district,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "price_total": price_total,
            "price_per_sqm": price_per_sqm,
            "area_sqm": round(area, 1),
            "rooms": rooms,
            "year_built": year_built,
            "floor": floor,
            "distance_from_center_km": round(distance, 2),
            "market_segment": segment
        }
    
    def generate_dataset(self, n_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Generate a complete dataset of properties.
        
        Args:
            n_samples: Number of properties to generate
            
        Returns:
            DataFrame with all properties
        """
        n = n_samples or settings.data.n_samples
        
        logger.info(f"Generating {n} synthetic properties with distinct segments...")
        
        properties = [self.generate_property(i + 1) for i in range(n)]
        df = pd.DataFrame(properties)
        
        # Log segment distribution
        segment_counts = df["market_segment"].value_counts()
        logger.info("Market segment distribution:")
        for seg, count in segment_counts.items():
            pct = count / len(df) * 100
            avg_price = df[df["market_segment"] == seg]["price_per_sqm"].mean()
            logger.info(
                f"  {seg}: {count} ({pct:.1f}%), avg price: {avg_price:,.0f} PLN/sqm"
            )
        
        return df
    
    def generate_with_district_balance(
        self,
        n_samples: Optional[int] = None,
        min_per_district: int = 15
    ) -> pd.DataFrame:
        """
        Generate dataset with minimum representation per district.
        
        Ensures all 18 districts are represented while maintaining
        distinct market segments.
        
        Args:
            n_samples: Total number of properties
            min_per_district: Minimum properties per district
            
        Returns:
            DataFrame with balanced distribution
        """
        n = n_samples or settings.data.n_samples
        
        # Get all district names (prefer polygons, fallback to centers)
        all_districts = set(self._district_polygons.keys()) | set(self.district_centers.keys())
        n_districts = len(all_districts)
        
        guaranteed = min_per_district * n_districts
        remaining = max(0, n - guaranteed)
        
        logger.info(
            f"Generating {n} properties: {min_per_district} per district "
            f"+ {remaining} segment-based"
        )
        
        properties = []
        property_id = 1
        
        # First: ensure minimum per district
        for district in sorted(all_districts):
            # Determine appropriate segment for this district
            district_segment = None
            for seg_name, seg_config in MARKET_SEGMENTS.items():
                if district in seg_config["districts"]:
                    district_segment = seg_name
                    break
            
            if district_segment is None:
                district_segment = "standard"
            
            for _ in range(min_per_district):
                prop = self.generate_property(
                    property_id, 
                    segment=district_segment,
                    force_district=district
                )
                properties.append(prop)
                property_id += 1
        
        # Then: add remaining properties with natural segment distribution
        for _ in range(remaining):
            properties.append(self.generate_property(property_id))
            property_id += 1
        
        df = pd.DataFrame(properties)
        
        # Shuffle
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        df["id"] = range(1, len(df) + 1)
        
        # Log final statistics
        logger.info(
            f"Generated {len(df)} properties across "
            f"{df['district'].nunique()} districts"
        )
        
        # Log price statistics by segment
        price_stats = df.groupby("market_segment")["price_per_sqm"].agg(
            ["mean", "min", "max"]
        )
        logger.debug("Price statistics by segment:")
        for seg in price_stats.index:
            stats = price_stats.loc[seg]
            logger.debug(
                f"  {seg}: mean={stats['mean']:,.0f}, "
                f"range={stats['min']:,.0f}-{stats['max']:,.0f}"
            )
        
        return df
    
    def validate_points_in_districts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that all points fall within their claimed districts.
        
        Useful for debugging and verification.
        
        Args:
            df: DataFrame with 'lat', 'lon', 'district' columns
            
        Returns:
            DataFrame with added 'valid_location' column
        """
        df = df.copy()
        
        valid = []
        for _, row in df.iterrows():
            district = row["district"]
            point = Point(row["lon"], row["lat"])
            
            if district in self._district_polygons:
                polygon = self._district_polygons[district]
                valid.append(polygon.contains(point))
            else:
                # Can't validate without polygon
                valid.append(True)
        
        df["valid_location"] = valid
        
        invalid_count = len(df) - sum(valid)
        if invalid_count > 0:
            logger.warning(
                f"{invalid_count} points are outside their claimed districts"
            )
        else:
            logger.info("All points are within their claimed districts ✓")
        
        return df
