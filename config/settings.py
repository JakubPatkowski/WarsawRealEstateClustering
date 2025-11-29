"""
Central configuration module for Warsaw Real Estate Clustering.

Contains all settings, constants, and project-wide configuration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class WarsawConfig:
    """Warsaw geographical configuration."""

    center: Tuple[float, float] = (52.2297, 21.0122)
    bbox: Dict[str, float] = field(
        default_factory=lambda: {
            "minx": 20.85,
            "maxx": 21.30,
            "miny": 52.10,
            "maxy": 52.37,
        }
    )

    # District centers (approximate) - used as fallback
    districts: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "Bemowo": (52.2543, 20.9127),
            "Bialoleka": (52.3292, 20.9875),
            "Bielany": (52.2917, 20.9347),
            "Mokotow": (52.1873, 21.0478),
            "Ochota": (52.2119, 20.9817),
            "Praga-Poludnie": (52.2364, 21.0806),
            "Praga-Polnoc": (52.2575, 21.0381),
            "Rembertow": (52.2611, 21.1714),
            "Srodmiescie": (52.2297, 21.0122),
            "Targowek": (52.2967, 21.0533),
            "Ursus": (52.1919, 20.8744),
            "Ursynow": (52.1486, 21.0344),
            "Wawer": (52.1772, 21.1578),
            "Wesola": (52.2456, 21.2253),
            "Wilanow": (52.1553, 21.0889),
            "Wlochy": (52.1961, 20.9083),
            "Wola": (52.2381, 20.9669),
            "Zoliborz": (52.2692, 20.9819),
        }
    )

    # Reference areas for validation (km²)
    reference_areas: Dict[str, float] = field(
        default_factory=lambda: {
            "Bemowo": 24.95,
            "Bialoleka": 73.04,
            "Bielany": 32.30,
            "Mokotow": 35.40,
            "Ochota": 9.70,
            "Praga-Poludnie": 22.40,
            "Praga-Polnoc": 11.40,
            "Rembertow": 19.30,
            "Srodmiescie": 15.60,
            "Targowek": 24.20,
            "Ursus": 9.35,
            "Ursynow": 43.80,
            "Wawer": 79.71,
            "Wesola": 22.60,
            "Wilanow": 36.70,
            "Wlochy": 28.60,
            "Wola": 19.26,
            "Zoliborz": 8.50,
        }
    )


@dataclass
class ClusteringConfig:
    """Clustering algorithm configuration."""

    k_range: Tuple[int, int] = (3, 10)
    random_state: int = 42
    feature_columns: List[str] = field(
        default_factory=lambda: ["price_per_sqm", "area_sqm", "distance_from_center_km"]
    )
    extended_features: List[str] = field(
        default_factory=lambda: [
            "price_per_sqm",
            "area_sqm",
            "distance_from_center_km",
            "year_built",
            "floor",
        ]
    )
    scaling_method: str = "standard"

    # DBSCAN parameters
    dbscan_min_samples: int = 5
    dbscan_eps_percentile: float = 90.0


@dataclass
class DataConfig:
    """Data generation and processing configuration."""

    n_samples: int = 500
    seed: int = 42
    min_per_district: int = 15

    # Outlier detection
    outlier_method: str = "iqr"
    outlier_threshold: float = 2.5

    # Price multipliers by district (relative to Warsaw average)
    district_price_multipliers: Dict[str, float] = field(
        default_factory=lambda: {
            "Srodmiescie": 1.25,
            "Mokotow": 1.15,
            "Zoliborz": 1.10,
            "Wilanow": 1.08,
            "Ochota": 1.05,
            "Wola": 1.02,
            "Bielany": 0.98,
            "Praga-Poludnie": 0.95,
            "Ursynow": 0.95,
            "Bemowo": 0.92,
            "Targowek": 0.88,
            "Wlochy": 0.88,
            "Wawer": 0.85,
            "Praga-Polnoc": 0.85,
            "Bialoleka": 0.82,
            "Ursus": 0.82,
            "Wesola": 0.78,
            "Rembertow": 0.75,
        }
    )


@dataclass
class APIConfig:
    """API configuration."""

    overpass_url: str = "https://overpass-api.de/api/interpreter"
    overpass_timeout: int = 60
    rate_limit_delay: float = 1.5
    max_retries: int = 3

    gus_base_url: str = "https://bdl.stat.gov.pl/api/v1"
    uldk_base_url: str = "https://uldk.gugik.gov.pl"


@dataclass
class OutputConfig:
    """Output and visualization configuration."""

    map_zoom: int = 11
    map_tiles: str = "CartoDB positron"
    dpi: int = 150
    figure_size: Tuple[int, int] = (12, 10)

    # Color schemes
    cluster_colormap: str = "viridis"
    district_colormap: str = "YlOrRd"


@dataclass
class Settings:
    """Main settings container."""

    warsaw: WarsawConfig = field(default_factory=WarsawConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    data: DataConfig = field(default_factory=DataConfig)
    api: APIConfig = field(default_factory=APIConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Paths
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        path = self.data_dir / "raw"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def processed_dir(self) -> Path:
        path = self.data_dir / "processed"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cache_dir(self) -> Path:
        path = self.data_dir / "cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def boundaries_dir(self) -> Path:
        path = self.data_dir / "boundaries"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def output_dir(self) -> Path:
        path = PROJECT_ROOT / "outputs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def maps_dir(self) -> Path:
        path = self.output_dir / "maps"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def plots_dir(self) -> Path:
        path = self.output_dir / "plots"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def reports_dir(self) -> Path:
        path = self.output_dir / "reports"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def districts_cache_path(self) -> Path:
        return self.boundaries_dir / "districts_cache.geojson"


# Global settings instance
settings = Settings()
