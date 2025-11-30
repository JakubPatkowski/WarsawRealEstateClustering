"""Settings and configuration for Warsaw Real Estate Clustering."""

from pathlib import Path
from typing import Dict, List, Tuple


class WarsawSettings:
    """Warsaw-specific geographic settings."""
    
    center: Tuple[float, float] = (52.2297, 21.0122)

    districts: Dict[str, Tuple[float, float]] = {
        "Śródmieście": (52.2319, 21.0067),
        "Mokotów": (52.1880, 21.0329),
        "Wola": (52.2389, 20.9783),
        "Praga-Południe": (52.2392, 21.0828),
        "Praga-Północ": (52.2558, 21.0378),
        "Żoliborz": (52.2719, 20.9814),
        "Bielany": (52.2900, 20.9350),
        "Bemowo": (52.2550, 20.9050),
        "Ursynów": (52.1440, 21.0380),
        "Ochota": (52.2130, 20.9720),
        "Targówek": (52.2817, 21.0514),
        "Białołęka": (52.3200, 20.9900),
        "Ursus": (52.1940, 20.8950),
        "Włochy": (52.1870, 20.9200),
        "Wawer": (52.1850, 21.1450),
        "Rembertów": (52.2550, 21.1700),
        "Wesoła": (52.2550, 21.2200),
        "Wilanów": (52.1530, 21.0900),
    }


class DataSettings:
    """Data generation settings."""
    
    seed: int = 42
    default_n_samples: int = 500
    min_per_district: int = 15


class ClusteringSettings:
    """Clustering algorithm settings."""
    random_state = 42
    feature_columns: List[str] = ["price_per_sqm", "area_sqm", "distance_from_center_km"]
    extended_features: List[str] = [
        "price_per_sqm", "area_sqm", "distance_from_center_km", "year_built", "floor"
    ]
    k_range: Tuple[int, int] = (2, 8)
    default_k: int = 4


class Settings:
    """Main settings container."""
    warsaw_center: Tuple[float, float] = (52.2297, 21.0122)

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.boundaries_dir = self.data_dir / "boundaries"
        
        self.output_dir = self.project_root / "output"
        self.maps_dir = self.output_dir / "maps"
        self.plots_dir = self.output_dir / "plots"
        self.reports_dir = self.output_dir / "reports"
        
        self.districts_cache_path = self.boundaries_dir / "districts_cache.geojson"
        
        self.warsaw = WarsawSettings()
        self.data = DataSettings()
        self.clustering = ClusteringSettings()
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.boundaries_dir,
                         self.maps_dir, self.plots_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
