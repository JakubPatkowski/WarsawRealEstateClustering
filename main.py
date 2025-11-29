#!/usr/bin/env python3
"""
Warsaw Real Estate Price Clustering - Main Execution Script

This script orchestrates the complete analysis pipeline:
1. Fetch real district boundaries from OpenStreetMap
2. Generate synthetic data INSIDE real district polygons
3. Clean and validate data
4. Run K-means/DBSCAN clustering with optimization
5. Generate visualizations (maps, plots)
6. Create HTML report

Usage:
    python main.py [options]

Examples:
    python main.py -v                     # Verbose mode
    python main.py -d                     # Debug mode
    python main.py --k 5                  # Force 5 clusters
    python main.py --extended-features    # Use more features
    python main.py --compare-dbscan       # Compare with DBSCAN
    python main.py --force-refresh        # Re-fetch district boundaries
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config.logging_config import get_logger, setup_debug, setup_logging, setup_verbose
from config.settings import settings
from src.analysis import (
    ClusterOptimizer,
    ClusterStatistics,
    DBSCANClusterer,
    KMeansClusterer,
)
from src.boundaries import DistrictFetcher
from src.data_collection import SyntheticDataGenerator
from src.data_processing import DataCleaner, FeatureTransformer

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Warsaw Real Estate Price Clustering Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -v                     Basic run with verbose output
  python main.py -d                     Debug mode with full logging
  python main.py --k 5                  Force 5 clusters
  python main.py --extended-features    Use all features (price, area, distance, year, floor)
  python main.py --compare-dbscan       Compare K-means with DBSCAN
  python main.py --force-refresh        Re-fetch district boundaries from OSM
        """,
    )

    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh of district boundaries from OSM",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of clusters (auto-detect if not specified)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory for results"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug logging (very detailed)",
    )
    parser.add_argument(
        "--extended-features",
        action="store_true",
        help="Use extended features: price, area, distance, year_built, floor",
    )
    parser.add_argument(
        "--compare-dbscan",
        action="store_true",
        help="Compare K-means results with DBSCAN",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of properties to generate (default: 500)",
    )

    return parser.parse_args()


def fetch_district_boundaries(
    force_refresh: bool = False, logger: Optional[logging.Logger] = None
) -> Tuple[DistrictFetcher, Any]:
    """
    Fetch district boundaries from cache or OSM.

    Args:
        force_refresh: Force re-fetch from API
        logger: Logger instance

    Returns:
        Tuple of (fetcher, GeoDataFrame)
    """
    if logger is None:
        logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("PHASE 1: DISTRICT BOUNDARIES")
    logger.info("=" * 60)

    fetcher = DistrictFetcher()

    cache_path = settings.districts_cache_path

    if not force_refresh and cache_path.exists():
        logger.info(f"Loading district boundaries from cache: {cache_path}")
        gdf_districts = fetcher.load_from_cache(cache_path)
        logger.info(f"Loaded {len(gdf_districts)} districts from cache")
    else:
        logger.info("Fetching district boundaries from OpenStreetMap...")
        gdf_districts = fetcher.fetch_and_process(verbose=True)

        # Save to cache
        fetcher.save_cache(cache_path)
        logger.info(f"Saved {len(gdf_districts)} districts to cache")

    # Log summary
    logger.info("\nDistrict summary:")
    for _, row in gdf_districts.iterrows():
        logger.info(f"  {row['name']}: {row['area_km2']:.1f} km²")

    return fetcher, gdf_districts


def generate_properties(
    gdf_districts: Any,
    n_samples: int,
    debug: bool = False,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Generate synthetic property data inside real district boundaries.

    Args:
        gdf_districts: GeoDataFrame with district polygons
        n_samples: Number of properties to generate
        debug: Enable debug logging
        logger: Logger instance

    Returns:
        DataFrame with property data
    """
    if logger is None:
        logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("PHASE 2: DATA GENERATION")
    logger.info("=" * 60)

    # Initialize generator with real district polygons
    generator = SyntheticDataGenerator(
        seed=settings.data.seed, district_polygons=gdf_districts
    )

    # Generate properties with balanced district distribution
    df = generator.generate_with_district_balance(
        n_samples=n_samples, min_per_district=15
    )

    logger.info(
        f"Generated {len(df)} properties across {df['district'].nunique()} districts"
    )

    if debug:
        logger.debug("Sample data:")
        for _, row in df.head().iterrows():
            logger.debug(
                f"  {row['district']}: {row['price_per_sqm']:,.0f} PLN/m², "
                f"{row['area_sqm']:.0f} m², {row['market_segment']}"
            )

    # Validate points are inside districts
    df = generator.validate_points_in_districts(df)

    return df


def process_data(
    df: pd.DataFrame, logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Clean and validate data.

    Args:
        df: Raw DataFrame
        debug: Enable debug logging
        logger: Logger instance

    Returns:
        Cleaned DataFrame
    """
    if logger is None:
        logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("PHASE 3: DATA PROCESSING")
    logger.info("=" * 60)

    # Save raw data
    raw_path = settings.raw_dir / "properties_raw.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"Saved raw data to {raw_path}")

    # Clean data
    cleaner = DataCleaner(outlier_method="iqr", outlier_threshold=2.5)
    df_cleaned = cleaner.clean(df)

    report = cleaner.get_cleaning_report()
    logger.info(f"Cleaning: {report['original_rows']} -> {report['final_rows']} rows")

    # Save processed data
    processed_path = settings.processed_dir / "properties_cleaned.csv"
    df_cleaned.to_csv(processed_path, index=False)
    logger.info(f"Saved processed data to {processed_path}")

    return df_cleaned


def run_clustering(
    df: pd.DataFrame,
    n_clusters: Optional[int] = None,
    extended_features: bool = False,
    compare_dbscan: bool = False,
    debug: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], Any]:
    """
    Run clustering analysis.

    Args:
        df: Cleaned DataFrame
        n_clusters: Number of clusters (auto if None)
        extended_features: Use extended feature set
        compare_dbscan: Compare with DBSCAN
        debug: Enable debug logging
        logger: Logger instance

    Returns:
        Tuple of (clustered_df, stats_df, metrics, elbow_data)
    """
    if logger is None:
        logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("PHASE 4: CLUSTERING")
    logger.info("=" * 60)

    # Select features
    if extended_features:
        feature_cols = settings.clustering.extended_features
        logger.info(f"Using extended features: {feature_cols}")
    else:
        feature_cols = settings.clustering.feature_columns
        logger.info(f"Using basic features: {feature_cols}")

    # Prepare features
    transformer = FeatureTransformer(
        scaling_method="standard", feature_columns=feature_cols
    )
    features = transformer.fit_transform(df)

    if debug:
        logger.debug(f"Feature matrix shape: {features.shape}")

    # Optimize k if not specified
    elbow_data = None
    if n_clusters is None:
        logger.info("Running cluster optimization...")
        optimizer = ClusterOptimizer(k_range=settings.clustering.k_range)
        optimizer.fit(features)

        n_clusters = optimizer.get_optimal_k()
        elbow_data = optimizer.get_elbow_data()

        logger.info(f"Optimal k: {n_clusters}")
        logger.info(optimizer.get_summary())

    # Run K-means
    logger.info(f"Running K-means with k={n_clusters}...")
    clusterer = KMeansClusterer(n_clusters=n_clusters)
    clusterer.fit(features)

    df = clusterer.assign_labels_to_df(df)
    metrics = clusterer.get_metrics()

    logger.info(f"K-means complete. Silhouette: {metrics['silhouette']:.4f}")

    # Per-cluster statistics
    silhouette_per_cluster = clusterer.get_silhouette_per_cluster(features)
    logger.info("Per-cluster silhouette:")
    for cluster_id, score in silhouette_per_cluster.items():
        count = len(df[df["cluster"] == cluster_id])
        logger.info(f"  Cluster {cluster_id}: {score:.4f} (n={count})")

    # Calculate statistics
    stats_calc = ClusterStatistics(df)
    stats_df = stats_calc.compute_summary()

    # DBSCAN comparison
    if compare_dbscan:
        logger.info("\n--- DBSCAN Comparison ---")
        from sklearn.neighbors import NearestNeighbors

        # Find optimal eps
        k = 5
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(features)
        distances, _ = nn.kneighbors(features)
        k_distances = np.sort(distances[:, k - 1])
        eps = np.percentile(k_distances, 90)

        logger.info(f"DBSCAN eps: {eps:.4f}")

        dbscan = DBSCANClusterer(eps=eps, min_samples=5)
        dbscan.fit(features)
        dbscan_metrics = dbscan.get_metrics()

        logger.info(f"DBSCAN found {dbscan_metrics['n_clusters']} clusters")
        logger.info(
            f"DBSCAN noise points: {dbscan_metrics['n_noise']} ({dbscan_metrics['noise_ratio']:.1%})"
        )
        logger.info(f"DBSCAN silhouette: {dbscan_metrics['silhouette']:.4f}")

        # Add DBSCAN labels for comparison
        df["dbscan_cluster"] = dbscan.labels_

    return df, stats_df, metrics, elbow_data


def create_visualizations(logger: Optional[logging.Logger] = None) -> Path:
    """
    Create maps and plots.

    Args:
        df: Clustered DataFrame
        gdf_districts: District boundaries
        elbow_data: Data for elbow plot
        debug: Enable debug logging
        logger: Logger instance

    Returns:
        Path to generated map
    """
    if logger is None:
        logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("PHASE 5: VISUALIZATION")
    logger.info("=" * 60)

    # Create interactive map with REAL district boundaries
    logger.info("Creating interactive map...")

    map_path = settings.maps_dir / "cluster_map.html"

    logger.info(f"Map saved to: {map_path}")

    return map_path


def print_summary(
    df: pd.DataFrame,
    metrics: Dict[str, float],
    map_path: Path,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Print final summary."""
    if logger is None:
        logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)

    logger.info(f"\nTotal properties: {len(df)}")
    logger.info(f"Districts covered: {df['district'].nunique()}")
    logger.info(f"Clusters found: {df['cluster'].nunique()}")
    logger.info(f"Silhouette score: {metrics['silhouette']:.4f}")

    logger.info("\nCluster summary:")

    for cluster_id in sorted(df["cluster"].unique()):
        avg_price = df[df["cluster"] == cluster_id]["price_per_sqm"].mean()
        count = len(df[df["cluster"] == cluster_id])
        logger.info(
            f"  Cluster {cluster_id}: {count} properties, avg {avg_price:,.0f} PLN/m²"
        )

    logger.info("\nOutputs:")
    logger.info(f"  Map: {map_path}")
    logger.info(f"  Data: {settings.processed_dir / 'properties_cleaned.csv'}")

    logger.info("\nTo view the map, open the HTML file in a web browser.")


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    # Setup logging
    if args.debug:
        setup_debug()
    elif args.verbose:
        setup_verbose()
    else:
        setup_logging()

    logger = get_logger("warsaw_clustering")

    logger.info("=" * 60)
    logger.info("WARSAW REAL ESTATE CLUSTERING ANALYSIS")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    try:

        gdf_districts = fetch_district_boundaries(
            force_refresh=args.force_refresh, logger=logger
        )

        # Phase 2: Generate data
        df = generate_properties(
            gdf_districts, n_samples=args.n_samples, debug=args.debug, logger=logger
        )

        # Phase 3: Process data
        df = process_data(df, logger=logger)

        # Phase 4: Clustering
        df, stats_df, metrics, elbow_data = run_clustering(
            df,
            n_clusters=args.k,
            extended_features=args.extended_features,
            compare_dbscan=args.compare_dbscan,
            debug=args.debug,
            logger=logger,
        )

        # Phase 5: Visualization
        map_path = create_visualizations(logger=logger)

        # Save clustered data
        clustered_path = settings.processed_dir / "properties_clustered.csv"
        df.to_csv(clustered_path, index=False)
        logger.info(f"Saved clustered data to {clustered_path}")

        # Print summary
        print_summary(df, metrics, map_path, logger)

        logger.info("\n✅ Analysis completed successfully!")
        return 0

    except Exception as e:
        logger.error("=" * 60)
        logger.error("ERROR")
        logger.error("=" * 60)
        logger.error(f"Exception: {type(e).__name__}")
        logger.error(f"Message: {str(e)}")
        logger.error("Traceback:")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
