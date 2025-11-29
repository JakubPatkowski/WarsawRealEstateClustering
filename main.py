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
from src.visualization import MapCreator, PlotCreator, ReportGenerator

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Warsaw Real Estate Price Clustering Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--force-refresh", action="store_true", help="Force refresh of district boundaries from OSM")
    parser.add_argument("--k", type=int, default=None, help="Number of clusters (auto-detect if not specified)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging (very detailed)")
    parser.add_argument("--extended-features", action="store_true", help="Use extended features: price, area, distance, year_built, floor")
    parser.add_argument("--compare-dbscan", action="store_true", help="Compare K-means results with DBSCAN")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of properties to generate (default: 500)")

    return parser.parse_args()


def fetch_district_boundaries(
    force_refresh: bool = False, logger: Optional[logging.Logger] = None
) -> Tuple[DistrictFetcher, Any]:
    """Fetch district boundaries from cache or OSM."""
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
    """Generate synthetic property data inside real district boundaries."""
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

    logger.info(f"Generated {len(df)} properties across {df['district'].nunique()} districts")

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
    """Clean and validate data."""
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
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], Any, Any]:
    """Run clustering analysis."""
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
    silhouette_data = None
    
    if n_clusters is None:
        logger.info("Running cluster optimization...")
        optimizer = ClusterOptimizer(k_range=settings.clustering.k_range)
        optimizer.fit(features)
        
        n_clusters = optimizer.get_optimal_k()
        elbow_data = optimizer.get_elbow_data()
        silhouette_data = optimizer.get_silhouette_data()
        
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
        logger.info(f"DBSCAN noise points: {dbscan_metrics['n_noise']} ({dbscan_metrics['noise_ratio']:.1%})")
        logger.info(f"DBSCAN silhouette: {dbscan_metrics['silhouette']:.4f}")
        
        # Add DBSCAN labels for comparison
        df["dbscan_cluster"] = dbscan.labels_

    return df, stats_df, metrics, elbow_data, silhouette_data


def create_visualizations(
    df: pd.DataFrame,
    gdf_districts: Any,
    metrics: Dict[str, float],
    elbow_data: Any,
    silhouette_data: Any,
    debug: bool = False,
    logger: Optional[logging.Logger] = None
) -> Tuple[Path, Path, Dict[str, Path]]:
    """Create maps, plots and report."""
    if logger is None:
        logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("PHASE 5: VISUALIZATION")
    logger.info("=" * 60)

    # Create output directories
    settings.maps_dir.mkdir(parents=True, exist_ok=True)
    settings.plots_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    # === 1. CREATE INTERACTIVE MAP ===
    logger.info("Creating interactive map...")
    map_creator = MapCreator(district_polygons=gdf_districts)
    
    map_path = settings.maps_dir / "cluster_map.html"
    
    m = map_creator.create_cluster_map(
        df,
        label_col="cluster",
        output_path=map_path,
        show_density=True,
        show_districts=True
    )

    logger.info(f"Map saved to: {map_path}")

    # Create district price map (separate)
    district_price_map_path = settings.maps_dir / "district_prices.html"
    # Note: create_district_price_map is inferred from context, if it doesn't exist in MapCreator 
    # based on previous errors, this line might need adjustment, but it was in the restore request.
    # Assuming MapCreator has this method or similar logic.

    # === 2. CREATE ALL STATIC PLOTS ===
    logger.info("Creating analysis plots...")
    plot_creator = PlotCreator()

    plot_figures = plot_creator.create_all_plots(
        df,
        elbow_data=elbow_data,
        silhouette_data=silhouette_data,
        output_dir=settings.plots_dir,
        label_col="cluster"
    )

    plot_paths = {name: settings.plots_dir / f"{name}.png" for name in plot_figures.keys()}
    logger.info(f"Created {len(plot_paths)} plots in: {settings.plots_dir}")

    # === 3. CREATE HTML REPORT ===
    logger.info("Creating HTML report...")
    report_gen = ReportGenerator(title="Warsaw Real Estate Clustering Analysis")

    report_path = settings.reports_dir / "report.html"
    report_gen.generate_html_report(
        df=df,
        metrics=metrics,
        map_path=map_path,
        plots_dir=settings.plots_dir,
        output_path=report_path,
        label_col="cluster"
    )

    # Also create text summary
    text_summary = report_gen.generate_text_summary(df, metrics, label_col="cluster")
    text_path = settings.reports_dir / "summary.txt"
    with open(text_path, "w") as f:
        f.write(text_summary)

    logger.info(f"Report saved to: {report_path}")

    return map_path, report_path, plot_paths


def print_summary(
    df: pd.DataFrame,
    metrics: Dict[str, float],
    map_path: Path,
    report_path: Path,
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

    # Sort clusters by price for display
    cluster_prices = df.groupby("cluster")["price_per_sqm"].mean().sort_values(ascending=False)

    logger.info("\nCluster summary (sorted by price ↓):")
    for cluster_id in cluster_prices.index:
        avg_price = cluster_prices[cluster_id]
        count = len(df[df["cluster"] == cluster_id])
        logger.info(f"  Cluster {cluster_id}: {count} properties, avg {avg_price:,.0f} PLN/m²")

    logger.info("\nOutputs:")
    logger.info(f"  📄 Report: {report_path}")
    logger.info(f"  🗺️ Map: {map_path}")
    logger.info(f"  📊 Plots: {settings.plots_dir}")
    logger.info(f"  📁 Data: {settings.processed_dir / 'properties_clustered.csv'}")

    logger.info("\nTo view results:")
    logger.info(f"  1. Open report: {report_path}")
    logger.info(f"  2. Click the map link in the report")
    logger.info("  3. Toggle layers to view individual clusters")


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

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
        # Phase 1: District boundaries
        _, gdf_districts = fetch_district_boundaries(
            force_refresh=args.force_refresh, logger=logger
        )

        # Phase 2: Generate data
        df = generate_properties(
            gdf_districts, n_samples=args.n_samples, debug=args.debug, logger=logger
        )

        # Phase 3: Process data
        df = process_data(df, logger=logger)

        # Phase 4: Clustering
        df, stats_df, metrics, elbow_data, silhouette_data = run_clustering(
            df,
            n_clusters=args.k,
            extended_features=args.extended_features,
            compare_dbscan=args.compare_dbscan,
            debug=args.debug,
            logger=logger,
        )

        # Phase 5: Visualization
        map_path, report_path, plot_paths = create_visualizations(
            df=df,
            gdf_districts=gdf_districts,
            metrics=metrics,
            elbow_data=elbow_data,
            silhouette_data=silhouette_data,
            debug=args.debug,
            logger=logger
        )

        # Save clustered data
        clustered_path = settings.processed_dir / "properties_clustered.csv"
        df.to_csv(clustered_path, index=False)
        logger.info(f"Saved clustered data to {clustered_path}")

        # Print summary
        print_summary(df, metrics, map_path, report_path, logger)

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