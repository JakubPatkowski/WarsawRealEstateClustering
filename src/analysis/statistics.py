"""
Cluster statistics module.

Computes summary statistics and profiles for clustered data.
"""

from typing import Any, Dict, Optional, cast

import pandas as pd

from config.logging_config import get_logger

logger = get_logger(__name__)


class ClusterStatistics:
    """
    Computes statistics for clustered data.

    Example:
        >>> stats = ClusterStatistics(df_clustered)
        >>> summary = stats.compute_summary()
        >>> profiles = stats.get_cluster_profiles()
    """

    # Columns to compute statistics for
    STAT_COLUMNS = [
        "price_per_sqm",
        "price_total",
        "area_sqm",
        "rooms",
        "year_built",
        "floor",
        "distance_from_center_km",
    ]

    def __init__(self, df: pd.DataFrame, label_col: str = "cluster"):
        """
        Initialize statistics calculator.

        Args:
            df: Clustered DataFrame
            label_col: Name of cluster label column
        """
        # Guard clause
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame")

        self.df = df.copy()  # Always copy
        self.label_col = label_col

        self._summary: Optional[pd.DataFrame] = None
        self._profiles: Optional[Dict[int, Dict[str, Any]]] = None

        logger.debug(f"Initialized ClusterStatistics with {len(df)} rows")

    def compute_summary(self) -> pd.DataFrame:
        """
        Compute summary statistics per cluster.

        Returns:
            DataFrame with statistics per cluster
        """
        # Find available columns
        available_cols = [c for c in self.STAT_COLUMNS if c in self.df.columns]

        if not available_cols:
            logger.warning("No statistical columns found")
            return pd.DataFrame()

        # Compute aggregations
        # FIX 1: Explicitly annotate agg_dict to satisfy .agg() signature
        agg_dict: Dict[str, Any] = {}
        for col in available_cols:
            agg_dict[col] = ["mean", "std", "min", "max", "count"]

        summary = self.df.groupby(self.label_col).agg(agg_dict)

        # FIX 2: Cast columns to MultiIndex so MyPy knows they are tuples (col, stat)
        # instead of strings.
        multi_cols = cast(pd.MultiIndex, summary.columns)

        # Flatten column names
        summary.columns = pd.Index([f"{col}_{stat}" for col, stat in multi_cols])

        # Add count column
        counts = self.df.groupby(self.label_col).size()
        summary["n_properties"] = counts

        # Calculate percentages
        total = len(self.df)
        summary["pct_of_total"] = (summary["n_properties"] / total * 100).round(1)

        # Sort by average price (if available)
        if "price_per_sqm_mean" in summary.columns:
            summary = summary.sort_values("price_per_sqm_mean", ascending=False)

        # Set proper index name
        summary.index = pd.Index(
            [f"Cluster {i}" for i in summary.index], name="cluster"
        )

        self._summary = summary
        logger.info(f"Computed summary statistics for {len(summary)} clusters")

        return summary

    def get_cluster_profiles(self) -> Dict[int, Dict[str, Any]]:
        """
        Create detailed profiles for each cluster.

        Returns:
            Dictionary mapping cluster_id to profile dict
        """
        profiles = {}

        for cluster_id in self.df[self.label_col].unique():
            cluster_data = self.df[self.df[self.label_col] == cluster_id]

            profile: Dict[str, Any] = {
                "n_properties": len(cluster_data),
                "pct_of_total": len(cluster_data) / len(self.df) * 100,
            }

            # Add statistics for each column
            for col in self.STAT_COLUMNS:
                if col in cluster_data.columns:
                    profile[f"{col}_mean"] = float(cluster_data[col].mean())
                    profile[f"{col}_std"] = float(cluster_data[col].std())
                    profile[f"{col}_min"] = float(cluster_data[col].min())
                    profile[f"{col}_max"] = float(cluster_data[col].max())

            # District distribution
            if "district" in cluster_data.columns:
                district_counts = cluster_data["district"].value_counts()
                profile["top_districts"] = district_counts.head(3).to_dict()
                profile["n_districts"] = cluster_data["district"].nunique()

            # Market segment distribution (if available)
            if "market_segment" in cluster_data.columns:
                segment_counts = cluster_data["market_segment"].value_counts()
                profile["market_segments"] = segment_counts.to_dict()

            profiles[int(cluster_id)] = profile

        self._profiles = profiles
        return profiles

    def get_cluster_comparison(self) -> pd.DataFrame:
        """
        Create side-by-side comparison of clusters.

        Returns:
            DataFrame with key metrics per cluster
        """
        rows = []

        for cluster_id in sorted(self.df[self.label_col].unique()):
            cluster_data = self.df[self.df[self.label_col] == cluster_id]

            row = {"cluster": cluster_id}
            row["count"] = len(cluster_data)

            if "price_per_sqm" in cluster_data.columns:
                row["avg_price"] = cluster_data["price_per_sqm"].mean()
                row["price_range"] = (
                    f"{cluster_data['price_per_sqm'].min():.0f}-"
                    f"{cluster_data['price_per_sqm'].max():.0f}"
                )

            if "area_sqm" in cluster_data.columns:
                row["avg_area"] = cluster_data["area_sqm"].mean()

            if "distance_from_center_km" in cluster_data.columns:
                row["avg_distance"] = cluster_data["distance_from_center_km"].mean()

            if "district" in cluster_data.columns:
                top_district = cluster_data["district"].value_counts().index[0]
                row["top_district"] = top_district

            rows.append(row)

        return pd.DataFrame(rows)

    def get_district_cluster_matrix(self) -> pd.DataFrame:
        """
        Create district x cluster crosstab.

        Returns:
            Crosstab DataFrame showing count per district/cluster
        """
        if "district" not in self.df.columns:
            return pd.DataFrame()

        return pd.crosstab(self.df["district"], self.df[self.label_col], margins=True)

    def describe_cluster(self, cluster_id: int) -> str:
        """
        Generate human-readable description of a cluster.

        Args:
            cluster_id: Cluster to describe

        Returns:
            Text description
        """
        cluster_data = self.df[self.df[self.label_col] == cluster_id]

        if len(cluster_data) == 0:
            return f"Cluster {cluster_id}: No data"

        lines = [
            f"=== Cluster {cluster_id} ===",
            f"Properties: {len(cluster_data)} ({len(cluster_data)/len(self.df)*100:.1f}%)",
        ]

        if "price_per_sqm" in cluster_data.columns:
            avg_price = cluster_data["price_per_sqm"].mean()
            lines.append(f"Average price: {avg_price:,.0f} PLN/m²")

        if "area_sqm" in cluster_data.columns:
            avg_area = cluster_data["area_sqm"].mean()
            lines.append(f"Average area: {avg_area:.1f} m²")

        if "distance_from_center_km" in cluster_data.columns:
            avg_dist = cluster_data["distance_from_center_km"].mean()
            lines.append(f"Average distance from center: {avg_dist:.1f} km")

        if "district" in cluster_data.columns:
            top_districts = cluster_data["district"].value_counts().head(3)
            lines.append("Top districts:")
            for dist, count in top_districts.items():
                pct = count / len(cluster_data) * 100
                lines.append(f"  - {dist}: {count} ({pct:.1f}%)")

        return "\n".join(lines)
