"""
Static plot creation module with advanced visualizations.

Creates matplotlib/seaborn plots including:
- 6-panel analysis overview
- Violin + Box plots
- Radar chart for cluster profiles  
- Correlation heatmap
- 3D scatter plot
- Price histogram by cluster
- Elbow + Silhouette optimization (dual axis)
- Dendrogram with heatmap
- Silhouette profile with cluster breakdown
- District distribution

IMPORTANT: Clusters sorted by average price (C0 = highest, C<k> = lowest)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from config.logging_config import get_logger

logger = get_logger(__name__)

# Configure matplotlib
plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
sns.set_style("whitegrid")


def get_price_sorted_palette(
    df: pd.DataFrame, 
    label_col: str = "cluster"
) -> Tuple[Dict[int, str], List[int]]:
    """
    Generate color palette sorted by average price.
    
    Returns:
        Tuple of (color_dict, sorted_cluster_ids)
    """
    if "price_per_sqm" not in df.columns:
        unique = sorted(df[label_col].unique())
        base = ["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#17becf",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#1f77b4"]
        return {c: base[i % len(base)] for i, c in enumerate(unique)}, unique
    
    cluster_prices = df.groupby(label_col)["price_per_sqm"].mean().sort_values(ascending=False)
    sorted_clusters = list(cluster_prices.index)
    
    n = len(sorted_clusters)
    colors = {}
    
    for rank, cluster_id in enumerate(sorted_clusters):
        if n == 1:
            colors[cluster_id] = "#ffff00"
        else:
            ratio = rank / (n - 1)
            # Red → Orange → Yellow → Green gradient
            if ratio < 0.33:
                t = ratio / 0.33
                r, g, b = 220, int(50 + 130 * t), int(40 * t)
            elif ratio < 0.66:
                t = (ratio - 0.33) / 0.33
                r, g, b = int(220 - 60 * t), int(180 - 20 * t), int(40 + 20 * t)
            else:
                t = (ratio - 0.66) / 0.34
                r, g, b = int(160 - 120 * t), int(160 + 40 * t), int(60 + 20 * t)
            colors[cluster_id] = f"#{r:02x}{g:02x}{b:02x}"
    
    return colors, sorted_clusters


class PlotCreator:
    """Creates comprehensive static plots for cluster analysis."""
    
    def __init__(self, style: str = "whitegrid"):
        """Initialize plot creator."""
        sns.set_style(style)
        self._cluster_colors: Dict[int, str] = {}
        self._sorted_clusters: List[int] = []
    
    def _setup_colors(self, df: pd.DataFrame, label_col: str = "cluster") -> None:
        """Setup cluster colors based on price sorting."""
        self._cluster_colors, self._sorted_clusters = get_price_sorted_palette(df, label_col)
    
    # ========== MAIN 6-PANEL ANALYSIS ==========
    
    def create_analysis_figure(
        self,
        df: pd.DataFrame,
        elbow_data: Optional[Tuple[List[int], List[float]]] = None,
        silhouette_data: Optional[Tuple[List[int], List[float]]] = None,
        output_path: Optional[Path] = None,
        label_col: str = "cluster"
    ) -> plt.Figure:
        """
        Create 6-panel analysis figure.
        
        Panels:
        1. Spatial distribution (colored by price gradient)
        2. Price boxplot by cluster
        3. Distance vs price scatter
        4. Cluster sizes with prices
        5. Feature comparison heatmap
        6. Elbow + Silhouette optimization
        """
        self._setup_colors(df, label_col)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle("🏠 Cluster Analysis Overview", fontsize=16, fontweight="bold", y=1.02)
        
        self._plot_spatial_distribution(axes[0, 0], df, label_col)
        self._plot_price_boxplot(axes[0, 1], df, label_col)
        self._plot_distance_vs_price(axes[0, 2], df, label_col)
        self._plot_cluster_sizes(axes[1, 0], df, label_col)
        self._plot_feature_heatmap(axes[1, 1], df, label_col)
        self._plot_optimization(axes[1, 2], elbow_data, silhouette_data)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Analysis figure saved to {output_path}")
        
        return fig
    
    def _plot_spatial_distribution(self, ax: plt.Axes, df: pd.DataFrame, label_col: str) -> None:
        """Plot spatial distribution with price gradient coloring."""
        for cluster_id in reversed(self._sorted_clusters):
            cluster_data = df[df[label_col] == cluster_id]
            color = self._cluster_colors[cluster_id]
            avg_price = cluster_data["price_per_sqm"].mean() if "price_per_sqm" in df.columns else 0
            
            ax.scatter(
                cluster_data["lon"], cluster_data["lat"],
                c=color, label=f"C{cluster_id}: {avg_price:,.0f} PLN/m²",
                alpha=0.6, s=35, edgecolors="white", linewidth=0.3
            )
        
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Spatial Distribution\n(colored by price: green→red)", fontweight="bold")
        ax.legend(loc="upper right", fontsize=7, title="Clusters (by price ↓)")
    
    def _plot_price_boxplot(self, ax: plt.Axes, df: pd.DataFrame, label_col: str) -> None:
        """Plot price distribution by cluster (sorted by avg price)."""
        if "price_per_sqm" not in df.columns:
            ax.text(0.5, 0.5, "No price data", ha="center", va="center", transform=ax.transAxes)
            return
        
        order = self._sorted_clusters
        palette = [self._cluster_colors[c] for c in order]
        
        sns.boxplot(data=df, x=label_col, y="price_per_sqm", order=order, palette=palette, ax=ax)
        
        ax.set_xlabel("Cluster (sorted by price)")
        ax.set_ylabel("Price (PLN/m²)")
        ax.set_title("Price Distribution by Cluster", fontweight="bold")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    
    def _plot_distance_vs_price(self, ax: plt.Axes, df: pd.DataFrame, label_col: str) -> None:
        """Plot distance from center vs price."""
        if "distance_from_center_km" not in df.columns or "price_per_sqm" not in df.columns:
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center", transform=ax.transAxes)
            return
        
        for cluster_id in self._sorted_clusters:
            cluster_data = df[df[label_col] == cluster_id]
            ax.scatter(
                cluster_data["distance_from_center_km"], cluster_data["price_per_sqm"],
                c=self._cluster_colors[cluster_id], label=f"C{cluster_id}",
                alpha=0.5, s=25
            )
        
        ax.set_xlabel("Distance from Center (km)")
        ax.set_ylabel("Price (PLN/m²)")
        ax.set_title("Distance vs Price", fontweight="bold")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    
    def _plot_cluster_sizes(self, ax: plt.Axes, df: pd.DataFrame, label_col: str) -> None:
        """Plot cluster sizes sorted by average price."""
        counts = df.groupby(label_col).size()
        prices = df.groupby(label_col)["price_per_sqm"].mean() if "price_per_sqm" in df.columns else None
        
        order = self._sorted_clusters
        heights = [counts[c] for c in order]
        colors = [self._cluster_colors[c] for c in order]
        
        bars = ax.bar(range(len(order)), heights, color=colors)
        
        for bar, count in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                   str(count), ha="center", va="bottom", fontsize=10, fontweight="bold")
        
        labels = [f"C{c}\n{prices[c]/1000:.1f}k" for c in order] if prices is not None else [f"C{c}" for c in order]
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel("Cluster (sorted by avg price ↓)")
        ax.set_ylabel("Number of Properties")
        ax.set_title("Cluster Sizes with Prices", fontweight="bold")
    
    def _plot_feature_heatmap(self, ax: plt.Axes, df: pd.DataFrame, label_col: str) -> None:
        """Plot normalized feature heatmap."""
        feature_cols = ["price_per_sqm", "area_sqm", "distance_from_center_km"]
        available = [c for c in feature_cols if c in df.columns]
        
        if not available:
            ax.text(0.5, 0.5, "No features", ha="center", va="center", transform=ax.transAxes)
            return
        
        cluster_means = df.groupby(label_col)[available].mean()
        normalized = cluster_means.copy()
        for col in available:
            min_val, max_val = normalized[col].min(), normalized[col].max()
            if max_val > min_val:
                normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
        
        normalized = normalized.loc[self._sorted_clusters]
        col_names = {"price_per_sqm": "Price Sqm", "area_sqm": "Area", "distance_from_center_km": "Distance"}
        normalized.columns = [col_names.get(c, c) for c in normalized.columns]
        
        sns.heatmap(normalized.T, ax=ax, annot=True, fmt=".2f", cmap="YlOrRd",
                   xticklabels=[f"C{c}" for c in self._sorted_clusters],
                   cbar_kws={"label": "Normalized Value"})
        
        ax.set_xlabel("Cluster (sorted by price ↓)")
        ax.set_title("Feature Comparison (Normalized)", fontweight="bold")
    
    def _plot_optimization(self, ax: plt.Axes, elbow_data: Optional[tuple], silhouette_data: Optional[tuple]) -> None:
        """Plot elbow + silhouette on dual axis."""
        if elbow_data is None and silhouette_data is None:
            ax.text(0.5, 0.5, "Run with ClusterOptimizer\nfor optimization plots",
                   ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_title("Cluster Optimization", fontweight="bold")
            return
        
        ax2 = ax.twinx()
        
        if elbow_data:
            k_vals, inertias = elbow_data
            ax.plot(k_vals, inertias, "b-o", linewidth=2, markersize=6, label="Inertia")
            ax.set_ylabel("Inertia", color="blue")
            ax.tick_params(axis="y", labelcolor="blue")
        
        if silhouette_data:
            k_vals, silhouettes = silhouette_data
            ax2.plot(k_vals, silhouettes, "r-s", linewidth=2, markersize=6, label="Silhouette")
            ax2.set_ylabel("Silhouette Score", color="red")
            ax2.tick_params(axis="y", labelcolor="red")
            
            best_k_idx = np.argmax(silhouettes)
            best_k = k_vals[best_k_idx]
            ax2.axvline(x=best_k, color="green", linestyle="--", alpha=0.7, label=f"Optimal k={best_k}")
        
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_title("Cluster Optimization\n(Elbow + Silhouette)", fontweight="bold")
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")
        ax.grid(True, alpha=0.3)
    
    # ========== VIOLIN + BOX PLOT ==========
    
    def create_violin_boxplot(
        self,
        df: pd.DataFrame,
        label_col: str = "cluster",
        output_path: Optional[Path] = None
    ) -> plt.Figure:
        """Create combined violin + box plot for price distribution."""
        self._setup_colors(df, label_col)
        
        if "price_per_sqm" not in df.columns:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No price data", ha="center", va="center")
            return fig
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        order = self._sorted_clusters
        palette = [self._cluster_colors[c] for c in order]
        
        # Violin plot (background)
        sns.violinplot(data=df, x=label_col, y="price_per_sqm", order=order,
                      palette=palette, inner=None, alpha=0.3, ax=ax)
        
        # Box plot (foreground)
        sns.boxplot(data=df, x=label_col, y="price_per_sqm", order=order,
                   palette=palette, width=0.3, ax=ax)
        
        # Add mean markers
        means = df.groupby(label_col)["price_per_sqm"].mean()
        for i, c in enumerate(order):
            ax.scatter([i], [means[c]], color="white", s=100, zorder=5, edgecolors="black")
            ax.text(i, means[c] + 500, f"{means[c]:,.0f}", ha="center", fontsize=9)
        
        ax.set_xlabel("Cluster (sorted by price)")
        ax.set_ylabel("Price (PLN/m²)")
        ax.set_title("Price Distribution: Violin + Box Plot\n(white dots = mean)", fontweight="bold")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Violin+box plot saved to {output_path}")
        
        return fig
    
    # ========== RADAR CHART ==========
    
    def create_radar_chart(
        self,
        df: pd.DataFrame,
        label_col: str = "cluster",
        output_path: Optional[Path] = None
    ) -> plt.Figure:
        """Create radar chart comparing cluster profiles."""
        self._setup_colors(df, label_col)
        
        feature_cols = ["price_per_sqm", "area_sqm", "distance_from_center_km", "rooms", "floor"]
        available = [c for c in feature_cols if c in df.columns]
        
        if len(available) < 3:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Not enough features (need 3+)", ha="center", va="center")
            return fig
        
        # Compute and normalize cluster means
        cluster_means = df.groupby(label_col)[available].mean()
        normalized = cluster_means.copy()
        for col in available:
            min_val, max_val = normalized[col].min(), normalized[col].max()
            if max_val > min_val:
                normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
        
        # Setup radar
        categories = [c.replace("_", " ").title()[:10] for c in available]
        n_cats = len(categories)
        angles = [n / n_cats * 2 * np.pi for n in range(n_cats)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for cluster_id in self._sorted_clusters:
            values = normalized.loc[cluster_id].values.tolist()
            values += values[:1]
            
            color = self._cluster_colors[cluster_id]
            avg_price = df[df[label_col] == cluster_id]["price_per_sqm"].mean()
            ax.plot(angles, values, 'o-', linewidth=2, label=f"C{cluster_id}: {avg_price:,.0f}", color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), title="Clusters (PLN/m²)")
        ax.set_title("Cluster Profiles Comparison (Radar)", fontweight="bold", size=14, y=1.1)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Radar chart saved to {output_path}")
        
        return fig
    
    # ========== CORRELATION HEATMAP ==========
    
    def create_correlation_heatmap(
        self,
        df: pd.DataFrame,
        output_path: Optional[Path] = None
    ) -> plt.Figure:
        """Create correlation heatmap for numeric features."""
        numeric_cols = ["price_per_sqm", "area_sqm", "distance_from_center_km", 
                       "rooms", "floor", "year_built", "price_total"]
        available = [c for c in numeric_cols if c in df.columns]
        
        if len(available) < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Not enough numeric features", ha="center", va="center")
            return fig
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        corr = df[available].corr()
        
        # Rename for display
        rename_map = {
            "price_per_sqm": "Price/m²", "area_sqm": "Area", 
            "distance_from_center_km": "Distance", "rooms": "Rooms",
            "floor": "Floor", "year_built": "Year Built", "price_total": "Total Price"
        }
        corr = corr.rename(index=rename_map, columns=rename_map)
        
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                   center=0, vmin=-1, vmax=1, ax=ax, square=True,
                   linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        ax.set_title("Feature Correlation Matrix", fontweight="bold", size=14)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Correlation heatmap saved to {output_path}")
        
        return fig
    
    # ========== 3D SCATTER ==========
    
    def create_3d_scatter(
        self,
        df: pd.DataFrame,
        label_col: str = "cluster",
        output_path: Optional[Path] = None
    ) -> plt.Figure:
        """Create 3D scatter plot of price, area, and distance."""
        self._setup_colors(df, label_col)
        
        required = ["price_per_sqm", "area_sqm", "distance_from_center_km"]
        if not all(c in df.columns for c in required):
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Required features not available", ha="center", va="center")
            return fig
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for cluster_id in self._sorted_clusters:
            cluster_data = df[df[label_col] == cluster_id]
            color = self._cluster_colors[cluster_id]
            avg_price = cluster_data["price_per_sqm"].mean()
            
            ax.scatter(
                cluster_data["area_sqm"],
                cluster_data["distance_from_center_km"],
                cluster_data["price_per_sqm"],
                c=color, label=f"C{cluster_id}: {avg_price:,.0f}",
                alpha=0.6, s=30
            )
        
        ax.set_xlabel("Area (m²)")
        ax.set_ylabel("Distance from Center (km)")
        ax.set_zlabel("Price (PLN/m²)")
        ax.set_title("3D Cluster Visualization", fontweight="bold", size=14)
        ax.legend(loc='upper left', title="Clusters (PLN/m²)")
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"3D scatter saved to {output_path}")
        
        return fig
    
    # ========== PRICE HISTOGRAM ==========
    
    def create_price_histogram(
        self,
        df: pd.DataFrame,
        label_col: str = "cluster",
        output_path: Optional[Path] = None
    ) -> plt.Figure:
        """Create overlapping price histograms by cluster."""
        self._setup_colors(df, label_col)
        
        if "price_per_sqm" not in df.columns:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No price data", ha="center", va="center")
            return fig
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for cluster_id in self._sorted_clusters:
            cluster_data = df[df[label_col] == cluster_id]
            color = self._cluster_colors[cluster_id]
            avg_price = cluster_data["price_per_sqm"].mean()
            
            ax.hist(cluster_data["price_per_sqm"], bins=25, alpha=0.5,
                   color=color, label=f"C{cluster_id}: {avg_price:,.0f} PLN/m²",
                   edgecolor="white", linewidth=0.5)
        
        ax.set_xlabel("Price (PLN/m²)")
        ax.set_ylabel("Frequency")
        ax.set_title("Price Distribution by Cluster (Histogram)", fontweight="bold")
        ax.legend(title="Clusters", loc="upper right")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Price histogram saved to {output_path}")
        
        return fig
    
    # ========== DENDROGRAM HEATMAP ==========
    
    def create_dendrogram_heatmap(
        self,
        df: pd.DataFrame,
        label_col: str = "cluster",
        output_path: Optional[Path] = None,
        feature_cols: Optional[List[str]] = None
    ) -> plt.Figure:
        """Create dendrogram with heatmap showing hierarchical structure."""
        self._setup_colors(df, label_col)
        
        if feature_cols is None:
            feature_cols = ["floor", "year_built", "price_per_sqm", "distance_from_center_km", "rooms", "area_sqm"]
        
        available = [c for c in feature_cols if c in df.columns]
        
        if len(available) < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Not enough features", ha="center", va="center")
            return fig
        
        # Sort by cluster
        df_sorted = df.copy()
        df_sorted["_sort_key"] = df_sorted[label_col].map({c: i for i, c in enumerate(self._sorted_clusters)})
        df_sorted = df_sorted.sort_values("_sort_key")
        
        # Standardize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df_sorted[available])
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 8], width_ratios=[1, 8, 0.3], hspace=0.02, wspace=0.02)
        
        # Top dendrogram
        ax_top = fig.add_subplot(gs[0, 1])
        sample_size = min(100, len(features_scaled))
        linkage_samples = linkage(features_scaled[:sample_size], method="ward")
        dendrogram(linkage_samples, ax=ax_top, no_labels=True, color_threshold=0, above_threshold_color="gray")
        ax_top.set_title("Hierarchical Clustering Dendrogram with Heatmap", fontweight="bold", fontsize=14)
        ax_top.axis("off")
        
        # Left dendrogram
        ax_left = fig.add_subplot(gs[1, 0])
        linkage_features = linkage(features_scaled.T, method="ward")
        dendrogram(linkage_features, ax=ax_left, orientation="left",
                  labels=[c.replace("_", " ").title()[:12] for c in available],
                  color_threshold=0, above_threshold_color="gray")
        ax_left.set_xlabel("Distance")
        
        # Heatmap
        ax_heat = fig.add_subplot(gs[1, 1])
        feature_order = dendrogram(linkage_features, no_plot=True)["leaves"]
        
        sns.heatmap(features_scaled[:, feature_order].T, ax=ax_heat, cmap="RdBu_r", center=0,
                   xticklabels=False, yticklabels=[available[i].replace("_", " ").title()[:12] for i in feature_order],
                   cbar_ax=fig.add_subplot(gs[1, 2]))
        ax_heat.set_xlabel("Samples (sorted by cluster)")
        
        # Cluster color bar
        cluster_colors = [self._cluster_colors[c] for c in df_sorted[label_col]]
        ax_cb = ax_heat.inset_axes([0, -0.05, 1, 0.03])
        for i, color in enumerate(cluster_colors[:len(features_scaled)]):
            ax_cb.axvline(x=i, color=color, linewidth=0.5)
        ax_cb.set_xlim(0, len(features_scaled))
        ax_cb.axis("off")
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Dendrogram heatmap saved to {output_path}")
        
        return fig
    
    # ========== SILHOUETTE PROFILE ==========
    
    def create_silhouette_profile(
        self,
        df: pd.DataFrame,
        label_col: str = "cluster",
        output_path: Optional[Path] = None,
        feature_cols: Optional[List[str]] = None
    ) -> plt.Figure:
        """Create silhouette profile with cluster quality assessment."""
        self._setup_colors(df, label_col)
        
        if feature_cols is None:
            feature_cols = ["price_per_sqm", "area_sqm", "distance_from_center_km"]
        
        available = [c for c in feature_cols if c in df.columns]
        
        if len(available) < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Not enough features", ha="center", va="center")
            return fig
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df[available])
        labels = df[label_col].values
        
        try:
            sample_silhouettes = silhouette_samples(features_scaled, labels)
            avg_silhouette = silhouette_score(features_scaled, labels)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
            return fig
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 8), gridspec_kw={"width_ratios": [2, 1, 1]})
        
        # Left: Silhouette profile
        ax_profile = axes[0]
        y_lower = 10
        
        for cluster_id in self._sorted_clusters:
            cluster_sil = sample_silhouettes[labels == cluster_id]
            cluster_sil.sort()
            
            size = len(cluster_sil)
            y_upper = y_lower + size
            
            color = self._cluster_colors[cluster_id]
            ax_profile.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                                    facecolor=color, edgecolor=color, alpha=0.7)
            
            avg_price = df[df[label_col] == cluster_id]["price_per_sqm"].mean()
            ax_profile.text(-0.08, y_lower + size / 2, f"C{cluster_id}\n{avg_price/1000:.1f}k",
                          fontsize=9, va="center", ha="right")
            
            y_lower = y_upper + 10
        
        ax_profile.axvline(x=avg_silhouette, color="red", linestyle="--", linewidth=2)
        ax_profile.text(avg_silhouette + 0.02, y_lower - 5, f"Avg: {avg_silhouette:.3f}",
                       color="red", fontsize=10, fontweight="bold")
        
        quality = "Good" if avg_silhouette > 0.5 else "Reasonable" if avg_silhouette > 0.25 else "Weak"
        quality_color = "green" if avg_silhouette > 0.5 else "orange" if avg_silhouette > 0.25 else "red"
        ax_profile.text(0.02, 0.98, f"{quality} structure", transform=ax_profile.transAxes,
                       fontsize=12, fontweight="bold", color=quality_color, va="top")
        
        ax_profile.set_xlabel("Silhouette Coefficient")
        ax_profile.set_ylabel("Cluster")
        ax_profile.set_title("Silhouette Profile Chart\n(Cluster Quality Assessment)", fontweight="bold")
        ax_profile.set_xlim([-0.2, 1])
        ax_profile.set_ylim([0, y_lower])
        ax_profile.set_yticks([])
        ax_profile.grid(True, axis="x", alpha=0.3)
        
        # Middle: Stats table
        ax_stats = axes[1]
        ax_stats.axis("off")
        
        cell_text = []
        for cluster_id in self._sorted_clusters:
            mask = labels == cluster_id
            avg_sil = np.mean(sample_silhouettes[mask])
            count = mask.sum()
            avg_price = df[df[label_col] == cluster_id]["price_per_sqm"].mean()
            cell_text.append([f"C{cluster_id}", f"{avg_sil:.3f}", str(count), f"{avg_price:,.0f}"])
        
        table = ax_stats.table(cellText=cell_text, colLabels=["Cluster", "Silhouette", "n", "Avg Price"],
                              loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax_stats.set_title("Cluster Statistics", fontweight="bold", y=0.85)
        
        # Right: Bar chart
        ax_sizes = axes[2]
        counts = [sum(labels == c) for c in self._sorted_clusters]
        colors = [self._cluster_colors[c] for c in self._sorted_clusters]
        prices = [df[df[label_col] == c]["price_per_sqm"].mean() for c in self._sorted_clusters]
        
        bars = ax_sizes.barh(range(len(self._sorted_clusters)), counts, color=colors)
        ax_sizes.set_yticks(range(len(self._sorted_clusters)))
        ax_sizes.set_yticklabels([f"C{c}: {p/1000:.1f}k" for c, p in zip(self._sorted_clusters, prices)])
        ax_sizes.set_xlabel("Number of Properties")
        ax_sizes.set_title("Cluster Sizes\n(sorted by avg price, desc)", fontweight="bold")
        ax_sizes.invert_yaxis()
        
        for bar, count in zip(bars, counts):
            ax_sizes.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, str(count), va="center", fontsize=9)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Silhouette profile saved to {output_path}")
        
        return fig
    
    # ========== DISTRICT DISTRIBUTION ==========
    
    def plot_district_distribution(
        self,
        df: pd.DataFrame,
        label_col: str = "cluster",
        output_path: Optional[Path] = None
    ) -> plt.Figure:
        """Plot district distribution across clusters."""
        self._setup_colors(df, label_col)
        
        if "district" not in df.columns:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No district data", ha="center", va="center")
            return fig
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        cross_tab = pd.crosstab(df["district"], df[label_col])
        cross_tab = cross_tab[self._sorted_clusters]
        cross_tab = cross_tab.loc[cross_tab.sum(axis=1).sort_values(ascending=True).index]
        
        colors = [self._cluster_colors[c] for c in self._sorted_clusters]
        cross_tab.plot(kind="barh", stacked=True, ax=ax, color=colors)
        
        prices = df.groupby(label_col)["price_per_sqm"].mean()
        handles, _ = ax.get_legend_handles_labels()
        labels = [f"C{c}: {prices[c]:,.0f} PLN/m²" for c in self._sorted_clusters]
        ax.legend(handles, labels, title="Clusters (by price ↓)", bbox_to_anchor=(1.02, 1), loc="upper left")
        
        ax.set_xlabel("Number of Properties")
        ax.set_ylabel("District")
        ax.set_title("Cluster Distribution by District\n(clusters colored by price)", fontweight="bold")
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"District distribution saved to {output_path}")
        
        return fig
    
    # ========== CREATE ALL PLOTS ==========
    
    def create_all_plots(
        self,
        df: pd.DataFrame,
        elbow_data: Optional[tuple] = None,
        silhouette_data: Optional[tuple] = None,
        output_dir: Optional[Path] = None,
        label_col: str = "cluster"
    ) -> Dict[str, plt.Figure]:
        """Create all analysis plots and save to directory."""
        self._setup_colors(df, label_col)
        
        figures = {}
        
        # 1. Main 6-panel
        figures["analysis_overview"] = self.create_analysis_figure(
            df, elbow_data, silhouette_data,
            output_path=output_dir / "analysis_overview.png" if output_dir else None,
            label_col=label_col
        )
        
        # 2. Dendrogram heatmap
        figures["dendrogram_heatmap"] = self.create_dendrogram_heatmap(
            df, label_col=label_col,
            output_path=output_dir / "dendrogram_heatmap.png" if output_dir else None
        )
        
        # 3. Silhouette profile
        figures["silhouette_profile"] = self.create_silhouette_profile(
            df, label_col=label_col,
            output_path=output_dir / "silhouette_profile.png" if output_dir else None
        )
        
        # 4. District distribution
        figures["district_distribution"] = self.plot_district_distribution(
            df, label_col=label_col,
            output_path=output_dir / "district_distribution.png" if output_dir else None
        )
        
        # 5. Violin + Box
        figures["violin_boxplot"] = self.create_violin_boxplot(
            df, label_col=label_col,
            output_path=output_dir / "violin_boxplot.png" if output_dir else None
        )
        
        # 6. Radar chart
        figures["radar_chart"] = self.create_radar_chart(
            df, label_col=label_col,
            output_path=output_dir / "radar_chart.png" if output_dir else None
        )
        
        # 7. Correlation heatmap
        figures["correlation_heatmap"] = self.create_correlation_heatmap(
            df, output_path=output_dir / "correlation_heatmap.png" if output_dir else None
        )
        
        # 8. Price histogram
        figures["price_histogram"] = self.create_price_histogram(
            df, label_col=label_col,
            output_path=output_dir / "price_histogram.png" if output_dir else None
        )
        
        # 9. 3D scatter
        figures["3d_scatter"] = self.create_3d_scatter(
            df, label_col=label_col,
            output_path=output_dir / "3d_scatter.png" if output_dir else None
        )
        
        logger.info(f"Created {len(figures)} plots")
        plt.close("all")
        
        return figures
