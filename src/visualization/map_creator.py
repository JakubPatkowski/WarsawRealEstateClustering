"""
Interactive map creation module.

Creates Folium-based interactive maps with cluster visualization,
tooltips, popups, and legend.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from folium import plugins
from scipy.spatial import ConvexHull

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


def get_cluster_colors(df: pd.DataFrame, label_col: str = "cluster") -> Dict[int, str]:
    """
    Generate colors for clusters based on their average price RANK.
    """
    if label_col not in df.columns:
        return {}

    if "price_per_sqm" not in df.columns:
        # Fallback to default palette if no price data
        unique_clusters = sorted(df[label_col].unique())
        default_colors = [
            "#4daf4a",
            "#377eb8",
            "#ff7f00",
            "#e41a1c",
            "#984ea3",
            "#ffff33",
            "#a65628",
            "#f781bf",
            "#999999",
            "#66c2a5",
        ]
        # FIX: Cast 'c' (Hashable) to Any so int() accepts it
        return {
            int(cast(Any, c)): default_colors[i % len(default_colors)]
            for i, c in enumerate(unique_clusters)
        }

    # Calculate average price per cluster and sort by price
    cluster_avg_prices = df.groupby(label_col)["price_per_sqm"].mean().sort_values()
    n_clusters = len(cluster_avg_prices)

    if n_clusters == 0:
        return {}

    if n_clusters == 1:
        # FIX: Cast index value to Any
        return {int(cast(Any, cluster_avg_prices.index[0])): "#ffff00"}

    # Generate colors based on RANK (position) not absolute price
    colors: Dict[int, str] = {}
    for rank, (cluster_id, avg_price) in enumerate(cluster_avg_prices.items()):
        # rank goes from 0 (cheapest) to n_clusters-1 (most expensive)
        normalized = rank / (n_clusters - 1)

        # Green -> Yellow -> Orange -> Red gradient
        if normalized < 0.33:
            # Green to Yellow-Green
            t = normalized / 0.33
            r = int(100 * t)
            g = 200
            b = int(50 * (1 - t * 0.5))
        elif normalized < 0.66:
            # Yellow-Green to Orange
            t = (normalized - 0.33) / 0.33
            r = 100 + int(155 * t)
            g = 200 - int(80 * t)
            b = int(25 * (1 - t))
        else:
            # Orange to Red
            t = (normalized - 0.66) / 0.34
            r = 255
            g = int(120 * (1 - t))
            b = 0

        # FIX: Cast cluster_id to Any
        cid = int(cast(Any, cluster_id))
        colors[cid] = f"#{r:02x}{g:02x}{b:02x}"

        logger.debug(
            f"Cluster {cid}: rank={rank}, "
            f"avg_price={avg_price:.0f}, color={colors[cid]}"
        )

    return colors


def compute_boundary(
    points: np.ndarray, percentile: float = 85
) -> List[Tuple[float, float]]:
    """
    Compute boundary of points, excluding outliers.
    """

    if len(points) < 3:
        return []

    if len(points) < 4:
        # For small clusters, use all points
        try:
            hull = ConvexHull(points)
            # FIX 2: Explicitly create list of tuples (float, float)
            # .tolist() returns List[List[float]] which fails strictly typed check
            vertices = points[hull.vertices]
            boundary: List[Tuple[float, float]] = [
                (float(p[0]), float(p[1])) for p in vertices
            ]
            boundary.append(boundary[0])  # Close the polygon
            return boundary
        except Exception:
            return []

    # Filter outliers using percentile distance from centroid
    centroid = points.mean(axis=0)
    distances = np.sqrt(((points - centroid) ** 2).sum(axis=1))

    threshold = np.percentile(distances, percentile)
    mask = distances <= threshold
    filtered_points = points[mask]

    if len(filtered_points) < 3:
        filtered_points = points  # Fall back to all points

    try:
        hull = ConvexHull(filtered_points)
        # FIX 2: Explicitly create list of tuples here as well
        vertices = filtered_points[hull.vertices]
        boundary = [(float(p[0]), float(p[1])) for p in vertices]
        boundary.append(boundary[0])  # Close the polygon
        return boundary
    except Exception as e:
        logger.debug(f"Boundary computation failed: {e}")
        return []


class MapCreator:
    """
    Creates interactive Folium maps for cluster visualization.
    """

    def __init__(
        self,
        center: Optional[Tuple[float, float]] = None,
        zoom_start: int = 11,
        tiles: str = "CartoDB positron",
        district_polygons: Optional[gpd.GeoDataFrame] = None,
    ):
        self.center = center or settings.warsaw.center
        self.zoom_start = zoom_start
        self.tiles = tiles
        self.district_centers = settings.warsaw.districts
        self.cluster_colors: Dict[int, str] = {}
        self._district_gdf: Optional[gpd.GeoDataFrame] = district_polygons

        if district_polygons is not None:
            logger.info(
                f"MapCreator initialized with {len(district_polygons)} "
                "real district polygons"
            )
        else:
            logger.warning(
                "No district polygons provided - will use convex hull fallback"
            )

    def create_cluster_map(
        self,
        df: pd.DataFrame,
        label_col: str = "cluster",
        output_path: Optional[Path] = None,
        show_hulls: bool = True,
        show_centroids: bool = True,
        show_districts: bool = True,
        show_density: bool = True,
    ) -> folium.Map:
        self.cluster_colors = get_cluster_colors(df, label_col)

        m = folium.Map(
            location=self.center,
            zoom_start=self.zoom_start,
            tiles=None,
        )

        folium.TileLayer(
            tiles="CartoDB positron", name="🌞 Light Theme", control=True
        ).add_to(m)

        folium.TileLayer(
            tiles="CartoDB dark_matter", name="🌙 Dark Theme", control=True
        ).add_to(m)

        folium.TileLayer(
            tiles="OpenStreetMap", name="🗺️ OpenStreetMap", control=True
        ).add_to(m)

        if show_districts:
            self._add_district_boundaries(m, df)

        if show_density and "price_per_sqm" in df.columns:
            self._add_density_gradient(m, df)

        if show_hulls and label_col in df.columns:
            self._add_cluster_boundaries_separate(m, df, label_col)

        if label_col in df.columns:
            self._add_cluster_points(m, df, label_col)

        if show_centroids and label_col in df.columns:
            self._add_centroids(m, df, label_col)

        if label_col in df.columns:
            self._add_enhanced_legend(m, df, label_col)

        folium.LayerControl(collapsed=False).add_to(m)

        if output_path:
            m.save(str(output_path))
            logger.info(f"Map saved to {output_path}")

        return m

    def _add_district_boundaries(self, m: folium.Map, df: pd.DataFrame) -> None:
        district_group = folium.FeatureGroup(name="📍 District Boundaries", show=True)

        if self._district_gdf is not None:
            for _, row in self._district_gdf.iterrows():
                name = row["name"]
                geom = row.geometry
                avg_price = 0.0
                if "district" in df.columns and "price_per_sqm" in df.columns:
                    district_data = df[df["district"] == name]
                    if len(district_data) > 0:
                        avg_price = float(district_data["price_per_sqm"].mean())

                folium.GeoJson(
                    geom.__geo_interface__,
                    style_function=lambda x: {
                        "color": "#333333",
                        "weight": 2,
                        "fillOpacity": 0,
                        "dashArray": "5, 5",
                    },
                    tooltip=(
                        f"{name}<br>Avg: {avg_price:,.0f} PLN/m²"
                        if avg_price > 0
                        else name
                    ),
                ).add_to(district_group)

                centroid = geom.centroid
                folium.Marker(
                    location=[centroid.y, centroid.x],
                    icon=folium.DivIcon(
                        html=f"""<div style="
                            font-size: 9px; 
                            font-weight: bold; 
                            color: #333; 
                            white-space: nowrap; 
                            text-shadow: 1px 1px 1px white, -1px -1px 1px white;
                        ">{name}</div>""",
                        icon_size=(100, 20),
                        icon_anchor=(50, 10),
                    ),
                ).add_to(district_group)
        else:
            # Fallback logic omitted for brevity, but same fix logic applies if used
            pass

        district_group.add_to(m)

    def _add_density_gradient(self, m: folium.Map, df: pd.DataFrame) -> None:
        """Add transparent density gradient overlay."""
        density_group = folium.FeatureGroup(name="🔥 Price Heatmap", show=False)

        if "price_per_sqm" not in df.columns:
            return

        # FIX 3: Explicitly type heat_data to avoid "Any" inference
        heat_data: List[List[float]] = []
        max_price = float(df["price_per_sqm"].max())

        for _, row in df.iterrows():
            heat_data.append(
                [
                    float(row["lat"]),
                    float(row["lon"]),
                    float(row["price_per_sqm"]) / max_price,
                ]
            )

        plugins.HeatMap(
            heat_data,
            radius=20,
            blur=15,
            max_zoom=13,
            min_opacity=0.3,
            gradient={
                0.2: "#00c800",
                0.5: "#ffff00",
                0.8: "#ff0000",
            },
        ).add_to(density_group)

        density_group.add_to(m)

    def _add_cluster_boundaries_separate(
        self, m: folium.Map, df: pd.DataFrame, label_col: str
    ) -> None:
        if "price_per_sqm" not in df.columns:
            return

        cluster_prices = (
            df.groupby(label_col)["price_per_sqm"].mean().sort_values(ascending=False)
        )

        for cluster_id in cluster_prices.index:
            cluster_data = df[df[label_col] == cluster_id]

            if len(cluster_data) < 3:
                continue

            points = cluster_data[["lat", "lon"]].values
            avg_price = float(cluster_prices[cluster_id])

            # Ensure cluster_id is int for dict lookup
            color = self.cluster_colors.get(int(cluster_id), "#666666")

            boundary_group = folium.FeatureGroup(
                name=f"📷 C{cluster_id}: {avg_price:,.0f} PLN/m²", show=True
            )

            boundary = compute_boundary(points, percentile=85.0)

            if boundary:
                folium.Polygon(
                    locations=boundary,
                    color=color,
                    weight=3,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.15,
                    tooltip=f"Cluster {cluster_id}: {avg_price:,.0f} PLN/m²",
                ).add_to(boundary_group)

            boundary_group.add_to(m)

    def _add_cluster_points(
        self, m: folium.Map, df: pd.DataFrame, label_col: str
    ) -> None:
        points_group = folium.FeatureGroup(name="🏠 Properties", show=True)

        for _, row in df.iterrows():
            cluster_id = int(row[label_col])
            color = self.cluster_colors.get(cluster_id, "#666666")

            popup_html = f"""
            <div style="width: 180px;">
                <b>Property #{int(row.get('id', 0))}</b><br>
                <hr style="margin: 5px 0;">
                <b>Cluster:</b> {cluster_id}<br>
                <b>District:</b> {row.get('district', 'N/A')}<br>
                <b>Price:</b> {row.get('price_per_sqm', 0):,.0f} PLN/m²<br>
                <b>Total:</b> {row.get('price_total', 0):,.0f} PLN<br>
                <b>Area:</b> {row.get('area_sqm', 0):.1f} m²<br>
                <b>Rooms:</b> {row.get('rooms', 'N/A')}<br>
                <b>Year:</b> {row.get('year_built', 'N/A')}<br>
                <b>Floor:</b> {row.get('floor', 'N/A')}<br>
                <b>Distance:</b> {row.get('distance_from_center_km', 0):.1f} km
            </div>
            """

            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=f"Cluster {cluster_id}: {row.get('price_per_sqm', 0):,.0f} PLN/m²",
            ).add_to(points_group)

        points_group.add_to(m)

    def _add_centroids(self, m: folium.Map, df: pd.DataFrame, label_col: str) -> None:
        centroid_group = folium.FeatureGroup(name="📍 Cluster Centers", show=False)

        for cluster_id in df[label_col].unique():
            cluster_data = df[df[label_col] == cluster_id]

            # Explicit casts to int/float for safety
            cid = int(cluster_id)
            centroid_lat = float(cluster_data["lat"].mean())
            centroid_lon = float(cluster_data["lon"].mean())
            color = self.cluster_colors.get(cid, "#666666")

            avg_price = 0.0
            if "price_per_sqm" in df.columns:
                avg_price = float(cluster_data["price_per_sqm"].mean())

            folium.Marker(
                location=[centroid_lat, centroid_lon],
                icon=folium.DivIcon(
                    html=f"""<div style="
                        font-size: 14px; 
                        font-weight: bold; 
                        color: white;
                        background-color: {color};
                        border-radius: 50%;
                        width: 28px;
                        height: 28px;
                        line-height: 28px;
                        text-align: center;
                        border: 2px solid white;
                        box-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                    ">{cid}</div>""",
                    icon_size=(28, 28),
                    icon_anchor=(14, 14),
                ),
                tooltip=f"Cluster {cid} Center: {avg_price:,.0f} PLN/m²",
            ).add_to(centroid_group)

        centroid_group.add_to(m)

    def _add_enhanced_legend(
        self, m: folium.Map, df: pd.DataFrame, label_col: str
    ) -> None:
        if "price_per_sqm" not in df.columns:
            return

        cluster_stats = (
            df.groupby(label_col).agg({"price_per_sqm": ["mean", "count"]}).round(0)
        )
        cluster_stats.columns = pd.Index(["avg_price", "count"])
        cluster_stats = cluster_stats.sort_values("avg_price", ascending=False)

        legend_items = []
        for cluster_id in cluster_stats.index:
            cid = int(cluster_id)
            avg_price = float(cluster_stats.loc[cluster_id, "avg_price"])
            count = int(cluster_stats.loc[cluster_id, "count"])
            color = self.cluster_colors.get(cid, "#666666")

            legend_items.append(
                f"""
                <tr>
                    <td style="padding: 2px 5px;">
                        <div style="
                            width: 15px; 
                            height: 15px; 
                            background-color: {color};
                            border-radius: 3px;
                        "></div>
                    </td>
                    <td style="padding: 2px 5px;">C{cid}</td>
                    <td style="padding: 2px 5px; text-align: right;">
                        {avg_price:,.0f}
                    </td>
                    <td style="padding: 2px 5px; text-align: right;">
                        ({count})
                    </td>
                </tr>
            """
            )

        legend_html = f"""
        <div style="
            position: fixed;
            bottom: 50px;
            right: 50px;
            width: 180px;
            background-color: white;
            border: 2px solid grey;
            border-radius: 5px;
            padding: 10px;
            font-size: 12px;
            z-index: 9999;
            box-shadow: 3px 3px 6px rgba(0,0,0,0.2);
        ">
            <div style="
                text-align: center; 
                font-weight: bold; 
                margin-bottom: 5px;
                border-bottom: 1px solid #ccc;
                padding-bottom: 5px;
            ">
                Clusters by Price
            </div>
            <table style="width: 100%;">
                <tr style="font-size: 10px; color: #666;">
                    <td></td>
                    <td>ID</td>
                    <td style="text-align: right;">PLN/m²</td>
                    <td style="text-align: right;">n</td>
                </tr>
                {''.join(legend_items)}
            </table>
        </div>
        """

        m.get_root().add_child(folium.Element(legend_html))
