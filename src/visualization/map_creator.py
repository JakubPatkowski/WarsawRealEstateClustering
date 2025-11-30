"""
Interactive map creation module.

Creates Folium maps with:
- Real district boundaries from OpenStreetMap
- District coloring by average price (green→red gradient)
- Per-cluster toggleable layers (properties + boundaries separately)
- Price heatmap (diffused)
- Cluster centers
- Multiple base layers (Light/Dark/Satellite)

IMPORTANT: Clusters sorted by price (C0 = highest price, C<k> = lowest)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import folium
from folium import plugins
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import mapping
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


def get_price_color(price: float, min_price: float, max_price: float) -> str:
    """
    Get color for price value on green→yellow→red gradient.
    
    Args:
        price: Current price
        min_price: Minimum price in dataset
        max_price: Maximum price in dataset
        
    Returns:
        Hex color string
    """
    if max_price == min_price:
        return "#ffff00"
    
    ratio = (price - min_price) / (max_price - min_price)
    ratio = max(0, min(1, ratio))
    
    # Green (cheap) → Yellow (mid) → Red (expensive)
    if ratio < 0.5:
        # Green to Yellow
        t = ratio * 2
        r = int(50 + 205 * t)
        g = int(180 + 75 * t)
        b = int(50 - 50 * t)
    else:
        # Yellow to Red
        t = (ratio - 0.5) * 2
        r = int(255 - 35 * t)
        g = int(255 - 200 * t)
        b = int(0 + 50 * t)
    
    return f"#{r:02x}{g:02x}{b:02x}"


def get_cluster_colors(
    df: pd.DataFrame,
    label_col: str = "cluster"
) -> Tuple[Dict[int, str], List[int]]:
    """
    Get cluster colors sorted by average price.
    
    Returns:
        Tuple of (cluster_id → color dict, sorted cluster list)
    """
    if "price_per_sqm" not in df.columns:
        clusters = sorted(df[label_col].unique())
        base_colors = ["#e41a1c", "#ff7f00", "#ffff33", "#4daf4a", "#377eb8",
                      "#984ea3", "#a65628", "#f781bf", "#999999", "#66c2a5"]
        return {c: base_colors[i % len(base_colors)] for i, c in enumerate(clusters)}, clusters
    
    # Sort by price descending
    cluster_prices = df.groupby(label_col)["price_per_sqm"].mean().sort_values(ascending=False)
    sorted_clusters = list(cluster_prices.index)
    
    min_price = cluster_prices.min()
    max_price = cluster_prices.max()
    
    colors = {}
    for cluster_id in sorted_clusters:
        price = cluster_prices[cluster_id]
        colors[cluster_id] = get_price_color(price, min_price, max_price)
    
    return colors, sorted_clusters


class MapCreator:
    """Creates interactive Folium maps with cluster visualization."""
    
    def __init__(
        self,
        center: Tuple[float, float] = None,
        zoom_start: int = 11,
        district_polygons: Optional[Any] = None
    ):
        """
        Initialize map creator.
        
        Args:
            center: Map center (lat, lon)
            zoom_start: Initial zoom level
            district_polygons: GeoDataFrame with district boundaries
        """
        self.center = center or settings.warsaw_center
        self.zoom_start = zoom_start
        self.district_polygons = district_polygons
        
        self._cluster_colors: Dict[int, str] = {}
        self._sorted_clusters: List[int] = []
    
    def create_cluster_map(
        self,
        df: pd.DataFrame,
        label_col: str = "cluster",
        output_path: Optional[Path] = None,
        show_heatmap: bool = True,
        show_districts: bool = True
    ) -> folium.Map:
        """
        Create interactive cluster map with all layers.
        
        Features:
        - Base layers: Light, Dark, Satellite
        - District boundaries (thick lines)
        - District price fill (optional layer)
        - Per-cluster property points (toggleable)
        - Per-cluster boundaries/convex hull (toggleable)
        - Price heatmap (diffused)
        - Cluster centers
        
        Args:
            df: DataFrame with lat, lon, cluster, price_per_sqm
            label_col: Cluster column name
            output_path: Path to save HTML
            show_heatmap: Include price heatmap layer
            show_districts: Include district layers
            
        Returns:
            Folium Map object
        """
        # Setup cluster colors
        self._cluster_colors, self._sorted_clusters = get_cluster_colors(df, label_col)
        
        # Create base map
        m = folium.Map(
            location=self.center,
            zoom_start=self.zoom_start,
            tiles=None,
            control_scale=True
        )
        
        # Add base tile layers
        self._add_base_layers(m)
        
        # Add district layers
        if show_districts and self.district_polygons is not None:
            self._add_district_boundaries_layer(m)
            self._add_district_price_fill_layer(m, df)
        
        # Add per-cluster layers (properties + boundaries separately)
        self._add_cluster_layers(m, df, label_col)
        
        # Add price heatmap (diffused)
        if show_heatmap and "price_per_sqm" in df.columns:
            self._add_price_heatmap(m, df)
        
        # Add cluster centers
        self._add_cluster_centers(m, df, label_col)
        
        # Add layer control
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Add legend
        self._add_legend(m, df, label_col)
        
        # Save if path provided
        if output_path:
            m.save(str(output_path))
            logger.info(f"Map saved to {output_path}")
        
        return m
    
    def _add_base_layers(self, m: folium.Map) -> None:
        """Add multiple base tile layers."""
        folium.TileLayer(
            tiles="cartodbpositron",
            name="🗺️ Light",
            control=True
        ).add_to(m)
        
        folium.TileLayer(
            tiles="cartodbdark_matter", 
            name="🌙 Dark",
            control=True
        ).add_to(m)
        
        folium.TileLayer(
            tiles="OpenStreetMap",
            name="🛣️ OpenStreetMap",
            control=True
        ).add_to(m)
        
        # Satellite (Esri)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="🛰️ Satellite",
            control=True
        ).add_to(m)
    
    def _add_district_boundaries_layer(self, m: folium.Map) -> None:
        """Add district boundaries as thick lines (no fill)."""
        if not HAS_GEOPANDAS or self.district_polygons is None:
            return
        
        fg = folium.FeatureGroup(name="📍 District Boundaries", show=True)
        
        for _, row in self.district_polygons.iterrows():
            name = row.get("name", "Unknown")
            geom = row.geometry
            
            # Convert to GeoJSON
            geojson = mapping(geom)
            
            # Style: thick dark line, no fill
            folium.GeoJson(
                geojson,
                style_function=lambda x: {
                    "fillColor": "transparent",
                    "fillOpacity": 0,
                    "color": "#333333",
                    "weight": 3.5,  # THICK border
                    "dashArray": "0"
                },
                tooltip=folium.Tooltip(f"<b>{name}</b>")
            ).add_to(fg)
        
        fg.add_to(m)
    
    def _add_district_price_fill_layer(
        self, 
        m: folium.Map, 
        df: pd.DataFrame
    ) -> None:
        """Add district polygons filled by average price (green→red)."""
        if not HAS_GEOPANDAS or self.district_polygons is None:
            return
        
        if "price_per_sqm" not in df.columns or "district" not in df.columns:
            return
        
        # Calculate average price per district
        district_prices = df.groupby("district")["price_per_sqm"].mean()
        min_price = district_prices.min()
        max_price = district_prices.max()
        
        fg = folium.FeatureGroup(name="🏠 District Avg Price", show=False)
        
        for _, row in self.district_polygons.iterrows():
            name = row.get("name", "Unknown")
            geom = row.geometry
            
            # Find matching district price
            avg_price = None
            for dist_name in district_prices.index:
                if self._normalize_name(dist_name) == self._normalize_name(name):
                    avg_price = district_prices[dist_name]
                    break
            
            if avg_price is None:
                color = "#cccccc"
                tooltip_text = f"<b>{name}</b><br>No data"
            else:
                color = get_price_color(avg_price, min_price, max_price)
                tooltip_text = f"<b>{name}</b><br>Avg: {avg_price:,.0f} PLN/m²"
            
            folium.GeoJson(
                mapping(geom),
                style_function=lambda x, c=color: {
                    "fillColor": c,
                    "fillOpacity": 0.6,
                    "color": "#333333",
                    "weight": 2
                },
                tooltip=folium.Tooltip(tooltip_text)
            ).add_to(fg)
        
        fg.add_to(m)
    
    def _normalize_name(self, name: str) -> str:
        """Normalize district name for matching."""
        replacements = {
            "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n",
            "ó": "o", "ś": "s", "ź": "z", "ż": "z",
            "Ą": "A", "Ć": "C", "Ę": "E", "Ł": "L", "Ń": "N",
            "Ó": "O", "Ś": "S", "Ź": "Z", "Ż": "Z"
        }
        result = name.lower()
        for pl, en in replacements.items():
            result = result.replace(pl, en.lower())
        return result.replace("-", " ").replace("_", " ").strip()
    
    def _add_cluster_layers(
        self,
        m: folium.Map,
        df: pd.DataFrame,
        label_col: str
    ) -> None:
        """Add per-cluster layers with separate properties and boundaries."""
        from scipy.spatial import ConvexHull
        
        for cluster_id in self._sorted_clusters:
            cluster_data = df[df[label_col] == cluster_id]
            color = self._cluster_colors[cluster_id]
            avg_price = cluster_data["price_per_sqm"].mean() if "price_per_sqm" in df.columns else 0
            
            # Layer name with price
            layer_name_props = f"🏠 C{cluster_id} Properties ({avg_price:,.0f} PLN/m²)"
            layer_name_bounds = f"⬡ C{cluster_id} Boundary"
            
            # === PROPERTIES LAYER ===
            fg_props = folium.FeatureGroup(name=layer_name_props, show=True)
            
            for _, row in cluster_data.iterrows():
                price = row.get("price_per_sqm", 0)
                area = row.get("area_sqm", 0)
                district = row.get("district", "N/A")
                rooms = row.get("rooms", "N/A")
                
                popup_html = f"""
                <div style="font-family: Arial; min-width: 150px;">
                    <b style="color: {color};">Cluster {cluster_id}</b><br>
                    <hr style="margin: 5px 0;">
                    <b>Price:</b> {price:,.0f} PLN/m²<br>
                    <b>Area:</b> {area:.1f} m²<br>
                    <b>Rooms:</b> {rooms}<br>
                    <b>District:</b> {district}
                </div>
                """
                
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=6,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=1.5,
                    popup=folium.Popup(popup_html, max_width=200),
                    tooltip=f"C{cluster_id}: {price:,.0f} PLN/m²"
                ).add_to(fg_props)
            
            fg_props.add_to(m)
            
            # === BOUNDARY LAYER (Convex Hull) ===
            if len(cluster_data) >= 3:
                try:
                    points = cluster_data[["lon", "lat"]].values
                    hull = ConvexHull(points)
                    hull_points = [[points[i, 1], points[i, 0]] for i in hull.vertices]
                    hull_points.append(hull_points[0])  # Close polygon
                    
                    fg_bounds = folium.FeatureGroup(name=layer_name_bounds, show=False)
                    
                    folium.Polygon(
                        locations=hull_points,
                        color=color,
                        weight=3,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.15,
                        tooltip=f"Cluster {cluster_id} boundary"
                    ).add_to(fg_bounds)
                    
                    fg_bounds.add_to(m)
                    
                except Exception as e:
                    logger.debug(f"Could not create hull for cluster {cluster_id}: {e}")
    
    def _add_price_heatmap(self, m: folium.Map, df: pd.DataFrame) -> None:
        """Add diffused price heatmap layer."""
        heat_data = [
            [row["lat"], row["lon"], row["price_per_sqm"]]
            for _, row in df.iterrows()
            if pd.notna(row.get("price_per_sqm"))
        ]
        
        if not heat_data:
            return
        
        # Normalize prices for heatmap intensity
        prices = [h[2] for h in heat_data]
        min_p, max_p = min(prices), max(prices)
        
        if max_p > min_p:
            heat_data_norm = [
                [h[0], h[1], (h[2] - min_p) / (max_p - min_p)]
                for h in heat_data
            ]
        else:
            heat_data_norm = [[h[0], h[1], 0.5] for h in heat_data]
        
        # DIFFUSED heatmap settings
        heatmap = plugins.HeatMap(
            heat_data_norm,
            name="🔥 Price Heatmap",
            min_opacity=0.3,
            max_zoom=18,
            radius=25,      # Larger radius for diffusion
            blur=20,        # More blur for smoother look
            gradient={
                0.0: "#00ff00",   # Green (low)
                0.25: "#80ff00",
                0.5: "#ffff00",   # Yellow (mid)
                0.75: "#ff8000",
                1.0: "#ff0000"    # Red (high)
            },
            show=False
        )
        heatmap.add_to(m)
    
    def _add_cluster_centers(
        self,
        m: folium.Map,
        df: pd.DataFrame,
        label_col: str
    ) -> None:
        """Add cluster center markers."""
        fg = folium.FeatureGroup(name="📍 Cluster Centers", show=False)
        
        for cluster_id in self._sorted_clusters:
            cluster_data = df[df[label_col] == cluster_id]
            
            center_lat = cluster_data["lat"].mean()
            center_lon = cluster_data["lon"].mean()
            
            avg_price = cluster_data["price_per_sqm"].mean() if "price_per_sqm" in df.columns else 0
            count = len(cluster_data)
            color = self._cluster_colors[cluster_id]
            
            # Star marker for center
            folium.Marker(
                location=[center_lat, center_lon],
                icon=folium.DivIcon(
                    html=f"""
                    <div style="
                        background-color: {color};
                        border: 3px solid white;
                        border-radius: 50%;
                        width: 30px;
                        height: 30px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: bold;
                        color: white;
                        font-size: 14px;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                    ">C{cluster_id}</div>
                    """,
                    icon_size=(30, 30),
                    icon_anchor=(15, 15)
                ),
                tooltip=f"Cluster {cluster_id} Center<br>Avg: {avg_price:,.0f} PLN/m²<br>n={count}"
            ).add_to(fg)
        
        fg.add_to(m)
    
    def _add_legend(
        self,
        m: folium.Map,
        df: pd.DataFrame,
        label_col: str
    ) -> None:
        """Add legend to map."""
        # Build legend HTML
        legend_items = []
        
        for cluster_id in self._sorted_clusters:
            cluster_data = df[df[label_col] == cluster_id]
            color = self._cluster_colors[cluster_id]
            count = len(cluster_data)
            avg_price = cluster_data["price_per_sqm"].mean() if "price_per_sqm" in df.columns else 0
            
            legend_items.append(f"""
                <div style="display: flex; align-items: center; margin: 3px 0;">
                    <span style="
                        background-color: {color};
                        width: 16px;
                        height: 16px;
                        border-radius: 50%;
                        display: inline-block;
                        margin-right: 8px;
                        border: 1px solid #333;
                    "></span>
                    <span><b>C{cluster_id}</b>: {avg_price:,.0f} PLN/m² (n={count})</span>
                </div>
            """)
        
        legend_html = f"""
        <div style="
            position: fixed;
            bottom: 30px;
            left: 30px;
            z-index: 1000;
            background-color: white;
            padding: 12px 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            font-family: Arial, sans-serif;
            font-size: 12px;
            max-height: 300px;
            overflow-y: auto;
        ">
            <div style="font-weight: bold; margin-bottom: 8px; font-size: 13px;">
                🏠 Clusters (by avg price ↓)
            </div>
            {"".join(legend_items)}
            <hr style="margin: 8px 0; border-color: #eee;">
            <div style="font-size: 10px; color: #666;">
                Green = cheap, Red = expensive
            </div>
        </div>
        """
        
        m.get_root().html.add_child(folium.Element(legend_html))
    
    def create_district_price_map(
        self,
        df: pd.DataFrame,
        output_path: Optional[Path] = None
    ) -> folium.Map:
        """
        Create map showing only district price distribution.
        
        Districts colored by average price with thick boundaries.
        """
        if not HAS_GEOPANDAS or self.district_polygons is None:
            logger.warning("Cannot create district price map without GeoDataFrame")
            return folium.Map(location=self.center, zoom_start=self.zoom_start)
        
        m = folium.Map(
            location=self.center,
            zoom_start=self.zoom_start,
            tiles="cartodbpositron"
        )
        
        # Calculate prices
        if "price_per_sqm" in df.columns and "district" in df.columns:
            district_prices = df.groupby("district")["price_per_sqm"].agg(["mean", "count", "std"])
            min_price = district_prices["mean"].min()
            max_price = district_prices["mean"].max()
        else:
            district_prices = pd.DataFrame()
            min_price = max_price = 0
        
        # Add districts
        for _, row in self.district_polygons.iterrows():
            name = row.get("name", "Unknown")
            geom = row.geometry
            
            # Find matching price
            matched_price = None
            matched_count = 0
            matched_std = 0
            
            for dist_name in district_prices.index:
                if self._normalize_name(dist_name) == self._normalize_name(name):
                    matched_price = district_prices.loc[dist_name, "mean"]
                    matched_count = int(district_prices.loc[dist_name, "count"])
                    matched_std = district_prices.loc[dist_name, "std"]
                    break
            
            if matched_price is not None:
                color = get_price_color(matched_price, min_price, max_price)
                tooltip = f"""
                <b>{name}</b><br>
                Avg: {matched_price:,.0f} PLN/m²<br>
                Std: {matched_std:,.0f}<br>
                n={matched_count}
                """
            else:
                color = "#cccccc"
                tooltip = f"<b>{name}</b><br>No data"
            
            folium.GeoJson(
                mapping(geom),
                style_function=lambda x, c=color: {
                    "fillColor": c,
                    "fillOpacity": 0.65,
                    "color": "#222222",
                    "weight": 3.5
                },
                tooltip=folium.Tooltip(tooltip)
            ).add_to(m)
        
        # Add legend
        self._add_price_legend(m, min_price, max_price)
        
        if output_path:
            m.save(str(output_path))
            logger.info(f"District price map saved to {output_path}")
        
        return m
    
    def _add_price_legend(
        self,
        m: folium.Map,
        min_price: float,
        max_price: float
    ) -> None:
        """Add gradient price legend."""
        legend_html = f"""
        <div style="
            position: fixed;
            bottom: 30px;
            right: 30px;
            z-index: 1000;
            background-color: white;
            padding: 12px 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            font-family: Arial, sans-serif;
            font-size: 12px;
        ">
            <div style="font-weight: bold; margin-bottom: 8px;">
                💰 Avg Price (PLN/m²)
            </div>
            <div style="
                width: 150px;
                height: 15px;
                background: linear-gradient(to right, #32b432, #ffff00, #dc3c28);
                border-radius: 3px;
                margin-bottom: 5px;
            "></div>
            <div style="display: flex; justify-content: space-between; font-size: 10px;">
                <span>{min_price:,.0f}</span>
                <span>{(min_price + max_price) / 2:,.0f}</span>
                <span>{max_price:,.0f}</span>
            </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
