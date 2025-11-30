"""
Report generation module.

Creates comprehensive HTML reports with:
- Summary statistics and metrics
- Embedded plots and charts
- Interactive map links
- Cluster comparison tables
- Quality assessment

IMPORTANT: Clusters sorted by price (C0 = highest price)
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class ReportGenerator:
    """Generates comprehensive HTML analysis reports."""
    
    def __init__(self, title: str = "Warsaw Real Estate Clustering Analysis"):
        """
        Initialize report generator.
        
        Args:
            title: Report title
        """
        self.title = title
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_html_report(
        self,
        df: pd.DataFrame,
        metrics: Dict[str, float],
        map_path: Optional[Path] = None,
        plots_dir: Optional[Path] = None,
        output_path: Optional[Path] = None,
        label_col: str = "cluster",
        elbow_data: Optional[tuple] = None,
        silhouette_data: Optional[tuple] = None
    ) -> str:
        """
        Generate comprehensive HTML report.
        """
        # Compute cluster statistics (sorted by price)
        stats_df = self._compute_cluster_stats(df, label_col)
        
        # Compute district statistics
        district_stats = self._compute_district_stats(df, label_col)
        
        # Generate HTML sections
        html = self._generate_header()
        html += self._generate_summary_section(df, metrics, label_col)
        html += self._generate_cluster_table(stats_df)
        html += self._generate_metrics_section(metrics)
        
        if plots_dir:
            html += self._generate_plots_section(plots_dir)
        
        if district_stats is not None:
            html += self._generate_district_section(district_stats)
        
        if map_path:
            html += self._generate_map_section(map_path)
        
        html += self._generate_methodology_section()
        html += self._generate_footer()
        
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info(f"HTML report saved to {output_path}")
        
        return html
    
    def _generate_header(self) -> str:
        """Generate HTML header with styles."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        :root {{
            --primary: #2c3e50;
            --secondary: #3498db;
            --success: #27ae60;
            --warning: #f39c12;
            --danger: #e74c3c;
            --light: #ecf0f1;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, var(--primary) 0%, #1a252f 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header .timestamp {{ opacity: 0.8; font-size: 0.95em; }}
        .content {{ padding: 40px; }}
        .section {{ margin-bottom: 40px; }}
        
        .section-title {{
            font-size: 1.5em;
            color: var(--primary);
            border-left: 4px solid var(--secondary);
            padding-left: 15px;
            margin-bottom: 20px;
        }}
        
        .cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        
        .card:hover {{ transform: translateY(-5px); }}
        .card-value {{ font-size: 2.2em; font-weight: bold; color: var(--secondary); margin-bottom: 5px; }}
        .card-label {{ font-size: 0.85em; color: #666; text-transform: uppercase; letter-spacing: 1px; }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        th, td {{ padding: 15px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: var(--primary); color: white; font-weight: 600; text-transform: uppercase; font-size: 0.85em; }}
        tr:hover {{ background-color: #f8f9fa; }}
        
        .cluster-badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            font-size: 0.9em;
        }}
        
        .metric-good {{ color: var(--success); }}
        .metric-ok {{ color: var(--warning); }}
        .metric-bad {{ color: var(--danger); }}
        
        .plots-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
        }}
        
        .plot-card {{
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        .plot-card img {{ width: 100%; height: auto; display: block; }}
        .plot-card .plot-title {{ padding: 15px; background: var(--light); font-weight: 600; color: var(--primary); }}
        
        .map-link {{
            display: inline-block;
            background: linear-gradient(135deg, var(--secondary) 0%, #2980b9 100%);
            color: white;
            padding: 15px 30px;
            border-radius: 30px;
            text-decoration: none;
            font-weight: 600;
            font-size: 1.1em;
            box-shadow: 0 4px 15px rgba(52, 152, 219, 0.4);
            transition: all 0.3s ease;
        }}
        
        .map-link:hover {{ transform: translateY(-3px); box-shadow: 0 6px 20px rgba(52, 152, 219, 0.6); }}
        
        .interpretation {{
            background: #f0f4f8;
            padding: 15px 20px;
            border-radius: 8px;
            border-left: 4px solid var(--secondary);
            margin: 15px 0;
        }}
        
        .footer {{
            background: var(--light);
            padding: 30px 40px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        
        .methodology {{
            background: #fafbfc;
            padding: 25px;
            border-radius: 12px;
            margin-top: 20px;
        }}
        
        .methodology h4 {{ color: var(--primary); margin-bottom: 10px; }}
        .methodology ul {{ margin-left: 20px; }}
        .methodology li {{ margin: 8px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏠 {self.title}</h1>
            <p class="timestamp">Generated: {self.timestamp}</p>
        </div>
        <div class="content">
"""
    
    def _generate_summary_section(self, df: pd.DataFrame, metrics: Dict[str, float], label_col: str) -> str:
        """Generate summary statistics cards."""
        n_properties = len(df)
        n_clusters = df[label_col].nunique()
        n_districts = df["district"].nunique() if "district" in df.columns else 0
        silhouette = metrics.get("silhouette", 0)
        avg_price = df["price_per_sqm"].mean() if "price_per_sqm" in df.columns else 0
        
        return f"""
            <div class="section">
                <h2 class="section-title">📊 Overview</h2>
                <div class="cards">
                    <div class="card">
                        <div class="card-value">{n_properties:,}</div>
                        <div class="card-label">Properties</div>
                    </div>
                    <div class="card">
                        <div class="card-value">{n_clusters}</div>
                        <div class="card-label">Clusters</div>
                    </div>
                    <div class="card">
                        <div class="card-value">{n_districts}</div>
                        <div class="card-label">Districts</div>
                    </div>
                    <div class="card">
                        <div class="card-value">{silhouette:.3f}</div>
                        <div class="card-label">Silhouette Score</div>
                    </div>
                    <div class="card">
                        <div class="card-value">{avg_price:,.0f}</div>
                        <div class="card-label">Avg Price (PLN/m²)</div>
                    </div>
                </div>
            </div>
"""
    
    def _compute_cluster_stats(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        """Compute cluster statistics sorted by price."""
        if "price_per_sqm" not in df.columns:
            return pd.DataFrame()
        
        agg_dict = {"price_per_sqm": ["mean", "std", "min", "max"]}
        if "area_sqm" in df.columns:
            agg_dict["area_sqm"] = "mean"
        if "distance_from_center_km" in df.columns:
            agg_dict["distance_from_center_km"] = "mean"
        
        stats = df.groupby(label_col).agg(agg_dict)
        stats.columns = ["_".join(col).strip() for col in stats.columns]
        stats = stats.reset_index()
        stats["count"] = df.groupby(label_col).size().values
        
        # Rename columns
        stats.columns = [c.replace("price_per_sqm_", "").replace("_mean", "") for c in stats.columns]
        stats = stats.rename(columns={
            "mean": "avg_price", "std": "std_price", "min": "min_price", "max": "max_price",
            "area_sqm": "avg_area", "distance_from_center_km": "avg_distance"
        })
        
        # Sort by price descending
        stats = stats.sort_values("avg_price", ascending=False).reset_index(drop=True)
        
        return stats
    
    def _generate_cluster_table(self, stats: pd.DataFrame) -> str:
        """Generate cluster statistics table."""
        if stats.empty:
            return ""
        
        n = len(stats)
        rows = []
        
        for i, row in stats.iterrows():
            ratio = i / (n - 1) if n > 1 else 0.5
            color = "#e74c3c" if ratio < 0.33 else "#f39c12" if ratio < 0.66 else "#27ae60"
            
            avg_area = row.get("avg_area", 0)
            avg_dist = row.get("avg_distance", 0)
            
            rows.append(f"""
                <tr>
                    <td><span class="cluster-badge" style="background-color: {color};">C{int(row['cluster'])}</span></td>
                    <td>{row['avg_price']:,.0f}</td>
                    <td>±{row['std_price']:,.0f}</td>
                    <td>{row['min_price']:,.0f} - {row['max_price']:,.0f}</td>
                    <td>{avg_area:.1f}</td>
                    <td>{avg_dist:.1f}</td>
                    <td><strong>{int(row['count'])}</strong></td>
                </tr>
            """)
        
        return f"""
            <div class="section">
                <h2 class="section-title">📈 Cluster Statistics</h2>
                <p style="margin-bottom: 15px; color: #666;">Clusters sorted by average price (C0 = most expensive)</p>
                <table>
                    <thead>
                        <tr>
                            <th>Cluster</th>
                            <th>Avg Price (PLN/m²)</th>
                            <th>Std Dev</th>
                            <th>Price Range</th>
                            <th>Avg Area (m²)</th>
                            <th>Avg Distance (km)</th>
                            <th>Count</th>
                        </tr>
                    </thead>
                    <tbody>{"".join(rows)}</tbody>
                </table>
            </div>
"""
    
    def _generate_metrics_section(self, metrics: Dict[str, float]) -> str:
        """Generate quality metrics section."""
        silhouette = metrics.get("silhouette", 0)
        calinski = metrics.get("calinski_harabasz", 0)
        davies = metrics.get("davies_bouldin", 0)
        inertia = metrics.get("inertia", 0)
        
        if silhouette > 0.5:
            sil_class, sil_text = "metric-good", "Strong cluster structure"
        elif silhouette > 0.25:
            sil_class, sil_text = "metric-ok", "Reasonable structure"
        else:
            sil_class, sil_text = "metric-bad", "Weak structure"
        
        return f"""
            <div class="section">
                <h2 class="section-title">🎯 Clustering Quality Metrics</h2>
                <table>
                    <thead>
                        <tr><th>Metric</th><th>Value</th><th>Optimal</th><th>Interpretation</th></tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Silhouette Score</strong></td>
                            <td class="{sil_class}"><strong>{silhouette:.4f}</strong></td>
                            <td>→ 1.0</td>
                            <td>{sil_text}</td>
                        </tr>
                        <tr>
                            <td><strong>Calinski-Harabasz Index</strong></td>
                            <td>{calinski:.2f}</td>
                            <td>Higher = better</td>
                            <td>Cluster separation</td>
                        </tr>
                        <tr>
                            <td><strong>Davies-Bouldin Index</strong></td>
                            <td>{davies:.4f}</td>
                            <td>→ 0.0</td>
                            <td>Cluster compactness</td>
                        </tr>
                        <tr>
                            <td><strong>Inertia</strong></td>
                            <td>{inertia:.2f}</td>
                            <td>Lower = tighter</td>
                            <td>Within-cluster sum of squares</td>
                        </tr>
                    </tbody>
                </table>
            </div>
"""
    
    def _generate_plots_section(self, plots_dir: Path) -> str:
        """Generate section with embedded plot images."""
        plot_info = [
            ("analysis_overview.png", "📊 Cluster Analysis Overview"),
            ("dendrogram_heatmap.png", "🌳 Hierarchical Dendrogram with Heatmap"),
            ("silhouette_profile.png", "📉 Silhouette Profile"),
            ("district_distribution.png", "🗺️ District Distribution"),
        ]
        
        plots = []
        for filename, title in plot_info:
            if (plots_dir / filename).exists():
                plots.append(f"""
                    <div class="plot-card">
                        <div class="plot-title">{title}</div>
                        <img src="../plots/{filename}" alt="{title}">
                    </div>
                """)
        
        if not plots:
            return ""
        
        return f"""
            <div class="section">
                <h2 class="section-title">📊 Analysis Plots</h2>
                <div class="plots-grid">{"".join(plots)}</div>
            </div>
"""
    
    def _compute_district_stats(self, df: pd.DataFrame, label_col: str) -> Optional[pd.DataFrame]:
        """Compute district-level statistics."""
        if "district" not in df.columns or "price_per_sqm" not in df.columns:
            return None
        
        stats = df.groupby("district").agg({
            "price_per_sqm": ["mean", "std", "count"],
            label_col: lambda x: x.mode().iloc[0] if len(x) > 0 else 0
        }).reset_index()
        
        stats.columns = ["district", "avg_price", "std_price", "count", "dominant_cluster"]
        stats = stats.sort_values("avg_price", ascending=False)
        
        return stats
    
    def _generate_district_section(self, stats: pd.DataFrame) -> str:
        """Generate district statistics table."""
        rows = "".join([f"""
            <tr>
                <td><strong>{row['district']}</strong></td>
                <td>{row['avg_price']:,.0f}</td>
                <td>±{row['std_price']:,.0f}</td>
                <td>{int(row['count'])}</td>
                <td>C{int(row['dominant_cluster'])}</td>
            </tr>
        """ for _, row in stats.iterrows()])
        
        return f"""
            <div class="section">
                <h2 class="section-title">🏘️ District Statistics</h2>
                <table>
                    <thead>
                        <tr><th>District</th><th>Avg Price</th><th>Std Dev</th><th>Properties</th><th>Dominant Cluster</th></tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
"""
    
    def _generate_map_section(self, map_path: Path) -> str:
        """Generate interactive map section."""
        return f"""
            <div class="section" style="text-align: center;">
                <h2 class="section-title">🗺️ Interactive Map</h2>
                <p style="margin-bottom: 20px; color: #666;">
                    Explore clusters with toggleable layers for each cluster.
                </p>
                <a href="../maps/{map_path.name}" class="map-link" target="_blank">🗺️ Open Interactive Map</a>
            </div>
"""
    
    def _generate_methodology_section(self) -> str:
        """Generate methodology explanation."""
        return """
            <div class="section">
                <h2 class="section-title">📚 Methodology</h2>
                <div class="methodology">
                    <h4>K-Means Clustering</h4>
                    <ul>
                        <li>Features: price/m², area, distance from center</li>
                        <li>StandardScaler normalization</li>
                        <li>Optimal k via silhouette + elbow method</li>
                    </ul>
                    <h4 style="margin-top: 15px;">District Boundaries</h4>
                    <ul>
                        <li>Real polygons from OpenStreetMap</li>
                        <li>18 Warsaw districts</li>
                    </ul>
                </div>
            </div>
"""
    
    def _generate_footer(self) -> str:
        """Generate report footer."""
        return f"""
        </div>
        <div class="footer">
            <p>Warsaw Real Estate Clustering | Zaawansowana Eksploracja Danych | {datetime.now().year}</p>
        </div>
    </div>
</body>
</html>
"""
    
    def generate_text_summary(self, df: pd.DataFrame, metrics: Dict[str, float], label_col: str = "cluster") -> str:
        """Generate plain text summary."""
        lines = [
            "=" * 60,
            f"  {self.title.upper()}",
            "=" * 60,
            f"Generated: {self.timestamp}",
            f"Properties: {len(df):,}  |  Clusters: {df[label_col].nunique()}",
            f"Silhouette: {metrics.get('silhouette', 0):.4f}",
            "=" * 60
        ]
        return "\n".join(lines)
