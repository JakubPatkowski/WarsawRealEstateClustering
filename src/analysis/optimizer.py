"""
Cluster optimization module.

Determines optimal number of clusters using elbow method
and silhouette analysis.
"""

from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class ClusterOptimizer:
    """
    Determines optimal number of clusters.
    
    Uses both elbow method (inertia) and silhouette analysis
    to recommend the best k value.
    
    Example:
        >>> optimizer = ClusterOptimizer(k_range=(3, 10))
        >>> optimizer.fit(features)
        >>> optimal_k = optimizer.get_optimal_k()
        >>> print(optimizer.get_summary())
    """
    
    def __init__(
        self,
        k_range: Optional[Tuple[int, int]] = None,
        random_state: Optional[int] = None,
        min_k: int = 3,
        prefer_elbow: bool = False
    ):
        """
        Initialize the optimizer.
        
        Args:
            k_range: Range of k values to test (min, max)
            random_state: Random seed for reproducibility
            min_k: Minimum k to consider (prevents k=2 results)
            prefer_elbow: If True, prefer elbow method over silhouette
        """
        self.k_range = k_range or settings.clustering.k_range
        self.random_state = random_state or settings.clustering.random_state
        self.min_k = min_k
        self.prefer_elbow = prefer_elbow
        
        # Type-annotated attributes
        self._results: Optional[pd.DataFrame] = None
        self._fitted: bool = False
        self._elbow_k: Optional[int] = None
        self._silhouette_k: Optional[int] = None
        self._optimal_k: Optional[int] = None
        
        logger.debug(f"Initialized ClusterOptimizer with k_range={self.k_range}")
    
    def fit(self, X: np.ndarray) -> "ClusterOptimizer":
        """
        Fit optimizer by testing different k values.
        
        Args:
            X: Feature matrix
            
        Returns:
            Self for method chaining
        """
        k_min, k_max = self.k_range
        k_values = list(range(k_min, k_max + 1))
        
        logger.info(f"Testing k values from {k_min} to {k_max}...")
        
        results = []
        
        for k in k_values:
            logger.debug(f"Testing k={k}...")
            
            model = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10
            )
            labels = model.fit_predict(X)
            
            # Calculate metrics
            inertia = float(model.inertia_)
            
            # Silhouette score (need at least 2 clusters)
            if len(np.unique(labels)) >= 2:
                silhouette = float(silhouette_score(X, labels))
            else:
                silhouette = 0.0
            
            results.append({
                "k": k,
                "inertia": inertia,
                "silhouette": silhouette
            })
            
            logger.debug(
                f"  k={k}: inertia={inertia:.2f}, silhouette={silhouette:.4f}"
            )
        
        self._results = pd.DataFrame(results)
        
        # Find optimal k values
        self._find_elbow_k()
        self._find_silhouette_k()
        self._determine_optimal_k()
        
        self._fitted = True
        
        logger.info(
            f"Optimization complete. Elbow k={self._elbow_k}, "
            f"Silhouette k={self._silhouette_k}, "
            f"Optimal k={self._optimal_k}"
        )
        
        return self
    
    def _find_elbow_k(self) -> None:
        """Find optimal k using elbow method."""
        if self._results is None or len(self._results) < 3:
            self._elbow_k = self.min_k
            return
        
        inertias = self._results["inertia"].values
        k_values = self._results["k"].values
        
        # Calculate rate of change (derivative)
        diffs = np.diff(inertias)
        
        # Calculate second derivative to find elbow
        diffs2 = np.diff(diffs)
        
        # Find elbow point (maximum second derivative)
        if len(diffs2) > 0:
            elbow_idx = np.argmax(diffs2) + 1  # +1 to account for diff offset
            # Ensure within valid range
            elbow_idx = max(0, min(elbow_idx, len(k_values) - 1))
            elbow_k = int(k_values[elbow_idx])
        else:
            elbow_k = int(k_values[len(k_values) // 2])
        
        # Apply minimum k constraint
        self._elbow_k = max(self.min_k, elbow_k)
    
    def _find_silhouette_k(self) -> None:
        """Find optimal k using silhouette analysis."""
        if self._results is None:
            self._silhouette_k = self.min_k
            return
        
        # Filter to k >= min_k
        valid_results = self._results[self._results["k"] >= self.min_k]
        
        if len(valid_results) == 0:
            self._silhouette_k = self.min_k
            return
        
        # Find k with maximum silhouette score
        best_idx = valid_results["silhouette"].idxmax()
        val = self._results.loc[best_idx, "k"]
        self._silhouette_k = int(cast(float, val))
    
    def _determine_optimal_k(self) -> None:
        """Determine final optimal k."""
        if self._elbow_k is None:
            self._elbow_k = self.min_k
        if self._silhouette_k is None:
            self._silhouette_k = self.min_k
        
        if self.prefer_elbow:
            self._optimal_k = self._elbow_k
        else:
            # Default: prefer silhouette, but consider elbow if close
            if abs(self._elbow_k - self._silhouette_k) <= 1:
                # If close, use silhouette
                self._optimal_k = self._silhouette_k
            else:
                # If different, prefer silhouette but log warning
                logger.warning(
                    f"Elbow ({self._elbow_k}) and silhouette ({self._silhouette_k}) "
                    "suggest different k values. Using silhouette."
                )
                self._optimal_k = self._silhouette_k
    
    def get_optimal_k(self) -> int:
        """
        Get the recommended optimal k value.
        
        Returns:
            Optimal number of clusters
            
        Raises:
            ValueError: If optimizer not fitted
        """
        if not self._fitted or self._optimal_k is None:
            raise ValueError("Optimizer not fitted. Call fit() first.")
        
        return self._optimal_k
    
    def get_elbow_k(self) -> Optional[int]:
        """Get k from elbow method."""
        return self._elbow_k
    
    def get_silhouette_k(self) -> Optional[int]:
        """Get k from silhouette analysis."""
        return self._silhouette_k
    
    def get_results(self) -> Optional[pd.DataFrame]:
        """Get full results DataFrame."""
        if self._results is not None:
            return self._results.copy()
        return None
    
    def get_elbow_data(self) -> Tuple[List[int], List[float]]:
        """
        Get data for elbow plot.
        
        Returns:
            Tuple of (k_values, inertias)
        """
        if self._results is None:
            return [], []
        
        return (
            self._results["k"].tolist(),
            self._results["inertia"].tolist()
        )
    
    def get_silhouette_data(self) -> Tuple[List[int], List[float]]:
        """
        Get data for silhouette plot.
        
        Returns:
            Tuple of (k_values, silhouette_scores)
        """
        if self._results is None:
            return [], []
        
        return (
            self._results["k"].tolist(),
            self._results["silhouette"].tolist()
        )
    
    def get_summary(self) -> str:
        """
        Get text summary of optimization results.
        
        Returns:
            Formatted summary string
        """
        if not self._fitted:
            return "Optimizer not fitted."
        
        lines = [
            "=" * 50,
            "CLUSTER OPTIMIZATION SUMMARY",
            "=" * 50,
            f"K range tested: {self.k_range[0]} - {self.k_range[1]}",
            f"Elbow method suggests: k = {self._elbow_k}",
            f"Silhouette analysis suggests: k = {self._silhouette_k}",
            f"RECOMMENDED: k = {self._optimal_k}",
            "",
        ]
        
        if self._results is not None:
            lines.append("Results by k:")
            lines.append("-" * 40)
            for _, row in self._results.iterrows():
                marker = " *" if row["k"] == self._optimal_k else ""
                lines.append(
                    f"  k={int(row['k'])}: silhouette={row['silhouette']:.4f}, "
                    f"inertia={row['inertia']:.0f}{marker}"
                )
        
        return "\n".join(lines)
