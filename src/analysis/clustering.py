"""
Clustering module for Warsaw real estate analysis.

Provides K-means and DBSCAN clustering implementations with
automatic metric calculation and label management.
"""

from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_samples, silhouette_score

from config.logging_config import get_logger

logger = get_logger(__name__)


class KMeansClusterer:
    """
    K-means clustering with automatic metric computation.
    
    Example:
        >>> clusterer = KMeansClusterer(n_clusters=5)
        >>> clusterer.fit(features)
        >>> df = clusterer.assign_labels_to_df(df)
        >>> metrics = clusterer.get_metrics()
    """
    
    def __init__(
        self,
        n_clusters: int = 5,
        random_state: Optional[int] = 42,
        n_init: int = 10,
        max_iter: int = 300
    ):
        """
        Initialize the K-means clusterer.
        
        Args:
            n_clusters: Number of clusters
            random_state: Random seed for reproducibility
            n_init: Number of initializations
            max_iter: Maximum iterations per initialization
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        
        # Type-annotated attributes
        self.model: Optional[KMeans] = None
        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self._metrics: Dict[str, float] = {}
        self._sample_silhouettes: Optional[np.ndarray] = None
        self._cluster_silhouettes: Dict[int, float] = {}
        
        logger.debug(f"Initialized KMeansClusterer with k={n_clusters}")
    
    def fit(self, X: np.ndarray) -> "KMeansClusterer":
        """
        Fit the K-means model to data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If X has fewer samples than clusters
        """
        # Guard clause
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"Need at least {self.n_clusters} samples, got {X.shape[0]}"
            )
        
        logger.info(f"Fitting K-means with k={self.n_clusters}...")
        
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter
        )
        
        self.labels_ = self.model.fit_predict(X)
        self.cluster_centers_ = self.model.cluster_centers_
        
        # Compute metrics
        self._compute_metrics(X)
        
        logger.info(
            f"K-means complete. Silhouette: {self._metrics.get('silhouette', 0):.4f}"
        )
        
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit model and return labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Cluster labels
        """
        self.fit(X)
        
        # Guard clause
        if self.labels_ is None:
            raise ValueError("Model fitting failed; labels are None.")
        
        return self.labels_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Cluster labels
            
        Raises:
            ValueError: If model not fitted
        """
        # Guard clause
        if self.model is None:
            raise ValueError("Model must be fitted before predict")
        
        return cast(np.ndarray, self.model.predict(X))
    
    def _compute_metrics(self, X: np.ndarray) -> None:
        """
        Compute clustering quality metrics.
        
        Args:
            X: Feature matrix used for clustering
        """
        # Guard clause
        if self.model is None or self.labels_ is None:
            logger.warning("Cannot compute metrics: Model not fitted")
            return
        
        # Check for degenerate clustering
        unique_labels = np.unique(self.labels_)
        if len(unique_labels) < 2:
            logger.warning("Less than 2 clusters found")
            self._metrics = {
                "silhouette": 0.0,
                "inertia": float(self.model.inertia_),
                "calinski_harabasz": 0.0
            }
            return
        
        # Silhouette score (overall)
        self._metrics["silhouette"] = float(silhouette_score(X, self.labels_))
        
        # Inertia (within-cluster sum of squares)
        self._metrics["inertia"] = float(self.model.inertia_)
        
        # Calinski-Harabasz index
        self._metrics["calinski_harabasz"] = float(
            calinski_harabasz_score(X, self.labels_)
        )
        
        # Per-sample silhouette scores
        self._sample_silhouettes = silhouette_samples(X, self.labels_)
        
        # Per-cluster silhouette scores
        self._cluster_silhouettes = {}
        for cluster_id in unique_labels:
            mask = self.labels_ == cluster_id
            cluster_silhouette = float(self._sample_silhouettes[mask].mean())
            self._cluster_silhouettes[int(cluster_id)] = cluster_silhouette
    
    def get_metrics(self) -> Dict[str, float]:
        """Get computed metrics."""
        return self._metrics.copy()
    
    def get_silhouette_per_cluster(
        self, 
        X: Optional[np.ndarray] = None
    ) -> Dict[int, float]:
        """
        Get silhouette score for each cluster.
        
        Args:
            X: Optional feature matrix (recomputes if provided)
            
        Returns:
            Dictionary mapping cluster_id to silhouette score
        """
        if X is not None and self.labels_ is not None:
            sample_silhouettes = silhouette_samples(X, self.labels_)
            result = {}
            for cluster_id in np.unique(self.labels_):
                mask = self.labels_ == cluster_id
                result[int(cluster_id)] = float(sample_silhouettes[mask].mean())
            return result
        
        return self._cluster_silhouettes.copy()
    
    def get_sample_silhouettes(self) -> Optional[np.ndarray]:
        """Get per-sample silhouette scores."""
        return self._sample_silhouettes
    
    def assign_labels_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cluster labels to DataFrame.
        
        Args:
            df: DataFrame to augment
            
        Returns:
            DataFrame with 'cluster' column added
            
        Raises:
            ValueError: If labels not available or length mismatch
        """
        # Guard clauses
        if self.labels_ is None:
            raise ValueError("No labels available - fit model first")
        
        if len(self.labels_) != len(df):
            raise ValueError(
                f"Label count ({len(self.labels_)}) doesn't match "
                f"DataFrame length ({len(df)})"
            )
        
        df = df.copy()  # Always copy before modification
        df["cluster"] = self.labels_
        
        return df


class DBSCANClusterer:
    """
    DBSCAN clustering for density-based cluster detection.
    
    Useful for comparing with K-means results and detecting outliers.
    
    Example:
        >>> clusterer = DBSCANClusterer(eps=0.5, min_samples=5)
        >>> clusterer.fit(features)
        >>> df = clusterer.assign_labels_to_df(df)
    """
    
    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean"
    ):
        """
        Initialize DBSCAN clusterer.
        
        Args:
            eps: Maximum distance between two samples in same neighborhood
            min_samples: Minimum samples in neighborhood for core point
            metric: Distance metric
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        
        # Type-annotated attributes
        self.model: Optional[DBSCAN] = None
        self.labels_: Optional[np.ndarray] = None
        self._metrics: Dict[str, Any] = {}
        
        logger.debug(f"Initialized DBSCANClusterer with eps={eps}")
    
    def fit(self, X: np.ndarray) -> "DBSCANClusterer":
        """
        Fit DBSCAN model to data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Self for method chaining
        """
        logger.info(
            f"Fitting DBSCAN with eps={self.eps}, min_samples={self.min_samples}..."
        )
        
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric
        )
        
        self.labels_ = self.model.fit_predict(X)
        
        # Compute metrics
        self._compute_metrics(X)
        
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = int((self.labels_ == -1).sum())
        
        logger.info(
            f"DBSCAN complete. Found {n_clusters} clusters, {n_noise} noise points"
        )
        
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return labels."""
        self.fit(X)
        
        if self.labels_ is None:
            raise ValueError("Model fitting failed")
        
        return self.labels_
    
    def _compute_metrics(self, X: np.ndarray) -> None:
        """Compute clustering metrics."""
        if self.labels_ is None:
            return
        
        unique_labels = set(self.labels_)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = int((self.labels_ == -1).sum())
        
        self._metrics["n_clusters"] = n_clusters
        self._metrics["n_noise"] = n_noise
        self._metrics["noise_ratio"] = n_noise / len(self.labels_)
        
        # Silhouette only if we have 2+ clusters and not all noise
        if n_clusters >= 2 and n_noise < len(self.labels_):
            # Exclude noise points for silhouette
            mask = self.labels_ != -1
            if mask.sum() >= 2:
                try:
                    self._metrics["silhouette"] = float(
                        silhouette_score(X[mask], self.labels_[mask])
                    )
                except Exception as e:
                    logger.debug(f"Could not compute silhouette: {e}")
                    self._metrics["silhouette"] = 0.0
            else:
                self._metrics["silhouette"] = 0.0
        else:
            self._metrics["silhouette"] = 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get computed metrics."""
        return self._metrics.copy()
    
    def assign_labels_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cluster labels to DataFrame.
        
        Note: DBSCAN uses -1 for noise points.
        """
        if self.labels_ is None:
            raise ValueError("No labels available - fit model first")
        
        if len(self.labels_) != len(df):
            raise ValueError("Label count doesn't match DataFrame length")
        
        df = df.copy()
        df["cluster"] = self.labels_
        
        return df
    
    def get_core_samples(self) -> Optional[np.ndarray]:
        """Get indices of core samples."""
        if self.model is None:
            return None
        return self.model.core_sample_indices_
