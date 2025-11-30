"""Feature transformation module for clustering preparation."""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config.logging_config import get_logger

logger = get_logger(__name__)


class FeatureTransformer:
    """
    Transform and scale features for clustering.
    
    Example:
        >>> transformer = FeatureTransformer(scaling_method="standard")
        >>> features = transformer.fit_transform(df)
    """
    
    def __init__(
        self,
        scaling_method: str = "standard",
        feature_columns: Optional[List[str]] = None
    ):
        """
        Initialize feature transformer.
        
        Args:
            scaling_method: "standard" (z-score) or "minmax" (0-1 range)
            feature_columns: Columns to use for features
        """
        self.scaling_method = scaling_method
        self.feature_columns = feature_columns or [
            "price_per_sqm", "area_sqm", "distance_from_center_km"
        ]
        
        if scaling_method == "standard":
            self.scaler = StandardScaler()
        elif scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
        
        logger.debug(f"Initialized FeatureTransformer with {scaling_method} scaling")
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit scaler and transform features.
        
        Args:
            df: DataFrame with feature columns
            
        Returns:
            Scaled feature matrix (n_samples, n_features)
        """
        # Get available feature columns
        available = [c for c in self.feature_columns if c in df.columns]
        
        if not available:
            raise ValueError(
                f"No feature columns found. Expected: {self.feature_columns}, "
                f"Got: {list(df.columns)}"
            )
        
        if len(available) < len(self.feature_columns):
            logger.warning(
                f"Some features not found: {set(self.feature_columns) - set(available)}"
            )
        
        features = df[available].values
        scaled = self.scaler.fit_transform(features)
        
        logger.info(
            f"Transformed {len(available)} features for {len(df)} samples "
            f"using {self.scaling_method} scaling"
        )
        
        return scaled
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scaler."""
        available = [c for c in self.feature_columns if c in df.columns]
        features = df[available].values
        return self.scaler.transform(features)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform scaled features back to original scale."""
        return self.scaler.inverse_transform(X)
