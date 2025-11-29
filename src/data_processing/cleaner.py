"""
Data cleaning module.

Provides data cleaning and outlier removal functionality.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class DataCleaner:
    """
    Cleans and preprocesses real estate data.
    
    Handles missing values, outliers, and data type issues.
    
    Example:
        >>> cleaner = DataCleaner(outlier_method="iqr")
        >>> df_clean = cleaner.clean(df)
        >>> report = cleaner.get_cleaning_report()
    """
    
    # Required columns that must exist
    REQUIRED_COLUMNS = ["lat", "lon", "price_per_sqm"]
    
    # Columns with expected ranges
    NUMERIC_RANGES = {
        "lat": (52.0, 52.5),
        "lon": (20.7, 21.4),
        "price_per_sqm": (3000, 50000),
        "area_sqm": (15, 500),
        "rooms": (1, 10),
        "year_built": (1900, 2025),
        "floor": (0, 50),
        "distance_from_center_km": (0, 30)
    }
    
    def __init__(
        self,
        outlier_method: str = "iqr",
        outlier_threshold: float = 2.5,
        remove_missing: bool = True
    ):
        """
        Initialize the cleaner.
        
        Args:
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'none')
            outlier_threshold: Threshold for outlier detection
            remove_missing: Whether to remove rows with missing required values
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.remove_missing = remove_missing
        
        # Report tracking
        self._cleaning_report: Dict[str, Any] = {}
        
        logger.debug(
            f"Initialized DataCleaner with method={outlier_method}, "
            f"threshold={outlier_threshold}"
        )
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()  # ALWAYS copy before modification
        original_len = len(df)
        
        logger.info(f"Starting cleaning on {original_len} rows...")
        
        self._cleaning_report = {
            "original_rows": original_len,
            "steps": []
        }
        
        # Step 1: Remove duplicates
        df = self._remove_duplicates(df)
        
        # Step 2: Handle missing values
        if self.remove_missing:
            df = self._handle_missing(df)
        
        # Step 3: Fix data types
        df = self._fix_data_types(df)
        
        # Step 4: Remove out-of-range values
        df = self._remove_out_of_range(df)
        
        # Step 5: Remove outliers
        if self.outlier_method != "none":
            df = self._remove_outliers(df)
        
        # Step 6: Reset index
        df = df.reset_index(drop=True)
        
        # Final report
        final_len = len(df)
        removed = original_len - final_len
        self._cleaning_report["final_rows"] = final_len
        self._cleaning_report["removed_rows"] = removed
        self._cleaning_report["removal_pct"] = removed / original_len * 100
        
        logger.info(
            f"Cleaning complete: {original_len} -> {final_len} rows "
            f"({removed} removed, {self._cleaning_report['removal_pct']:.1f}%)"
        )
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        before = len(df)
        
        # Check for ID column
        if "id" in df.columns:
            df = df.drop_duplicates(subset=["id"])
        else:
            df = df.drop_duplicates()
        
        removed = before - len(df)
        if removed > 0:
            logger.debug(f"Removed {removed} duplicate rows")
            self._cleaning_report["steps"].append(
                f"Removed {removed} duplicates"
            )
        
        return df
    
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        before = len(df)
        
        # Check required columns
        for col in self.REQUIRED_COLUMNS:
            if col in df.columns:
                df = df.dropna(subset=[col])
        
        removed = before - len(df)
        if removed > 0:
            logger.debug(f"Removed {removed} rows with missing required values")
            self._cleaning_report["steps"].append(
                f"Removed {removed} rows with missing values"
            )
        
        # Fill optional missing values
        if "district" in df.columns:
            mode = df["district"].mode()
            fill_value = mode[0] if len(mode) > 0 else "Unknown"
            df["district"] = df["district"].fillna(fill_value)
        
        return df
    
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure correct data types."""
        # Numeric columns
        numeric_cols = [
            "lat", "lon", "price_per_sqm", "price_total", 
            "area_sqm", "distance_from_center_km"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Integer columns
        int_cols = ["id", "rooms", "year_built", "floor"]
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(0).astype(int)
        
        return df
    
    def _remove_out_of_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with values outside expected ranges."""
        before = len(df)
        
        for col, (min_val, max_val) in self.NUMERIC_RANGES.items():
            if col in df.columns:
                mask = (df[col] >= min_val) & (df[col] <= max_val)
                invalid_count = (~mask).sum()
                if invalid_count > 0:
                    logger.debug(
                        f"Removing {invalid_count} rows with {col} "
                        f"outside [{min_val}, {max_val}]"
                    )
                    df = df[mask]
        
        removed = before - len(df)
        if removed > 0:
            self._cleaning_report["steps"].append(
                f"Removed {removed} rows with out-of-range values"
            )
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers."""
        before = len(df)
        
        if self.outlier_method == "iqr":
            df = self._remove_outliers_iqr(df)
        elif self.outlier_method == "zscore":
            df = self._remove_outliers_zscore(df)
        
        removed = before - len(df)
        if removed > 0:
            logger.debug(f"Removed {removed} outliers using {self.outlier_method}")
            self._cleaning_report["steps"].append(
                f"Removed {removed} outliers ({self.outlier_method})"
            )
        
        return df
    
    def _remove_outliers_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        if "price_per_sqm" not in df.columns:
            return df
        
        q1 = df["price_per_sqm"].quantile(0.25)
        q3 = df["price_per_sqm"].quantile(0.75)
        iqr = q3 - q1
        
        lower = q1 - self.outlier_threshold * iqr
        upper = q3 + self.outlier_threshold * iqr
        
        return df[(df["price_per_sqm"] >= lower) & (df["price_per_sqm"] <= upper)]
    
    def _remove_outliers_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using Z-score method."""
        if "price_per_sqm" not in df.columns:
            return df
        
        mean = df["price_per_sqm"].mean()
        std = df["price_per_sqm"].std()
        
        if std == 0:
            return df
        
        z_scores = (df["price_per_sqm"] - mean) / std
        
        return df[np.abs(z_scores) <= self.outlier_threshold]
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """Get summary of cleaning operations performed."""
        return self._cleaning_report.copy()


class FeatureTransformer:
    """
    Transforms features for clustering.
    
    Handles scaling and feature selection.
    
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
        Initialize the transformer.
        
        Args:
            scaling_method: Scaling method ('standard', 'minmax', 'none')
            feature_columns: Columns to use as features
        """
        self.scaling_method = scaling_method
        self.feature_columns = feature_columns or settings.clustering.feature_columns
        
        # Fitted parameters
        self._means: Optional[np.ndarray] = None
        self._stds: Optional[np.ndarray] = None
        self._mins: Optional[np.ndarray] = None
        self._maxs: Optional[np.ndarray] = None
        self._fitted: bool = False
        
        logger.debug(
            f"Initialized FeatureTransformer with method={scaling_method}, "
            f"features={feature_columns}"
        )
    
    def fit(self, df: pd.DataFrame) -> "FeatureTransformer":
        """
        Fit transformer to data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Self for method chaining
        """
        # Get available features
        available = [c for c in self.feature_columns if c in df.columns]
        
        if not available:
            raise ValueError(
                f"None of the specified features found in DataFrame. "
                f"Expected: {self.feature_columns}, Found: {list(df.columns)}"
            )
        
        self.feature_columns = available
        data = df[available].values
        
        if self.scaling_method == "standard":
            self._means = data.mean(axis=0)
            self._stds = data.std(axis=0)
            # Prevent division by zero
            self._stds = np.where(self._stds == 0, 1, self._stds)
            
        elif self.scaling_method == "minmax":
            self._mins = data.min(axis=0)
            self._maxs = data.max(axis=0)
            # Prevent division by zero
            ranges = self._maxs - self._mins
            ranges = np.where(ranges == 0, 1, ranges)
            self._maxs = self._mins + ranges
        
        self._fitted = True
        logger.debug(f"Fitted transformer on {len(available)} features")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted parameters.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Scaled feature matrix
            
        Raises:
            ValueError: If not fitted
        """
        if not self._fitted:
            raise ValueError("Transformer not fitted. Call fit() first.")
        
        data = df[self.feature_columns].values
        
        if self.scaling_method == "standard":
            if self._means is None or self._stds is None:
                raise ValueError("Fitting failed: means/stds are None")
            return (data - self._means) / self._stds
            
        elif self.scaling_method == "minmax":
            if self._mins is None or self._maxs is None:
                raise ValueError("Fitting failed: mins/maxs are None")
            return (data - self._mins) / (self._maxs - self._mins)
        
        return data
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names used."""
        return list(self.feature_columns)
