"""
Boundaries module.

Provides functionality for fetching and managing Warsaw district boundaries
from OpenStreetMap via Overpass API.
"""

from src.boundaries.district_fetcher import (
    DistrictFetcher,
    stitch_line_segments,
    clean_multipolygon
)

__all__ = [
    "DistrictFetcher",
    "stitch_line_segments",
    "clean_multipolygon"
]
