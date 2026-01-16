"""
Utility Functions for Biosolids Optimization

Collection of helper functions used across modules:
- Distance calculations
- Data validation
- Conversion helpers
- Logging utilities
"""

import numpy as np
import pandas as pd
from typing import Tuple
from datetime import datetime, timedelta

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two lat/lon points.
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
        
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def validate_submission_format(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate submission DataFrame matches Kaggle format.
    
    Required columns: stp_id, farm_id, day, tons
    
    Args:
        df: Submission DataFrame
        
    Returns:
        (is_valid, error_message)
    """
    required_cols = ["stp_id", "farm_id", "day", "tons"]
    
    for col in required_cols:
        if col not in df.columns:
            return False, f"Missing column: {col}"
    
    # Check data types
    if not pd.api.types.is_numeric_dtype(df["tons"]):
        return False, "Column 'tons' must be numeric"
    
    if not pd.api.types.is_numeric_dtype(df["day"]):
        return False, "Column 'day' must be numeric"
    
    # Check value ranges
    if (df["tons"] < 0).any():
        return False, "Negative tons values found"
    
    if (df["day"] < 1).any() or (df["day"] > 365).any():
        return False, "Day values must be in range 1-365"
    
    return True, "OK"


def day_to_date(day: int, year: int = 2025) -> datetime:
    """
    Convert day-of-year to datetime.
    
    Args:
        day: Day of year (1-365)
        year: Year (default 2025)
        
    Returns:
        datetime object
    """
    return datetime(year, 1, 1) + timedelta(days=day - 1)


def date_to_day(date: datetime) -> int:
    """
    Convert datetime to day-of-year.
    
    Args:
        date: datetime object
        
    Returns:
        Day of year (1-365)
    """
    return (date - datetime(date.year, 1, 1)).days + 1


def kg_to_tons(kg: float) -> float:
    """Convert kilograms to metric tons."""
    return kg / 1000


def tons_to_kg(tons: float) -> float:
    """Convert metric tons to kilograms."""
    return tons * 1000


def calculate_nitrogen_content(biosolid_kg: float, n_percentage: float = 5.0) -> float:
    """
    Calculate nitrogen content in biosolids.
    
    Args:
        biosolid_kg: Biosolid amount in kg
        n_percentage: Nitrogen percentage (default 5%)
        
    Returns:
        Nitrogen content in kg
    """
    return biosolid_kg * (n_percentage / 100)


def estimate_transport_emissions(distance_km: float, tons: float) -> float:
    """
    Estimate CO2 emissions from truck transport.
    
    Formula: 0.9 kg CO2 per km (loaded truck)
    
    Args:
        distance_km: One-way distance
        tons: Load size in tons
        
    Returns:
        CO2 emissions in kg
    """
    # Assume round trip (loaded one way, empty return)
    # Simplification: full emissions for loaded trip
    return distance_km * 0.9


def calculate_fertilizer_offset(nitrogen_kg: float) -> float:
    """
    Calculate CO2 offset from replacing synthetic fertilizer.
    
    Formula: 5.0 kg CO2 saved per kg N
    
    Args:
        nitrogen_kg: Nitrogen amount in kg
        
    Returns:
        CO2 offset in kg (positive value)
    """
    return nitrogen_kg * 5.0


def calculate_soil_carbon_gain(biosolid_kg: float) -> float:
    """
    Calculate CO2 sequestration from biosolid application.
    
    Formula: 0.2 kg CO2 per kg biosolid
    
    Args:
        biosolid_kg: Biosolid amount in kg
        
    Returns:
        CO2 sequestered in kg (positive value)
    """
    return biosolid_kg * 0.2


def format_action_summary(actions: list) -> str:
    """
    Create human-readable summary of actions.
    
    Args:
        actions: List of action dictionaries
        
    Returns:
        Formatted string summary
    """
    if not actions:
        return "No actions taken"
    
    total_tons = sum(a["tons"] for a in actions)
    unique_stps = len(set(a["stp_id"] for a in actions))
    unique_farms = len(set(a["farm_id"] for a in actions))
    
    return (
        f"{len(actions)} shipments | "
        f"{total_tons:.1f} tons total | "
        f"{unique_stps} STPs | "
        f"{unique_farms} farms"
    )


def check_data_integrity(data: dict) -> Tuple[bool, str]:
    """
    Validate loaded data for completeness and consistency.
    
    Args:
        data: Dictionary from data_loader.load_all_data()
        
    Returns:
        (is_valid, error_message)
    """
    required_keys = [
        "config", "stp_registry", "farm_locations",
        "daily_weather", "daily_n_demand", "planting_schedule"
    ]
    
    for key in required_keys:
        if key not in data:
            return False, f"Missing data key: {key}"
    
    # Check for empty DataFrames
    for key in required_keys[1:]:  # Skip config (dict)
        if len(data[key]) == 0:
            return False, f"Empty dataset: {key}"
    
    # Check coordinate validity - FLEXIBLE COLUMN NAMES
    for df_name in ["stp_registry", "farm_locations"]:
        df = data[df_name]
        
        # Find latitude column (could be "latitude", "lat", "Latitude", etc.)
        lat_col = None
        lon_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'lat' in col_lower and 'lon' not in col_lower:
                lat_col = col
            if 'lon' in col_lower or 'lng' in col_lower:
                lon_col = col
        
        # Only validate if coordinate columns exist
        if lat_col and (df[lat_col].abs() > 90).any():
            return False, f"Invalid latitude in {df_name}"
        if lon_col and (df[lon_col].abs() > 180).any():
            return False, f"Invalid longitude in {df_name}"
    
    return True, "OK"