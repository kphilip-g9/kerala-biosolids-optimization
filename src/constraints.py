"""
Constraint Validation for Biosolids Transport
PATCHED VERSION - Works with farm indices
"""

from typing import Tuple
import numpy as np

# Constants
RAIN_THRESHOLD_MM = 30.0
NITROGEN_BUFFER = 0.10
MAX_TRUCK_CAPACITY_TONS = 20.0
NITROGEN_CONTENT = 0.05

def check_rain_lock(state, farm_idx: int) -> Tuple[bool, str]:
    """
    Check if farm is rain-locked (5-day forecast > 30mm).
    
    FIXED: Uses farm_idx directly
    """
    rainfall_5day = state.get_5day_rainfall(farm_idx)
    
    if rainfall_5day > RAIN_THRESHOLD_MM:
        return False, f"Rain-locked: {rainfall_5day:.1f}mm forecast"
    
    return True, "OK"

def check_nitrogen_limit(state, farm_idx: int, biosolid_kg: float) -> Tuple[bool, str]:
    """Check if application would exceed nitrogen cap."""
    n_in_biosolid = biosolid_kg * NITROGEN_CONTENT
    n_remaining = state.farm_n_remaining[farm_idx]
    n_cap = n_remaining * (1 + NITROGEN_BUFFER)
    
    if n_in_biosolid > n_cap:
        return False, f"Nitrogen excess: {n_in_biosolid:.1f} > {n_cap:.1f} kg"
    
    return True, "OK"

def check_storage_available(state, stp_idx: int, amount_kg: float) -> Tuple[bool, str]:
    """Check if STP has enough biosolids in storage."""
    available = state.stp_storage[stp_idx]
    
    if amount_kg > available:
        return False, f"Insufficient storage: {amount_kg:.0f} > {available:.0f} kg"
    
    return True, "OK"

def check_truck_capacity(tons: float) -> Tuple[bool, str]:
    """Check if shipment fits in truck capacity."""
    if tons > MAX_TRUCK_CAPACITY_TONS:
        return False, f"Exceeds truck capacity: {tons} > {MAX_TRUCK_CAPACITY_TONS} tons"
    
    return True, "OK"

def validate_action(state, stp_idx: int, farm_idx: int, tons: float) -> Tuple[bool, str]:
    """Comprehensive validation of a proposed action."""
    kg = tons * 1000
    
    # Check 1: Truck capacity
    valid, msg = check_truck_capacity(tons)
    if not valid:
        return False, msg
    
    # Check 2: STP has enough biosolids
    valid, msg = check_storage_available(state, stp_idx, kg)
    if not valid:
        return False, msg
    
    # Check 3: Rain lock at farm
    valid, msg = check_rain_lock(state, farm_idx)
    if not valid:
        return False, msg
    
    # Check 4: Nitrogen cap at farm
    valid, msg = check_nitrogen_limit(state, farm_idx, kg)
    if not valid:
        return False, msg
    
    return True, "OK"

def filter_valid_farms(state, stp_idx: int, max_tons: float = MAX_TRUCK_CAPACITY_TONS):
    """
    Get list of farms that STP can validly ship to today.
    
    Returns:
        List of (farm_idx, max_allowed_tons) tuples
    """
    valid_farms = []
    n_farms = len(state.farm_locations)
    
    for farm_idx in range(n_farms):
        # Skip farms with no remaining demand
        if state.farm_n_remaining[farm_idx] <= 0:
            continue
        
        # Check rain lock first (cheap check)
        if not check_rain_lock(state, farm_idx)[0]:
            continue
        
        # Find max valid shipment size
        n_remaining = state.farm_n_remaining[farm_idx]
        n_cap = n_remaining * (1 + NITROGEN_BUFFER)
        max_by_nitrogen = n_cap / NITROGEN_CONTENT  # kg
        
        max_by_storage = state.stp_storage[stp_idx]
        max_by_truck = max_tons * 1000  # kg
        
        max_allowed_kg = min(max_by_nitrogen, max_by_storage, max_by_truck)
        
        if max_allowed_kg > 1000:  # At least 1 ton
            valid_farms.append((farm_idx, max_allowed_kg / 1000))
    
    return valid_farms