"""
ML-Enhanced Greedy Solver
PATCHED VERSION - Direct submission generation
"""

import numpy as np
from typing import List, Dict
from src.constraints import filter_valid_farms
from src.ml_models import FarmPriorityModel, RainRiskSmoother

class GreedySolver:
    """Greedy allocation policy enhanced with ML priority scoring."""
    
    def __init__(self, priority_model: FarmPriorityModel, rain_smoother: RainRiskSmoother):
        self.priority_model = priority_model
        self.rain_smoother = rain_smoother
        
    def solve_day(self, state) -> List[Dict]:
        """Plan allocations for current day across all STPs."""
        actions = []
        
        for stp_idx in range(len(state.stp_registry)):
            stp_actions = self._solve_stp(state, stp_idx)
            actions.extend(stp_actions)
        
        return actions
    
    def _solve_stp(self, state, stp_idx: int) -> List[Dict]:
        """Allocate biosolids from one STP for today."""
        actions = []
        
        # Available biosolids (leave buffer)
        available_kg = state.stp_storage[stp_idx] * 0.85
        
        if available_kg < 1000:
            return actions
        
        # Get valid farms
        valid_farms = filter_valid_farms(state, stp_idx)
        
        if not valid_farms:
            return actions
        
        # Rank farms by ML priority
        farm_indices = [f[0] for f in valid_farms]
        ranked_farms = self.priority_model.rank_farms(state, stp_idx, farm_indices)
        
        # Greedy allocation
        remaining_kg = available_kg
        
        for farm_idx, priority_score in ranked_farms:
            if remaining_kg < 1000:
                break
            
            # Find max allowed for this farm
            max_allowed_tons = next(
                f[1] for f in valid_farms if f[0] == farm_idx
            )
            
            # Allocate
            allocate_tons = min(
                max_allowed_tons,
                remaining_kg / 1000,
                10.0  # Cap at 10 tons per shipment for efficiency
            )
            
            if allocate_tons < 0.5:
                continue
            
            # Apply action (this records in state.action_history)
            state.apply_action(stp_idx, farm_idx, allocate_tons)
            remaining_kg -= allocate_tons * 1000
        
        return []  # State records actions internally now


def solve_full_year(state, priority_model: FarmPriorityModel):
    """
    Run greedy solver for entire year (365 days).
    
    FIXED: Returns nothing - state.action_history has everything
    """
    rain_smoother = RainRiskSmoother(alpha=0.3)
    solver = GreedySolver(priority_model, rain_smoother)
    
    print("\n  Solving 365 days...")
    
    for day in range(1, 366):
        # Solve today
        solver.solve_day(state)
        
        # Advance to next day
        state.advance_day()
        
        # Progress
        if day % 73 == 0:  # Every ~10 weeks
            summary = state.get_state_summary()
            print(f"    Day {day}/365: Storage={summary['total_stp_storage_kg']/1e6:.1f}M kg, "
                  f"Demand={summary['total_farm_demand_kg']/1e6:.1f}M kg")
    
    print(f"  ✓ Generated {len(state.action_history)} actions")