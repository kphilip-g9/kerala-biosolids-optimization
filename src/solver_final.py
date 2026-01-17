"""
FINAL SOLVER - EVEN DISTRIBUTION STRATEGY
Spreads deliveries across ALL farms to minimize transport distance

Author: Person 2 - Final Attempt
"""

import pandas as pd
import numpy as np
from datetime import datetime


class FinalSolver:
    """
    Final optimization attempt: Distribute evenly across ALL farms
    to minimize average transport distance.
    """
    
    def __init__(self, state):
        self.state = state
        self.daily_demand = state.daily_n_demand.copy()
        
        # Clean columns
        self.daily_demand.columns = self.daily_demand.columns.str.strip()
        
        if 'date' in self.daily_demand.columns:
            self.daily_demand['date'] = pd.to_datetime(self.daily_demand['date'])
            self.daily_demand.set_index('date', inplace=True)
        
        # Pre-compute rain blocks
        print("  Pre-computing rain blocks...")
        self.rain_blocked = self._precompute_rain_blocks()
        print("  ✓ Rain blocks cached")
        
        # Track farm usage to ensure even distribution
        self.farm_nitrogen_used = np.zeros(len(state.farm_locations))
        self.farm_delivery_count = np.zeros(len(state.farm_locations))
    
    def _precompute_rain_blocks(self):
        """Pre-compute rain-blocked days for all farms"""
        blocked = {}
        n_farms = len(self.state.farm_locations)
        
        original_day = self.state.current_day
        original_date = self.state.current_date
        
        for farm_idx in range(n_farms):
            blocked[farm_idx] = set()
            
            for day in range(1, 366):
                self.state.current_day = day
                self.state.current_date = datetime(2025, 1, 1) + pd.Timedelta(days=day-1)
                
                rainfall = self.state.get_5day_rainfall(farm_idx)
                if rainfall > 30.0:
                    blocked[farm_idx].add(day)
        
        self.state.current_day = original_day
        self.state.current_date = original_date
        
        return blocked
    
    def is_rain_blocked(self, farm_idx, day):
        """Fast rain check"""
        return day in self.rain_blocked.get(farm_idx, set())
    
    def _calculate_score(self, farm_idx, stp_idx, current_date):
        """
        Calculate allocation score with even distribution bonus
        """
        # Check nitrogen remaining
        if self.state.farm_n_remaining[farm_idx] <= 0:
            return 0.0
        
        # Distance score
        dist = self.state.distance_matrix[stp_idx, farm_idx]
        distance_score = 10000.0 / (dist + 1.0)
        
        # Rain check
        if self.is_rain_blocked(farm_idx, self.state.current_day):
            return 0.0
        
        # EVEN DISTRIBUTION BONUS
        # Farms that haven't received much get priority
        total_n_limit = self.state.farm_locations.iloc[farm_idx]['n_limit_with_buffer'] if 'n_limit_with_buffer' in self.state.farm_locations.columns else self.state.farm_n_remaining[farm_idx] * 1.10
        
        # Utilization ratio (0 = unused, 1 = full)
        if total_n_limit > 0:
            utilization = self.farm_nitrogen_used[farm_idx] / total_n_limit
        else:
            utilization = 1.0
        
        # Bonus for underutilized farms (prioritize spreading evenly)
        distribution_bonus = 2.0 if utilization < 0.1 else (1.5 if utilization < 0.5 else 1.0)
        
        score = distance_score * distribution_bonus
        
        # Daily demand bonus
        farm_id = str(self.state.farm_locations.iloc[farm_idx]['farm_id']).strip()
        if current_date in self.daily_demand.index and farm_id in self.daily_demand.columns:
            if self.daily_demand.loc[current_date, farm_id] > 0:
                score *= 1.5
        
        return score
    
    def allocate_day(self):
        """
        Daily allocation with even distribution strategy
        """
        current_date = self.state.current_date
        
        # Collect opportunities
        opportunities = []
        
        for stp_idx in range(len(self.state.stp_registry)):
            available_kg = self.state.stp_storage[stp_idx]
            if available_kg < 1000:  # Lower threshold to ship more
                continue
            
            for farm_idx in range(len(self.state.farm_locations)):
                if self.state.farm_n_remaining[farm_idx] <= 0:
                    continue
                
                score = self._calculate_score(farm_idx, stp_idx, current_date)
                
                if score > 0:
                    opportunities.append({
                        'stp_idx': stp_idx,
                        'farm_idx': farm_idx,
                        'score': score
                    })
        
        # Sort by score (prioritizes underutilized farms)
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        # Execute with SMALLER shipments to spread across more farms
        for opp in opportunities:
            stp_idx = opp['stp_idx']
            farm_idx = opp['farm_idx']
            
            if self.state.stp_storage[stp_idx] < 1000:
                continue
            if self.state.farm_n_remaining[farm_idx] <= 0:
                continue
            
            # SMALLER shipments to reach more farms
            max_truck = 15000.0  # Reduced from 20000 to spread more
            farm_cap = (self.state.farm_n_remaining[farm_idx] * 1.10) / 0.05
            
            # Limit per shipment to encourage multiple smaller deliveries
            max_per_shipment = min(5000.0, farm_cap / 3.0)  # Split into smaller loads
            
            send_kg = min(self.state.stp_storage[stp_idx], max_truck, farm_cap, max_per_shipment)
            send_tons = send_kg / 1000.0
            
            if send_tons >= 0.5:  # Lower minimum to allow smaller shipments
                self.state.apply_action(stp_idx, farm_idx, send_tons)
                
                # Track usage
                n_delivered = send_kg * 0.05
                self.farm_nitrogen_used[farm_idx] += n_delivered
                self.farm_delivery_count[farm_idx] += 1


def run_final_solver(state):
    """
    Run final solver with even distribution strategy
    """
    print("\n" + "="*70)
    print("FINAL SOLVER - EVEN DISTRIBUTION STRATEGY")
    print("="*70)
    
    solver = FinalSolver(state)
    
    print("\n  Starting 365-day simulation...")
    
    for day in range(365):
        solver.allocate_day()
        state.advance_day()
        
        if day % 50 == 0:
            print(f"    Day {day}/365 complete...")
    
    print(f"\n  ✓ Final solver complete!")
    print(f"  ✓ Generated {len(state.action_history)} actions")
    
    # Stats
    farms_used = (solver.farm_delivery_count > 0).sum()
    avg_deliveries = solver.farm_delivery_count[solver.farm_delivery_count > 0].mean() if farms_used > 0 else 0
    
    print(f"  ✓ Farms utilized: {farms_used}/250")
    print(f"  ✓ Avg deliveries per farm: {avg_deliveries:.1f}")
    print("="*70 + "\n")
    
    return state