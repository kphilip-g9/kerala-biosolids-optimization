"""
Improved allocation solver with rain-risk weighted urgency.
Aggressive clearing mode.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import math

class ImprovedSolver:
    def __init__(self, state):
        self.state = state
        self.daily_demand = state.daily_n_demand.copy()
        
        # CLEANING: Strip whitespace from columns to match State's cleaning
        self.daily_demand.columns = self.daily_demand.columns.str.strip()
        
        if 'date' in self.daily_demand.columns:
            self.daily_demand['date'] = pd.to_datetime(self.daily_demand['date'])
            self.daily_demand.set_index('date', inplace=True)
            
    def _calculate_urgency_score(self, farm_idx: int, stp_idx: int, current_date: datetime) -> float:
        # 1. Yearly Nitrogen Cap (Main Constraint)
        if self.state.farm_n_remaining[farm_idx] <= 0: return 0.0
            
        # 2. Distance Factor (Closer is better)
        dist = self.state.distance_matrix[stp_idx, farm_idx]
        score = 1000.0 / (dist + 1.0) 
        
        # 3. Rain Risk
        if self.state.get_5day_rainfall(farm_idx) > 30.0: return 0.0
            
        # 4. Daily Demand Bonus (Optional Boost)
        farm_id = str(self.state.farm_locations.iloc[farm_idx]['farm_id']).strip()
        
        if current_date in self.daily_demand.index and farm_id in self.daily_demand.columns:
            if self.daily_demand.loc[current_date, farm_id] > 0:
                score *= 2.0 # Boost if needed today
                
        return score

    def allocate_day(self):
        current_date = self.state.current_date
        
        # 1. Identify Opportunities
        opportunities = []
        
        for stp_idx in range(len(self.state.stp_registry)):
            available_kg = self.state.stp_storage[stp_idx]
            if available_kg < 5000: continue 
                
            for farm_idx in range(len(self.state.farm_locations)):
                if self.state.farm_n_remaining[farm_idx] <= 0: continue
                
                score = self._calculate_urgency_score(farm_idx, stp_idx, current_date)
                
                if score > 0:
                    opportunities.append({
                        'stp_idx': stp_idx, 'farm_idx': farm_idx,
                        'score': score
                    })
        
        # 2. Sort
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        # 3. Execute
        for opp in opportunities:
            stp_idx, farm_idx = opp['stp_idx'], opp['farm_idx']
            
            if self.state.stp_storage[stp_idx] < 2000: continue
            if self.state.farm_n_remaining[farm_idx] <= 0: continue
            
            # Send max possible
            max_truck = 20000.0
            farm_cap = (self.state.farm_n_remaining[farm_idx] * 1.10) / 0.05
            
            send_kg = min(self.state.stp_storage[stp_idx], max_truck, farm_cap)
            send_tons = send_kg / 1000.0
            
            if send_tons < 1.0: continue
                
            self.state.apply_action(stp_idx, farm_idx, send_tons)

def run_improved_solver(state):
    print("\n[ImprovedSolver] Starting 365-day simulation...")
    solver = ImprovedSolver(state)
    for day in range(365):
        solver.allocate_day()
        state.advance_day()
        if day % 50 == 0:
            print(f"  Day {day}/365 complete...")
    print("✓ Improved simulation complete.")