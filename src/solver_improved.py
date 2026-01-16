"""
Improved allocation solver with rain-risk weighted urgency.
Person 4's contribution - does NOT modify existing files.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import math

class ImprovedSolver:
    """
    Rain-aware allocation solver that prioritizes farms based on:
    1. Days until rain lockout
    2. Nitrogen demand urgency
    3. Transport distance efficiency
    """
    
    def __init__(self, state, constraints, daily_n_demand, weather_data, farm_data, stp_data):
        self.state = state
        self.constraints = constraints
        self.daily_n_demand = daily_n_demand
        self.weather_data = weather_data
        self.farm_data = farm_data
        self.stp_data = stp_data
        
        # Precompute distance matrix (do this once!)
        self.distance_matrix = self._precompute_distances()
        
    def _precompute_distances(self) -> Dict[Tuple[str, str], float]:
        """Calculate all STP-to-Farm distances once."""
        distances = {}
        
        for _, stp in self.stp_data.iterrows():
            stp_id = stp['stp_id']
            stp_lat, stp_lon = stp['lat'], stp['lon']
            
            for _, farm in self.farm_data.iterrows():
                farm_id = farm['farm_id']
                farm_lat, farm_lon = farm['lat'], farm['lon']
                
                # Haversine distance
                dist = self._haversine(stp_lat, stp_lon, farm_lat, farm_lon)
                distances[(stp_id, farm_id)] = dist
                
        return distances
    
    def _haversine(self, lat1, lon1, lat2, lon2) -> float:
        """Calculate distance in km between two lat/lon points."""
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def _get_days_until_rain(self, farm_id: str, current_date: datetime) -> int:
        """
        Calculate how many days until this farm will be rain-locked.
        Returns days until next 5-day window with >30mm rain.
        """
        # Get farm's weather zone
        farm_zone = self.farm_data[self.farm_data['farm_id'] == farm_id]['zone'].values[0]
        
        # Map zone to weather column
        zone_map = {
            'Kuttanad': 'rain_kuttanad_mm',
            'Palakkad': 'rain_palakkad_mm',
            'Highlands': 'rain_highlands_mm',
            'Coastal': 'rain_coastal_mm'
        }
        rain_col = zone_map.get(farm_zone, 'rain_kuttanad_mm')
        
        # Check next 14 days for rain lockout
        for days_ahead in range(14):
            check_date = current_date + timedelta(days=days_ahead)
            
            # Get 5-day rainfall window
            rain_5day = 0
            for i in range(5):
                forecast_date = check_date + timedelta(days=i)
                if forecast_date.strftime('%Y-%m-%d') in self.weather_data.index:
                    rain_5day += self.weather_data.loc[forecast_date.strftime('%Y-%m-%d'), rain_col]
            
            # If this 5-day window exceeds threshold, rain lockout starts here
            if rain_5day > 30:
                return days_ahead
        
        return 14  # No rain lockout in next 2 weeks
    
    def _calculate_urgency_score(self, farm_id: str, stp_id: str, current_date: datetime) -> float:
        """
        Calculate priority score for this farm-STP pair.
        Higher score = higher priority.
        
        Formula: (nitrogen_need × rain_urgency_factor) / (distance + 1)
        """
        # Get nitrogen need for this farm today
        date_str = current_date.strftime('%Y-%m-%d')
        if date_str not in self.daily_n_demand.index:
            return 0.0
        
        n_need = self.daily_n_demand.loc[date_str, farm_id]
        if n_need <= 0:
            return 0.0
        
        # Get days until rain lockout
        days_until_rain = self._get_days_until_rain(farm_id, current_date)
        
        # Rain urgency factor (exponential decay)
        # If rain is 0 days away: factor = 10.0
        # If rain is 7 days away: factor = 1.5
        # If rain is 14+ days away: factor = 1.0
        rain_urgency = 1.0 + 9.0 * math.exp(-days_until_rain / 2.5)
        
        # Get distance (already precomputed)
        distance = self.distance_matrix.get((stp_id, farm_id), 100)
        
        # Combined score
        score = (n_need * rain_urgency) / (distance + 1)
        
        return score
    
    def allocate_day(self, current_date: datetime) -> List[Dict]:
        """
        Allocate biosolids for one day using rain-risk weighted urgency.
        
        Returns list of allocations: [{stp_id, farm_id, tons}, ...]
        """
        allocations = []
        date_str = current_date.strftime('%Y-%m-%d')
        
        # Step 1: Build priority queue of all feasible (STP, Farm) pairs
        priority_queue = []
        
        for _, stp in self.stp_data.iterrows():
            stp_id = stp['stp_id']
            available = self.state.get_stp_available(stp_id)
            
            if available <= 0:
                continue
            
            for _, farm in self.farm_data.iterrows():
                farm_id = farm['farm_id']
                
                # Check if farm can receive today (rain check)
                if not self.constraints.check_rain_constraint(farm_id, current_date):
                    continue
                
                # Calculate urgency score
                score = self._calculate_urgency_score(farm_id, stp_id, current_date)
                
                if score > 0:
                    priority_queue.append({
                        'stp_id': stp_id,
                        'farm_id': farm_id,
                        'score': score,
                        'distance': self.distance_matrix[(stp_id, farm_id)]
                    })
        
        # Step 2: Sort by score (highest first)
        priority_queue.sort(key=lambda x: x['score'], reverse=True)
        
        # Step 3: Allocate greedily from highest priority
        for item in priority_queue:
            stp_id = item['stp_id']
            farm_id = item['farm_id']
            
            # Get nitrogen need
            n_need = self.daily_n_demand.loc[date_str, farm_id]
            if n_need <= 0:
                continue
            
            # Calculate tons needed (nitrogen content = 2%)
            tons_for_n_need = n_need / 0.02
            
            # Check how much we can actually send
            available = self.state.get_stp_available(stp_id)
            capacity_remaining = self.state.get_farm_capacity_remaining(farm_id)
            
            tons_to_send = min(tons_for_n_need, available, capacity_remaining, 20)  # 20 ton truck limit
            
            if tons_to_send < 0.1:  # Minimum threshold
                continue
            
            # Validate with constraints
            if self.constraints.check_nitrogen_constraint(farm_id, tons_to_send, current_date):
                allocations.append({
                    'date': date_str,
                    'stp_id': stp_id,
                    'farm_id': farm_id,
                    'tons': round(tons_to_send, 2)
                })
                
                # Update state
                self.state.add_delivery(stp_id, farm_id, tons_to_send)
        
        return allocations
    
    def solve_full_year(self) -> pd.DataFrame:
        """
        Run allocation for all 365 days of 2025.
        Returns DataFrame in submission format.
        """
        all_allocations = []
        
        start_date = datetime(2025, 1, 1)
        
        for day in range(365):
            current_date = start_date + timedelta(days=day)
            
            # Daily production arrives at STPs
            self.state.add_daily_production()
            
            # Allocate using improved strategy
            daily_allocations = self.allocate_day(current_date)
            all_allocations.extend(daily_allocations)
            
            # Move to next day
            self.state.advance_day()
            
            # Progress indicator
            if day % 30 == 0:
                print(f"  Improved solver: Day {day}/365 complete")
        
        # Convert to DataFrame
        if not all_allocations:
            print("Warning: No allocations made!")
            return pd.DataFrame(columns=['date', 'stp_id', 'farm_id', 'tons'])
        
        result_df = pd.DataFrame(all_allocations)
        
        print(f"\n✅ Improved solver complete: {len(result_df)} allocations")
        return result_df


def run_improved_solver(state, constraints, daily_n_demand, weather_data, farm_data, stp_data):
    """
    Entry point for improved solver.
    Called from main.py as alternative to baseline.
    """
    print("\n🚀 Running IMPROVED solver (Rain-Risk Weighted Urgency)...")
    
    solver = ImprovedSolver(
        state=state,
        constraints=constraints,
        daily_n_demand=daily_n_demand,
        weather_data=weather_data,
        farm_data=farm_data,
        stp_data=stp_data
    )
    
    solution_df = solver.solve_full_year()
    
    return solution_df