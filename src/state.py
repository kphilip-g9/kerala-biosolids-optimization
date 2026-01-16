"""
System State Management for Biosolids Optimization
PATCHED VERSION - Handles regional weather data
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from datetime import datetime, timedelta

class SystemState:
    """Central state object that all components query."""
    
    def __init__(self, data: Dict):
        """Initialize state from loaded data."""
        self.config = data["config"]
        self.stp_registry = data["stp_registry"]
        self.farm_locations = data["farm_locations"]
        self.weather_df = data["daily_weather"]
        self.planting_schedule = data["planting_schedule"]
        self.daily_n_demand = data["daily_n_demand"]
        
        # Parse weather dates
        self.weather_df["date"] = pd.to_datetime(self.weather_df["date"])
        
        # Initialize dynamic state
        self.current_day = 1
        self.current_date = datetime(2025, 1, 1)
        
        # STP storage: start at 50% capacity
        # FIXED: Use actual column names from CSV
        n_stps = len(self.stp_registry)
        self.stp_storage = np.zeros(n_stps)
        for i, row in self.stp_registry.iterrows():
            # storage_max_tons is in TONS, convert to kg
            self.stp_storage[i] = row["storage_max_tons"] * 1000 * 0.5
        
        # Farm nitrogen demand: aggregate full year demand
        n_farms = len(self.farm_locations)
        self.farm_n_remaining = np.zeros(n_farms)
        self._initialize_farm_demands()
        
        # Precompute distance matrix
        self.distance_matrix = self._compute_distances()
        
        # Track actions for export
        self.action_history = []
        
        # Map farms to regions (CRITICAL FIX)
        self._map_farms_to_regions()
        
    def _map_farms_to_regions(self):
        """Map each farm to a weather region based on location."""
        # Kerala regions approximately:
        # Kuttanad: Southern backwaters (lat ~9.0-9.5)
        # Palakkad: Central plains (lat ~10.5-11.0)
        # Highlands: Eastern hills (lon ~76.5-77.5)
        # Coastal: Western coast (lon ~75.0-76.0)
        
        self.farm_regions = []
        for _, farm in self.farm_locations.iterrows():
            # FIXED: Use 'lat' and 'lon' column names
            lat = farm["lat"]
            lon = farm["lon"]
            
            # Priority order: check specific regions first
            if 9.0 <= lat <= 9.5:
                region = "Kuttanad"
            elif 10.5 <= lat <= 11.0 and lon < 76.5:
                region = "Palakkad"
            elif lon >= 76.5:
                region = "Highlands"
            else:
                region = "Coastal"
            
            self.farm_regions.append(region)
    
    def _initialize_farm_demands(self):
        """Set initial nitrogen demand per farm for entire year."""
        for _, row in self.daily_n_demand.iterrows():
            farm_id = row["farm_id"]
            farm_idx = self.farm_locations[
                self.farm_locations["farm_id"] == farm_id
            ].index[0]
            
            self.farm_n_remaining[farm_idx] += row["n_demand_kg"]
    
    def _compute_distances(self) -> np.ndarray:
        """Compute STP to Farm distances using Haversine formula."""
        n_stps = len(self.stp_registry)
        n_farms = len(self.farm_locations)
        distances = np.zeros((n_stps, n_farms))
        
        for i, stp in self.stp_registry.iterrows():
            for j, farm in self.farm_locations.iterrows():
                # FIXED: Use 'lat' and 'lon' column names
                distances[i, j] = self._haversine(
                    stp["lat"], stp["lon"],
                    farm["lat"], farm["lon"]
                )
        
        return distances
    
    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance between two points."""
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def get_5day_rainfall(self, farm_idx: int) -> float:
        """
        Get total rainfall forecast for next 5 days at farm's region.
        
        FIXED: Uses regional weather data
        """
        region = self.farm_regions[farm_idx]
        
        # Get next 5 days of weather
        end_date = self.current_date + timedelta(days=5)
        mask = (
            (self.weather_df["date"] >= self.current_date) &
            (self.weather_df["date"] < end_date)
        )
        
        region_weather = self.weather_df[mask]
        
        if len(region_weather) == 0:
            return 0.0
        
        # Sum rainfall for this region
        return region_weather[region].sum()
    
    def get_daily_supply(self, stp_id: str) -> float:
        """Get today's biosolid production for an STP."""
        stp_row = self.stp_registry[
            self.stp_registry["stp_id"] == stp_id
        ].iloc[0]
        
        # FIXED: daily_output_tons is in TONS, convert to kg
        return stp_row["daily_output_tons"] * 1000
    
    def apply_action(self, stp_idx: int, farm_idx: int, tons: float):
        """
        Apply a transport action and update state.
        
        FIXED: Now records day and date for export
        """
        kg = tons * 1000
        
        # Deduct from STP storage
        self.stp_storage[stp_idx] -= kg
        
        # Deduct nitrogen from farm demand (5% N content)
        n_content = kg * 0.05
        self.farm_n_remaining[farm_idx] -= n_content
        
        # Log action with ALL needed fields
        stp_id = self.stp_registry.iloc[stp_idx]["stp_id"]
        farm_id = self.farm_locations.iloc[farm_idx]["farm_id"]
        
        self.action_history.append({
            "date": self.current_date,
            "stp_id": stp_id,
            "farm_id": farm_id,
            "tons_delivered": tons  # Match submission column name
        })
    
    def advance_day(self):
        """Move to next day and update state."""
        # Add daily production to each STP
        for i, stp in self.stp_registry.iterrows():
            # FIXED: daily_output_tons is in TONS, convert to kg
            daily_production_kg = stp["daily_output_tons"] * 1000
            self.stp_storage[i] += daily_production_kg
            
            # Clip at max capacity
            max_cap = stp["storage_max_tons"] * 1000  # Convert to kg
            if self.stp_storage[i] > max_cap:
                overflow = self.stp_storage[i] - max_cap
                print(f"WARNING: STP {stp['stp_id']} overflowed by {overflow:.0f} kg on day {self.current_day}")
                self.stp_storage[i] = max_cap
        
        # Advance calendar
        self.current_day += 1
        self.current_date += timedelta(days=1)
    
    def get_state_summary(self) -> Dict:
        """Get human-readable state summary for debugging."""
        return {
            "day": self.current_day,
            "date": self.current_date.strftime("%Y-%m-%d"),
            "total_stp_storage_kg": self.stp_storage.sum(),
            "total_farm_demand_kg": self.farm_n_remaining.sum(),
            "actions_taken_today": len([a for a in self.action_history if a["date"] == self.current_date])
        }