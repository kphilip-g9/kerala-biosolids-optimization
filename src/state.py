"""
System State Management
CORRECT FIX: Smarter Data Cleaning (No Hard-Coding)
"""

import numpy as np
import pandas as pd
from typing import Dict
from datetime import datetime, timedelta

class SystemState:
    """Central state object that all components query."""
    
    def __init__(self, data: Dict):
        """Initialize state from loaded data."""
        self.config = data.get("config", {})
        self.stp_registry = data["stp_registry"]
        self.farm_locations = data["farm_locations"]
        self.weather_df = data["daily_weather"]
        self.planting_schedule = data["planting_schedule"]
        self.daily_n_demand = data["daily_n_demand"]
        
        # 1. Parse dates
        self.weather_df["date"] = pd.to_datetime(self.weather_df["date"])
        
        # 2. Simulation Clock
        self.current_day = 1
        self.current_date = datetime(2025, 1, 1)
        
        # 3. Initialize STP Storage (Start at 50% capacity)
        self.stp_storage = self.stp_registry["storage_max_tons"].values * 1000 * 0.5
        
        # 4. Initialize Farm Demand (SMART MATCHING)
        self.farm_n_remaining = np.zeros(len(self.farm_locations))
        self._initialize_farm_demands()
        
        # 5. Precompute Distances
        self.distance_matrix = self._compute_distances()
        
        # 6. Map Regions
        self.farm_regions = self._map_farms_to_regions()
        
        # 7. Action Log
        self.action_history = []

    def _initialize_farm_demands(self):
        """
        Sum total yearly demand for each farm.
        Uses CLEANING, not hard-coding, to fix mismatches.
        """
        # 1. Create a map of {Cleaned_ID: Index}
        # e.g., "F_1001" -> 0
        farm_map = {}
        for idx, row in self.farm_locations.iterrows():
            raw_id = str(row['farm_id'])
            clean_id = raw_id.strip() # Remove spaces
            farm_map[clean_id] = idx
            
        matches = 0
        total_demand_found = 0
        
        # 2. Iterate columns in demand file
        for col in self.daily_n_demand.columns:
            if col == 'date':
                continue
                
            # Clean the column header (e.g. " F_1001 " -> "F_1001")
            clean_col = str(col).strip()
            
            if clean_col in farm_map:
                farm_idx = farm_map[clean_col]
                
                # Sum the column to get yearly demand
                yearly_sum = self.daily_n_demand[col].sum()
                self.farm_n_remaining[farm_idx] = yearly_sum
                total_demand_found += yearly_sum
                matches += 1
        
        print(f"  > [State] Successfully matched {matches} farms.")
        print(f"  > [State] Total Nitrogen Demand Found: {total_demand_found:,.0f} kg")
        
        if matches == 0:
            raise ValueError("CRITICAL: No farm columns matched! Check CSV headers manually.")

    def _map_farms_to_regions(self):
        regions = []
        for _, row in self.farm_locations.iterrows():
            lat, lon = row["lat"], row["lon"]
            if 9.0 <= lat <= 9.5: regions.append("Kuttanad")
            elif 10.5 <= lat <= 11.0 and lon < 76.5: regions.append("Palakkad")
            elif lon >= 76.5: regions.append("Highlands")
            else: regions.append("Coastal")
        return regions

    def _compute_distances(self) -> np.ndarray:
        n_stps = len(self.stp_registry)
        n_farms = len(self.farm_locations)
        dists = np.zeros((n_stps, n_farms))
        
        stp_coords = self.stp_registry[["lat", "lon"]].to_numpy()
        farm_coords = self.farm_locations[["lat", "lon"]].to_numpy()
        
        for i in range(n_stps):
            for j in range(n_farms):
                dists[i, j] = self._haversine(
                    stp_coords[i][0], stp_coords[i][1],
                    farm_coords[j][0], farm_coords[j][1]
                )
        return dists

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        R = 6371
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    def get_5day_rainfall(self, farm_idx: int) -> float:
        region = self.farm_regions[farm_idx]
        end_date = self.current_date + timedelta(days=5)
        mask = (self.weather_df["date"] >= self.current_date) & \
               (self.weather_df["date"] < end_date)
        return self.weather_df.loc[mask, region].sum()

    def apply_action(self, stp_idx: int, farm_idx: int, tons: float):
        kg = tons * 1000
        self.stp_storage[stp_idx] -= kg
        # Assume biosolid is 5% Nitrogen
        self.farm_n_remaining[farm_idx] -= (kg * 0.05)
        
        self.action_history.append({
            "date": self.current_date,
            "stp_id": self.stp_registry.iloc[stp_idx]["stp_id"],
            "farm_id": self.farm_locations.iloc[farm_idx]["farm_id"],
            "tons_delivered": tons
        })

    def advance_day(self):
        daily_prod = self.stp_registry["daily_output_tons"].values * 1000
        max_cap = self.stp_registry["storage_max_tons"].values * 1000
        self.stp_storage += daily_prod
        self.stp_storage = np.minimum(self.stp_storage, max_cap)
        self.current_date += timedelta(days=1)
        self.current_day += 1

    def get_state_summary(self):
        return {
            "date": self.current_date.strftime("%Y-%m-%d"),
            "total_stp_storage_kg": np.sum(self.stp_storage),
            "total_farm_demand_kg": np.sum(self.farm_n_remaining)
        }
        