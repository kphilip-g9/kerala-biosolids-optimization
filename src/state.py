"""
System State Management
FIXED: Handles Wide-Format Data + Whitespace stripping
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
        
        # CLEANUP: Strip whitespace from column headers to match Farm IDs
        self.daily_n_demand.columns = self.daily_n_demand.columns.str.strip()
        self.farm_locations["farm_id"] = self.farm_locations["farm_id"].str.strip()
        
        # 1. Parse dates
        self.weather_df["date"] = pd.to_datetime(self.weather_df["date"])
        
        # 2. Simulation Clock
        self.current_day = 1
        self.current_date = datetime(2025, 1, 1)
        
        # 3. Initialize STP Storage (Start at 50% capacity)
        self.stp_storage = self.stp_registry["storage_max_tons"].values * 1000 * 0.5
        
        # 4. Initialize Farm Demand
        self.farm_n_remaining = np.zeros(len(self.farm_locations))
        self._initialize_farm_demands()
        
        # 5. Precompute Distances
        self.distance_matrix = self._compute_distances()
        
        # 6. Map Regions
        self.farm_regions = self._map_farms_to_regions()
        
        # 7. Action Log
        self.action_history = []

    def _initialize_farm_demands(self):
        """Sum total yearly demand for each farm from columns."""
        farm_id_to_idx = {fid: i for i, fid in enumerate(self.farm_locations["farm_id"])}
        
        for col_name in self.daily_n_demand.columns:
            if col_name == "date": continue
            
            if col_name in farm_id_to_idx:
                idx = farm_id_to_idx[col_name]
                total_kg = self.daily_n_demand[col_name].sum()
                self.farm_n_remaining[idx] = total_kg

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