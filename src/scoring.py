"""
Carbon Credit Scoring Module for Kerala Biosolids Optimization
Merged Version: Integrates Teammate's Constants + Official Data Logic
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class CarbonConstants:
    """Carbon credit/penalty constants from competition rules."""
    # Gains
    NITROGEN_OFFSET = 5.0       # kg CO₂ per kg N used (fertilizer replacement)
    SOIL_CARBON = 0.2           # kg CO₂ per kg biosolid applied
    
    # Penalties
    TRANSPORT = 0.9             # kg CO₂ per km truck travel
    EXCESS_NITROGEN = 10.0      # kg CO₂ per kg excess N (leaching)
    EMERGENCY_DUMP = 1000.0     # kg CO₂ per ton dumped
    
    # Operational
    TRUCK_CAPACITY_TONS = 20.0  # Max load per truck
    NITROGEN_CONTENT = 0.05     # 5% Nitrogen content in biosolids
    NITROGEN_BUFFER = 0.10      # 10% allowed buffer over demand

class CarbonScorer:
    """
    Calculates total net carbon credits using SystemState data.
    """
    
    def __init__(self, state):
        """
        Initialize with the system state to access ground-truth data.
        """
        self.state = state
        self.constants = CarbonConstants()
        
        # Create fast lookup maps
        self.farm_id_to_idx = {
            fid: i for i, fid in enumerate(state.farm_locations["farm_id"])
        }
        self.stp_id_to_idx = {
            sid: i for i, sid in enumerate(state.stp_registry["stp_id"])
        }
        
        # Pre-calculate Total Annual Demand per Farm (Ground Truth)
        # We sum the columns from the raw CSV data
        self.farm_total_demand_kg = np.zeros(len(state.farm_locations))
        self._calculate_total_demands()

    def _calculate_total_demands(self):
        """Sum the daily demand CSV to get hard annual limits."""
        demand_df = self.state.daily_n_demand
        
        for col in demand_df.columns:
            if col in self.farm_id_to_idx:
                idx = self.farm_id_to_idx[col]
                self.farm_total_demand_kg[idx] = demand_df[col].sum()

    def score_run(self, solution_df: pd.DataFrame) -> Dict:
        """
        Calculate total carbon score for a solution CSV.
        """
        print("\n[Scoring] Calculating Net Carbon Credit Score...")
        
        scores = {
            'nitrogen_offset': 0.0,
            'soil_carbon': 0.0,
            'transport_penalty': 0.0,
            'excess_nitrogen_penalty': 0.0,
            'dump_penalty': 0.0,
            'total': 0.0,
            'total_tons_delivered': 0.0
        }
        
        # Track nitrogen received per farm
        farm_received_n_kg = np.zeros(len(self.state.farm_locations))
        
        # 1. Process Deliveries (Iterate rows)
        # Note: Vectorizing this would be faster, but loop is safer for logic verification
        for _, row in solution_df.iterrows():
            stp_id = row['stp_id']
            farm_id = row['farm_id']
            tons = row['tons_delivered']
            
            if tons <= 0:
                continue
                
            scores['total_tons_delivered'] += tons
            
            # Identify indices
            if stp_id not in self.stp_id_to_idx or farm_id not in self.farm_id_to_idx:
                continue # Skip invalid IDs (should not happen in valid solution)
                
            s_idx = self.stp_id_to_idx[stp_id]
            f_idx = self.farm_id_to_idx[farm_id]
            
            # --- Transport Penalty ---
            # Distance from State Matrix
            dist_km = self.state.distance_matrix[s_idx, f_idx]
            
            # Number of Trucks (Ceiling division)
            num_trucks = np.ceil(tons / self.constants.TRUCK_CAPACITY_TONS)
            
            # Round trip distance * Trucks * Penalty
            trip_distance = dist_km * 2.0 
            scores['transport_penalty'] += (trip_distance * num_trucks * self.constants.TRANSPORT)
            
            # --- Soil Carbon Gain ---
            kg_product = tons * 1000.0
            scores['soil_carbon'] += (kg_product * self.constants.SOIL_CARBON)
            
            # --- Nitrogen Tracking ---
            n_content = kg_product * self.constants.NITROGEN_CONTENT
            farm_received_n_kg[f_idx] += n_content
            
            # --- Nitrogen Offset Gain ---
            # We credit ALL N applied here, then penalize excess later
            scores['nitrogen_offset'] += (n_content * self.constants.NITROGEN_OFFSET)

        # 2. Calculate Excess Nitrogen Penalty
        for f_idx in range(len(self.state.farm_locations)):
            received = farm_received_n_kg[f_idx]
            demand = self.farm_total_demand_kg[f_idx]
            
            # Cap = Demand + Buffer
            limit = demand * (1 + self.constants.NITROGEN_BUFFER)
            
            if received > limit:
                excess_kg = received - limit
                penalty = excess_kg * self.constants.EXCESS_NITROGEN
                scores['excess_nitrogen_penalty'] += penalty
                
                # OPTIONAL: Does competition void the GAIN for excess? 
                # Usually standard practice is Penalty ONLY, so we keep the gain.
                # If rules say "Waste", we might subtract the offset gain for the excess part.
                # For now, following strict penalty rules:
        
        # 3. Calculate Dump Penalty (If any recorded in state)
        # Note: If solution_df has dump rows, handle them. 
        # Currently assuming 'tons_delivered' implies valid delivery.
        
        # 4. Total Net Score
        # Gains are Positive, Penalties are Negative
        scores['total'] = (
            scores['nitrogen_offset'] +
            scores['soil_carbon'] - 
            scores['transport_penalty'] - 
            scores['excess_nitrogen_penalty'] -
            scores['dump_penalty']
        )
        
        self._print_report(scores)
        return scores

    def _print_report(self, scores: Dict):
        """Pretty print using the teammate's layout."""
        print("\n" + "="*60)
        print("CARBON CREDIT SCORE REPORT (OFFICIAL)")
        print("="*60)
        
        print("\n🌱 GAINS:")
        print(f"  Nitrogen Offset:           +{scores['nitrogen_offset']/1e6:,.2f} M kg CO₂")
        print(f"  Soil Sequestration:        +{scores['soil_carbon']/1e6:,.2f} M kg CO₂")
        
        print("\n🚛 PENALTIES:")
        print(f"  Transport Emissions:       -{scores['transport_penalty']/1e6:,.2f} M kg CO₂")
        print(f"  Excess N (Leaching):       -{scores['excess_nitrogen_penalty']/1e6:,.2f} M kg CO₂")
        
        print("-" * 60)
        print(f"NET SCORE:                   {scores['total']/1e6:,.3f} MILLION kg CO₂")
        print("=" * 60 + "\n")