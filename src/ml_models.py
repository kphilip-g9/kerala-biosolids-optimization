"""
ML-Enhanced Farm Priority Scoring
PATCHED VERSION - Simplified for speed
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple

class FarmPriorityModel:
    """Predicts priority score (0-100) for each farm-STP pair."""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=30,  # Reduced for speed
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        self.crop_encoder = LabelEncoder()
        self.is_trained = False
        
    def generate_synthetic_training_data(self, state) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data from expert heuristics."""
        X_list = []
        y_list = []
        
        # Get all crop types
        all_crops = ["Rice", "Coconut", "Rubber", "Banana", "Vegetables", "fallow"]
        self.crop_encoder.fit(all_crops)
        
        n_stps = len(state.stp_registry)
        n_farms = len(state.farm_locations)
        
        # Sample subset for speed (not all pairs)
        sample_size = min(5000, n_stps * n_farms)
        
        for _ in range(sample_size):
            stp_idx = np.random.randint(0, n_stps)
            farm_idx = np.random.randint(0, n_farms)
            
            features = self._extract_features(state, stp_idx, farm_idx)
            if features is not None:
                X_list.append(features)
                score = self._expert_score(state, stp_idx, farm_idx)
                y_list.append(score)
        
        return np.array(X_list), np.array(y_list)
    
    def _extract_features(self, state, stp_idx: int, farm_idx: int) -> List[float]:
        """Extract feature vector for STP-farm pair."""
        # Simplified features for speed
        
        # Feature 1: Crop type (use dummy value for now)
        crop_encoded = 0  # Simplified
        
        # Feature 2: Demand urgency
        n_remaining = state.farm_n_remaining[farm_idx]
        max_demand = state.farm_n_remaining.max()
        demand_ratio = n_remaining / max_demand if max_demand > 0 else 0
        
        # Feature 3: Days remaining
        days_left = (365 - state.current_day) / 365
        
        # Feature 4: Storage pressure at STP
        # FIXED: storage_max_tons is in TONS, convert to kg
        stp_capacity = state.stp_registry.iloc[stp_idx]["storage_max_tons"] * 1000
        storage_ratio = state.stp_storage[stp_idx] / stp_capacity
        
        # Feature 5: Distance efficiency
        distance = state.distance_matrix[stp_idx, farm_idx]
        distance_score = 1 / (1 + distance / 100)
        
        # Feature 6: Rain risk
        rainfall_5day = state.get_5day_rainfall(farm_idx)
        rain_risk = min(rainfall_5day / 30, 1.0)
        
        return [
            crop_encoded,
            demand_ratio,
            days_left,
            storage_ratio,
            distance_score,
            rain_risk
        ]
    
    def _expert_score(self, state, stp_idx: int, farm_idx: int) -> float:
        """Hand-crafted expert scoring function."""
        
        # Base score from demand
        n_remaining = state.farm_n_remaining[farm_idx]
        if n_remaining <= 0:
            return 0
        
        score = n_remaining / 1000
        
        # Distance penalty
        distance = state.distance_matrix[stp_idx, farm_idx]
        distance_penalty = np.exp(-distance / 50)
        score *= distance_penalty
        
        # Rain risk penalty
        rainfall_5day = state.get_5day_rainfall(farm_idx)
        if rainfall_5day > 30:
            return 0
        elif rainfall_5day > 20:
            score *= 0.5
        
        # STP storage urgency
        # FIXED: storage_max_tons is in TONS, convert to kg
        stp_capacity = state.stp_registry.iloc[stp_idx]["storage_max_tons"] * 1000
        storage_ratio = state.stp_storage[stp_idx] / stp_capacity
        if storage_ratio > 0.8:
            score *= 1.5
        
        # Time urgency
        if state.current_day > 300:
            score *= 1.3
        
        return max(0, min(100, score))
    
    def train(self, state):
        """Train model on synthetic expert data."""
        print("  Generating training data...")
        X, y = self.generate_synthetic_training_data(state)
        
        print(f"  Training on {len(X)} examples...")
        self.model.fit(X, y)
        self.is_trained = True
        
        print("  ✓ Model trained")
    
    def predict_priority(self, state, stp_idx: int, farm_idx: int) -> float:
        """Predict priority score for a farm."""
        if not self.is_trained:
            return self._expert_score(state, stp_idx, farm_idx)
        
        features = self._extract_features(state, stp_idx, farm_idx)
        if features is None:
            return 0
        
        try:
            score = self.model.predict([features])[0]
            return max(0, min(100, score))
        except:
            return self._expert_score(state, stp_idx, farm_idx)
    
    def rank_farms(self, state, stp_idx: int, valid_farms: List[int]) -> List[Tuple[int, float]]:
        """Rank list of valid farms by priority."""
        scored_farms = []
        for farm_idx in valid_farms:
            score = self.predict_priority(state, stp_idx, farm_idx)
            scored_farms.append((farm_idx, score))
        
        scored_farms.sort(key=lambda x: x[1], reverse=True)
        return scored_farms


class RainRiskSmoother:
    """Simple exponential smoothing on rainfall forecasts."""
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.smoothed_forecasts = {}
    
    def smooth_forecast(self, raw_forecast: float, location_id: str) -> float:
        """Apply exponential smoothing to rainfall forecast."""
        if location_id not in self.smoothed_forecasts:
            self.smoothed_forecasts[location_id] = raw_forecast
            return raw_forecast
        
        prev_smooth = self.smoothed_forecasts[location_id]
        new_smooth = self.alpha * raw_forecast + (1 - self.alpha) * prev_smooth
        self.smoothed_forecasts[location_id] = new_smooth
        
        return new_smooth