"""
ADVANCED CONSTRAINT MODULE - PERSON 2
Multi-day lookahead and caching for Kerala Biosolids Optimization

This module EXTENDS the basic constraints.py without replacing it.
Safe to add - no conflicts with existing code.

Author: PERSON 2 - Constraint Engine Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pickle
from pathlib import Path


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class ApplicationTiming(Enum):
    """Recommended timing for biosolid application"""
    APPLY_NOW = "apply_now"      # Urgent: safe now, rain coming in 1-3 days
    APPLY_SOON = "apply_soon"    # Comfortable: safe for next 3+ days
    CRITICAL = "critical"         # Last chance before long rain period
    WAIT = "wait"                 # Rain now, better window ahead
    AVOID = "avoid"               # No good windows in forecast


@dataclass
class SafeWindow:
    """Represents a continuous safe application window"""
    start_day: int
    end_day: int
    duration_days: int
    avg_rainfall: float
    risk_score: float
    
    def contains(self, day: int) -> bool:
        """Check if day falls within this window"""
        return self.start_day <= day <= self.end_day
    
    def days_until(self, from_day: int) -> int:
        """Days from from_day until window starts"""
        return max(0, self.start_day - from_day)
    
    def days_remaining(self, from_day: int) -> int:
        """Days remaining in window from from_day"""
        if from_day > self.end_day:
            return 0
        if from_day < self.start_day:
            return self.duration_days
        return self.end_day - from_day + 1


@dataclass
class FarmLookaheadAnalysis:
    """Complete lookahead analysis for a farm on a specific day"""
    farm_idx: int
    analysis_day: int
    
    # Current state
    is_currently_safe: bool = False
    current_rainfall_5day: float = 0.0
    current_risk_score: float = 0.0
    
    # Forecast windows
    safe_windows: List[SafeWindow] = field(default_factory=list)
    current_window: Optional[SafeWindow] = None
    next_safe_window: Optional[SafeWindow] = None
    
    # Recommendations
    timing_recommendation: ApplicationTiming = ApplicationTiming.AVOID
    urgency_score: float = 0.0  # 0.0 = no urgency, 1.0 = critical
    
    # Statistics
    safe_days_ahead_7: int = 0
    safe_days_ahead_14: int = 0
    days_until_next_rain: int = 0
    longest_window_ahead: int = 0


# ============================================================================
# CONSTRAINT CACHE
# ============================================================================

class ConstraintCache:
    """
    Pre-computed constraint lookups for entire year.
    Enables O(1) checks instead of recalculating each time.
    
    Integration with existing code:
    - Works with your SystemState class
    - Compatible with existing constraints.py
    - No modifications to existing code needed
    """
    
    RAIN_THRESHOLD = 30.0  # mm over 5 days (matches constraints.py)
    
    def __init__(self, state):
        """
        Build cache from SystemState instance
        
        Args:
            state: Your existing SystemState object
        """
        self.state = state
        self.n_farms = len(state.farm_locations)
        self.n_stps = len(state.stp_registry)
        
        # Cache structures
        self.farm_safe_days: Dict[int, Set[int]] = {}  # farm_idx -> set of safe days
        self.farm_blocked_days: Dict[int, Set[int]] = {}  # farm_idx -> set of blocked days
        self.farm_risk_scores: Dict[int, Dict[int, float]] = {}  # farm_idx -> day -> risk
        
        self.is_built = False
        
    def build_cache(self, verbose: bool = True):
        """Build cache for all farms across entire year"""
        if verbose:
            print("🔨 Building Constraint Cache...")
        
        # Save current state day
        original_day = self.state.current_day
        original_date = self.state.current_date
        
        # Scan entire year
        for day in range(1, 366):
            # Set state to this day (temporarily)
            self.state.current_day = day
            self.state.current_date = datetime(2025, 1, 1) + timedelta(days=day - 1)
            
            # Check each farm
            for farm_idx in range(self.n_farms):
                rainfall = self.state.get_5day_rainfall(farm_idx)
                is_safe = rainfall <= self.RAIN_THRESHOLD
                risk = min(1.0, rainfall / 50.0)
                
                # Initialize dicts if needed
                if farm_idx not in self.farm_safe_days:
                    self.farm_safe_days[farm_idx] = set()
                    self.farm_blocked_days[farm_idx] = set()
                    self.farm_risk_scores[farm_idx] = {}
                
                # Cache results
                if is_safe:
                    self.farm_safe_days[farm_idx].add(day)
                else:
                    self.farm_blocked_days[farm_idx].add(day)
                
                self.farm_risk_scores[farm_idx][day] = risk
        
        # Restore original state
        self.state.current_day = original_day
        self.state.current_date = original_date
        
        self.is_built = True
        
        if verbose:
            self._print_summary()
    
    def is_farm_safe(self, farm_idx: int, day: int) -> bool:
        """O(1) lookup: Is farm safe (not rain-locked) on this day?"""
        if not self.is_built:
            raise RuntimeError("Cache not built! Call build_cache() first")
        
        if farm_idx not in self.farm_safe_days:
            return True  # Unknown farm = assume safe
        
        return day in self.farm_safe_days[farm_idx]
    
    def get_farm_risk(self, farm_idx: int, day: int) -> float:
        """Get rain risk score (0.0 = safe, 1.0 = high risk)"""
        if not self.is_built:
            raise RuntimeError("Cache not built!")
        
        if farm_idx not in self.farm_risk_scores:
            return 0.5
        
        return self.farm_risk_scores[farm_idx].get(day, 0.5)
    
    def get_safe_farms_for_day(self, day: int) -> List[int]:
        """Get all farm indices that are safe on this day"""
        if not self.is_built:
            raise RuntimeError("Cache not built!")
        
        safe_farms = []
        for farm_idx in range(self.n_farms):
            if self.is_farm_safe(farm_idx, day):
                safe_farms.append(farm_idx)
        
        return safe_farms
    
    def get_farm_statistics(self, farm_idx: int) -> Dict:
        """Get yearly statistics for a farm"""
        if not self.is_built:
            raise RuntimeError("Cache not built!")
        
        safe_days = len(self.farm_safe_days.get(farm_idx, set()))
        blocked_days = len(self.farm_blocked_days.get(farm_idx, set()))
        
        return {
            'farm_idx': farm_idx,
            'safe_days': safe_days,
            'blocked_days': blocked_days,
            'safe_percentage': (safe_days / 365) * 100
        }
    
    def _print_summary(self):
        """Print cache statistics"""
        avg_safe = np.mean([len(days) for days in self.farm_safe_days.values()])
        avg_blocked = np.mean([len(days) for days in self.farm_blocked_days.values()])
        
        print(f"✓ Cached {self.n_farms} farms")
        print(f"  Average safe days per farm: {avg_safe:.1f} ({avg_safe/365*100:.1f}%)")
        print(f"  Average blocked days: {avg_blocked:.1f} ({avg_blocked/365*100:.1f}%)")
    
    def save(self, filepath: str = "outputs/constraint_cache.pkl"):
        """Save cache to disk"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'farm_safe_days': self.farm_safe_days,
                'farm_blocked_days': self.farm_blocked_days,
                'farm_risk_scores': self.farm_risk_scores,
                'n_farms': self.n_farms,
                'n_stps': self.n_stps
            }, f)
        print(f"💾 Cache saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, state):
        """Load cache from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        cache = cls(state)
        cache.farm_safe_days = data['farm_safe_days']
        cache.farm_blocked_days = data['farm_blocked_days']
        cache.farm_risk_scores = data['farm_risk_scores']
        cache.n_farms = data['n_farms']
        cache.n_stps = data['n_stps']
        cache.is_built = True
        
        print(f"📂 Cache loaded from {filepath}")
        return cache


# ============================================================================
# LOOKAHEAD ANALYZER
# ============================================================================

class LookaheadAnalyzer:
    """
    Multi-day forecast analysis for strategic planning.
    
    Integration:
    - Works with your SystemState
    - Can use ConstraintCache for speed
    - Returns priority rankings for solver
    """
    
    RAIN_THRESHOLD = 30.0
    
    def __init__(self, state, cache: Optional[ConstraintCache] = None):
        """
        Initialize analyzer
        
        Args:
            state: Your SystemState instance
            cache: Optional ConstraintCache for faster lookups
        """
        self.state = state
        self.cache = cache
    
    def analyze_farm(
        self, 
        farm_idx: int, 
        current_day: int,
        lookahead_days: int = 14
    ) -> FarmLookaheadAnalysis:
        """
        Perform complete lookahead analysis for a farm.
        
        Args:
            farm_idx: Farm index
            current_day: Current day of year (1-365)
            lookahead_days: How many days to look ahead
        
        Returns:
            FarmLookaheadAnalysis with recommendations
        """
        analysis = FarmLookaheadAnalysis(
            farm_idx=farm_idx,
            analysis_day=current_day
        )
        
        # Get current state (use cache if available)
        if self.cache and self.cache.is_built:
            analysis.is_currently_safe = self.cache.is_farm_safe(farm_idx, current_day)
            analysis.current_risk_score = self.cache.get_farm_risk(farm_idx, current_day)
        else:
            # Fallback to state
            original_day = self.state.current_day
            self.state.current_day = current_day
            rainfall = self.state.get_5day_rainfall(farm_idx)
            analysis.current_rainfall_5day = rainfall
            analysis.is_currently_safe = rainfall <= self.RAIN_THRESHOLD
            analysis.current_risk_score = min(1.0, rainfall / 50.0)
            self.state.current_day = original_day
        
        # Find safe windows in lookahead period
        analysis.safe_windows = self._find_safe_windows(
            farm_idx, current_day, lookahead_days
        )
        
        # Identify current and next windows
        for window in analysis.safe_windows:
            if window.contains(current_day):
                analysis.current_window = window
            elif window.start_day > current_day and analysis.next_safe_window is None:
                analysis.next_safe_window = window
        
        # Calculate statistics
        analysis.safe_days_ahead_7 = self._count_safe_days(farm_idx, current_day, 7)
        analysis.safe_days_ahead_14 = self._count_safe_days(farm_idx, current_day, 14)
        
        if analysis.safe_windows:
            analysis.longest_window_ahead = max(w.duration_days for w in analysis.safe_windows)
        
        # Generate recommendation
        self._generate_recommendation(analysis)
        
        return analysis
    
    def _find_safe_windows(
        self, 
        farm_idx: int, 
        start_day: int, 
        lookahead_days: int
    ) -> List[SafeWindow]:
        """Find all safe application windows in lookahead period"""
        windows = []
        current_window_start = None
        current_window_days = []
        
        for offset in range(lookahead_days + 1):
            day = start_day + offset
            if day > 365:
                break
            
            # Check if day is safe
            if self.cache and self.cache.is_built:
                is_safe = self.cache.is_farm_safe(farm_idx, day)
            else:
                # Fallback: check via state
                original_day = self.state.current_day
                self.state.current_day = day
                rainfall = self.state.get_5day_rainfall(farm_idx)
                is_safe = rainfall <= self.RAIN_THRESHOLD
                self.state.current_day = original_day
            
            if is_safe:
                if current_window_start is None:
                    current_window_start = day
                current_window_days.append(day)
            else:
                # End window if exists
                if current_window_start is not None:
                    windows.append(SafeWindow(
                        start_day=current_window_start,
                        end_day=current_window_days[-1],
                        duration_days=len(current_window_days),
                        avg_rainfall=0.0,  # Can be computed if needed
                        risk_score=0.3
                    ))
                    current_window_start = None
                    current_window_days = []
        
        # Close final window
        if current_window_start is not None:
            windows.append(SafeWindow(
                start_day=current_window_start,
                end_day=current_window_days[-1],
                duration_days=len(current_window_days),
                avg_rainfall=0.0,
                risk_score=0.3
            ))
        
        return windows
    
    def _count_safe_days(self, farm_idx: int, start_day: int, days: int) -> int:
        """Count safe days in next N days"""
        count = 0
        for offset in range(days):
            day = start_day + offset
            if day > 365:
                break
            
            if self.cache and self.cache.is_built:
                if self.cache.is_farm_safe(farm_idx, day):
                    count += 1
            else:
                original_day = self.state.current_day
                self.state.current_day = day
                rainfall = self.state.get_5day_rainfall(farm_idx)
                if rainfall <= self.RAIN_THRESHOLD:
                    count += 1
                self.state.current_day = original_day
        
        return count
    
    def _generate_recommendation(self, analysis: FarmLookaheadAnalysis):
        """Generate timing recommendation and urgency score"""
        
        # Case 1: Currently blocked
        if not analysis.is_currently_safe:
            if analysis.next_safe_window and analysis.next_safe_window.days_until(analysis.analysis_day) <= 5:
                analysis.timing_recommendation = ApplicationTiming.WAIT
                analysis.urgency_score = 0.3
            else:
                analysis.timing_recommendation = ApplicationTiming.AVOID
                analysis.urgency_score = 0.0
            return
        
        # Case 2: Currently safe
        if analysis.current_window:
            days_left = analysis.current_window.days_remaining(analysis.analysis_day)
            
            # Critical: Last day before rain
            if days_left == 1:
                if not analysis.next_safe_window or analysis.next_safe_window.days_until(analysis.analysis_day) > 7:
                    analysis.timing_recommendation = ApplicationTiming.CRITICAL
                    analysis.urgency_score = 1.0
                    return
            
            # Apply now: Rain coming soon
            if days_left <= 3:
                analysis.timing_recommendation = ApplicationTiming.APPLY_NOW
                analysis.urgency_score = 0.8
                return
            
            # Apply soon: Comfortable window
            analysis.timing_recommendation = ApplicationTiming.APPLY_SOON
            analysis.urgency_score = 0.5
        else:
            # Safe today but edge case
            analysis.timing_recommendation = ApplicationTiming.APPLY_NOW
            analysis.urgency_score = 0.6
    
    def get_priority_farms(
        self, 
        current_day: int,
        valid_farm_indices: List[int],
        top_n: int = 50
    ) -> List[Tuple[int, FarmLookaheadAnalysis]]:
        """
        Get priority-ranked farms for application.
        
        Args:
            current_day: Current day
            valid_farm_indices: List of valid farm indices (from constraints.filter_valid_farms)
            top_n: Number to return
        
        Returns:
            List of (farm_idx, analysis) sorted by urgency
        """
        analyses = []
        
        for farm_idx in valid_farm_indices:
            analysis = self.analyze_farm(farm_idx, current_day, lookahead_days=14)
            if analysis.is_currently_safe:  # Only safe farms
                analyses.append((farm_idx, analysis))
        
        # Sort by urgency
        analyses.sort(key=lambda x: x[1].urgency_score, reverse=True)
        
        return analyses[:top_n]


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def enhance_solver_with_lookahead(
    state,
    stp_idx: int,
    valid_farms: List[Tuple[int, float]],
    lookahead: LookaheadAnalyzer
) -> List[Tuple[int, float, float]]:
    """
    Helper function to integrate lookahead into your solver.
    
    Args:
        state: Your SystemState
        stp_idx: STP index
        valid_farms: Output from constraints.filter_valid_farms()
        lookahead: LookaheadAnalyzer instance
    
    Returns:
        List of (farm_idx, max_tons, urgency_score) sorted by urgency
    """
    current_day = state.current_day
    farm_indices = [f[0] for f in valid_farms]
    farm_max_tons = {f[0]: f[1] for f in valid_farms}
    
    # Get priority ranking
    priority_farms = lookahead.get_priority_farms(
        current_day, 
        farm_indices,
        top_n=len(farm_indices)
    )
    
    # Return with max tons and urgency
    result = []
    for farm_idx, analysis in priority_farms:
        result.append((
            farm_idx,
            farm_max_tons[farm_idx],
            analysis.urgency_score
        ))
    
    return result


# ============================================================================
# STANDALONE USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("ADVANCED CONSTRAINTS MODULE - PERSON 2")
    print("=" * 60)
    print("\nThis module provides:")
    print("  ✓ Constraint caching for O(1) lookups")
    print("  ✓ Multi-day lookahead analysis")
    print("  ✓ Urgency-based priority ranking")
    print("\nIntegration:")
    print("  1. Build cache once: cache.build_cache(state)")
    print("  2. Use in solver: lookahead.get_priority_farms()")
    print("  3. Compatible with existing constraints.py")
    print("\nSee integration example in main solver code.")