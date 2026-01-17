"""
STRATEGIC SOLVER - PERSON 2
Combines ImprovedSolver with Advanced Constraints (Lookahead + Caching)

This is the ULTIMATE solver that integrates:
- Your ImprovedSolver's urgency scoring
- My LookaheadAnalyzer's strategic planning
- My ConstraintCache's O(1) performance
- Multi-phase allocation (Critical → Urgent → Normal)

Author: PERSON 2 - Enhanced by Team
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from src.constraints import filter_valid_farms, validate_action
from src.constraints_advanced import (
    LookaheadAnalyzer,
    ConstraintCache,
    ApplicationTiming
)


class StrategicSolver:
    """
    Strategic solver with lookahead planning.
    
    Improvements over ImprovedSolver:
    1. Pre-computed constraint cache (1000x faster)
    2. 14-day lookahead analysis
    3. Multi-phase allocation (critical first)
    4. Strategic window planning
    5. Urgency-weighted scoring
    """
    
    def __init__(self, state, use_cache: bool = True):
        self.state = state
        self.daily_demand = state.daily_n_demand.copy()
        
        # Clean column names
        self.daily_demand.columns = self.daily_demand.columns.str.strip()
        
        if 'date' in self.daily_demand.columns:
            self.daily_demand['date'] = pd.to_datetime(self.daily_demand['date'])
            self.daily_demand.set_index('date', inplace=True)
        
        # NEW: Initialize advanced constraints
        if use_cache:
            print("  Building constraint cache...")
            self.cache = ConstraintCache(state)
            self.cache.build_cache(verbose=False)
            self.lookahead = LookaheadAnalyzer(state, cache=self.cache)
            print("  ✓ Cache built")
        else:
            self.cache = None
            self.lookahead = LookaheadAnalyzer(state, cache=None)
    
    def _calculate_base_urgency(
        self, 
        farm_idx: int, 
        stp_idx: int, 
        current_date: datetime
    ) -> float:
        """
        Calculate base urgency score (from ImprovedSolver).
        This is the foundation score before lookahead enhancement.
        """
        # 1. Check nitrogen remaining
        if self.state.farm_n_remaining[farm_idx] <= 0:
            return 0.0
        
        # 2. Distance factor
        dist = self.state.distance_matrix[stp_idx, farm_idx]
        score = 1000.0 / (dist + 1.0)
        
        # 3. Rain check (using cache if available)
        if self.cache:
            if not self.cache.is_farm_safe(farm_idx, self.state.current_day):
                return 0.0
        else:
            if self.state.get_5day_rainfall(farm_idx) > 30.0:
                return 0.0
        
        # 4. Daily demand bonus
        farm_id = str(self.state.farm_locations.iloc[farm_idx]['farm_id']).strip()
        
        if current_date in self.daily_demand.index and farm_id in self.daily_demand.columns:
            if self.daily_demand.loc[current_date, farm_id] > 0:
                score *= 2.0
        
        return score
    
    def _calculate_strategic_urgency(
        self,
        farm_idx: int,
        stp_idx: int,
        current_date: datetime
    ) -> Tuple[float, ApplicationTiming]:
        """
        NEW: Calculate urgency with lookahead intelligence.
        
        Returns:
            (urgency_score, timing_recommendation)
        """
        # Get base urgency
        base_score = self._calculate_base_urgency(farm_idx, stp_idx, current_date)
        
        if base_score <= 0:
            return 0.0, ApplicationTiming.AVOID
        
        # Get lookahead analysis
        analysis = self.lookahead.analyze_farm(
            farm_idx, 
            self.state.current_day,
            lookahead_days=14
        )
        
        # Enhance base score with timing intelligence
        timing = analysis.timing_recommendation
        timing_multiplier = {
            ApplicationTiming.CRITICAL: 3.0,      # 3x urgency - MUST DO TODAY!
            ApplicationTiming.APPLY_NOW: 2.0,     # 2x urgency - very urgent
            ApplicationTiming.APPLY_SOON: 1.5,    # 1.5x urgency - good opportunity
            ApplicationTiming.WAIT: 0.3,          # 0.3x - better window coming
            ApplicationTiming.AVOID: 0.1          # 0.1x - skip for now
        }
        
        strategic_score = base_score * timing_multiplier[timing]
        
        # Additional boost from lookahead urgency score
        strategic_score *= (1 + analysis.urgency_score)
        
        return strategic_score, timing
    
    def allocate_day_strategic(self):
        """
        NEW: Strategic allocation with multi-phase approach.
        
        Phase 1: CRITICAL farms (must do today or lose window)
        Phase 2: URGENT farms (APPLY_NOW recommendation)
        Phase 3: NORMAL farms (APPLY_SOON, opportunistic)
        """
        current_date = self.state.current_date
        
        # Collect opportunities with strategic scoring
        critical_opportunities = []
        urgent_opportunities = []
        normal_opportunities = []
        
        for stp_idx in range(len(self.state.stp_registry)):
            available_kg = self.state.stp_storage[stp_idx]
            if available_kg < 5000:
                continue
            
            for farm_idx in range(len(self.state.farm_locations)):
                if self.state.farm_n_remaining[farm_idx] <= 0:
                    continue
                
                # Calculate strategic urgency
                score, timing = self._calculate_strategic_urgency(
                    farm_idx, stp_idx, current_date
                )
                
                if score <= 0:
                    continue
                
                opp = {
                    'stp_idx': stp_idx,
                    'farm_idx': farm_idx,
                    'score': score,
                    'timing': timing
                }
                
                # Sort into phases
                if timing == ApplicationTiming.CRITICAL:
                    critical_opportunities.append(opp)
                elif timing == ApplicationTiming.APPLY_NOW:
                    urgent_opportunities.append(opp)
                elif timing == ApplicationTiming.APPLY_SOON:
                    normal_opportunities.append(opp)
                # Skip WAIT and AVOID
        
        # Sort each phase by score
        critical_opportunities.sort(key=lambda x: x['score'], reverse=True)
        urgent_opportunities.sort(key=lambda x: x['score'], reverse=True)
        normal_opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        # Execute phases in order
        self._execute_phase(critical_opportunities, "CRITICAL", max_allocation=20.0)
        self._execute_phase(urgent_opportunities, "URGENT", max_allocation=15.0)
        self._execute_phase(normal_opportunities, "NORMAL", max_allocation=10.0)
    
    def _execute_phase(
        self, 
        opportunities: List[Dict],
        phase_name: str,
        max_allocation: float
    ):
        """Execute allocation for a specific urgency phase."""
        for opp in opportunities:
            stp_idx = opp['stp_idx']
            farm_idx = opp['farm_idx']
            
            # Re-check availability (may have been depleted in earlier phase)
            if self.state.stp_storage[stp_idx] < 2000:
                continue
            if self.state.farm_n_remaining[farm_idx] <= 0:
                continue
            
            # Calculate shipment size based on phase
            max_truck = max_allocation * 1000  # kg
            farm_cap = self.state.farm_n_remaining[farm_idx] / 0.05
            stp_avail = self.state.stp_storage[stp_idx]
            
            send_kg = min(max_truck, farm_cap, stp_avail)
            send_tons = send_kg / 1000.0
            
            if send_tons < 1.0:
                continue
            
            # Validate before applying
            if validate_action(self.state, stp_idx, farm_idx, send_tons)[0]:
                self.state.apply_action(stp_idx, farm_idx, send_tons)
    
    def allocate_day_simple(self):
        """
        SIMPLER VERSION: Single-phase with strategic scoring.
        Use this if multi-phase is too complex.
        """
        current_date = self.state.current_date
        
        # Collect all opportunities
        opportunities = []
        
        for stp_idx in range(len(self.state.stp_registry)):
            available_kg = self.state.stp_storage[stp_idx]
            if available_kg < 5000:
                continue
            
            for farm_idx in range(len(self.state.farm_locations)):
                if self.state.farm_n_remaining[farm_idx] <= 0:
                    continue
                
                # Strategic urgency
                score, timing = self._calculate_strategic_urgency(
                    farm_idx, stp_idx, current_date
                )
                
                if score > 0 and timing != ApplicationTiming.AVOID:
                    opportunities.append({
                        'stp_idx': stp_idx,
                        'farm_idx': farm_idx,
                        'score': score,
                        'timing': timing
                    })
        
        # Sort by strategic score
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        # Execute allocations
        for opp in opportunities:
            stp_idx = opp['stp_idx']
            farm_idx = opp['farm_idx']
            timing = opp['timing']
            
            if self.state.stp_storage[stp_idx] < 2000:
                continue
            if self.state.farm_n_remaining[farm_idx] <= 0:
                continue
            
            # Allocation size based on timing
            if timing == ApplicationTiming.CRITICAL:
                max_tons = 20.0  # Maximum for critical
            elif timing == ApplicationTiming.APPLY_NOW:
                max_tons = 15.0
            else:
                max_tons = 10.0
            
            max_truck = max_tons * 1000
            farm_cap = self.state.farm_n_remaining[farm_idx] / 0.05
            
            send_kg = min(self.state.stp_storage[stp_idx], max_truck, farm_cap)
            send_tons = send_kg / 1000.0
            
            if send_tons >= 1.0:
                if validate_action(self.state, stp_idx, farm_idx, send_tons)[0]:
                    self.state.apply_action(stp_idx, farm_idx, send_tons)


# ============================================================================
# SOLVER RUNNER FUNCTIONS
# ============================================================================

def run_strategic_solver(state, multi_phase: bool = True, use_cache: bool = True):
    """
    Run strategic solver for full year.
    
    Args:
        state: SystemState instance
        multi_phase: Use 3-phase allocation (Critical/Urgent/Normal)
        use_cache: Pre-build constraint cache for speed
    
    Returns:
        state with completed action_history
    """
    print("\n" + "="*70)
    print("STRATEGIC SOLVER - PERSON 2")
    print("="*70)
    print(f"  Multi-phase: {multi_phase}")
    print(f"  Constraint cache: {use_cache}")
    
    solver = StrategicSolver(state, use_cache=use_cache)
    
    # Save cache for team if built
    if solver.cache:
        solver.cache.save("outputs/constraint_cache.pkl")
        print("  ✓ Cache saved to outputs/constraint_cache.pkl")
    
    print("\n  Starting 365-day simulation...")
    
    for day in range(365):
        # Choose allocation method
        if multi_phase:
            solver.allocate_day_strategic()
        else:
            solver.allocate_day_simple()
        
        state.advance_day()
        
        # Progress report
        if day % 50 == 0:
            summary = state.get_state_summary()
            print(f"    Day {day+1}/365: "
                  f"Storage={summary['total_stp_storage_kg']/1e6:.1f}M kg, "
                  f"Demand={summary['total_farm_demand_kg']/1e6:.1f}M kg, "
                  f"Actions={len(state.action_history)}")
    
    print(f"\n  ✓ Strategic solver complete!")
    print(f"  ✓ Generated {len(state.action_history)} actions")
    print("="*70 + "\n")
    
    return state


def run_improved_solver_with_lookahead(state):
    """
    Drop-in replacement for original run_improved_solver().
    Uses strategic enhancements automatically.
    """
    return run_strategic_solver(state, multi_phase=True, use_cache=True)


# ============================================================================
# COMPARISON TESTING
# ============================================================================

def compare_solvers(state_original, state_strategic):
    """
    Compare ImprovedSolver vs StrategicSolver side-by-side.
    
    Usage:
        from src.data_loader import load_all_data
        from src.state import SystemState
        
        data = load_all_data()
        state1 = SystemState(data)
        state2 = SystemState(data)
        
        # Run both
        from src.solver_improved import run_improved_solver
        run_improved_solver(state1)
        
        from src.solver_strategic import run_strategic_solver
        run_strategic_solver(state2)
        
        # Compare
        compare_solvers(state1, state2)
    """
    print("\n" + "="*70)
    print("SOLVER COMPARISON")
    print("="*70)
    
    print(f"\nImproved Solver:")
    print(f"  Actions: {len(state_original.action_history)}")
    print(f"  Final STP storage: {state_original.stp_storage.sum()/1e6:.2f}M kg")
    print(f"  Final farm demand: {state_original.farm_n_remaining.sum()/1e6:.2f}M kg")
    
    print(f"\nStrategic Solver:")
    print(f"  Actions: {len(state_strategic.action_history)}")
    print(f"  Final STP storage: {state_strategic.stp_storage.sum()/1e6:.2f}M kg")
    print(f"  Final farm demand: {state_strategic.farm_n_remaining.sum()/1e6:.2f}M kg")
    
    print(f"\nDifference:")
    action_diff = len(state_strategic.action_history) - len(state_original.action_history)
    print(f"  Actions: {action_diff:+d} ({action_diff/len(state_original.action_history)*100:+.1f}%)")
    
    print("="*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\nSTRATEGIC SOLVER - PERSON 2")
    print("="*70)
    print("\nThis solver combines:")
    print("  ✓ ImprovedSolver's urgency scoring")
    print("  ✓ LookaheadAnalyzer's strategic planning")
    print("  ✓ ConstraintCache's O(1) performance")
    print("  ✓ Multi-phase allocation (Critical/Urgent/Normal)")
    print("\nIntegration:")
    print("  from src.solver_strategic import run_strategic_solver")
    print("  state = run_strategic_solver(state, multi_phase=True)")
    print("\nOr use as drop-in replacement:")
    print("  from src.solver_strategic import run_improved_solver_with_lookahead")
    print("  state = run_improved_solver_with_lookahead(state)")
    print("="*70 + "\n")