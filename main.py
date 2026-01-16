"""
Main Execution Pipeline for Biosolids Optimization
PATCHED VERSION - Direct submission export

Run: python main.py
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

from src.data_loader import load_all_data
from src.state import SystemState
from src.ml_models import FarmPriorityModel
from src.solver_greedy import solve_full_year
from src.utils import check_data_integrity

# Configuration
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    """Execute full optimization pipeline."""
    
    print("=" * 60)
    print("KERALA BIOSOLIDS OPTIMIZATION")
    print("ML-Enhanced Greedy Solver")
    print("=" * 60)
    print()
    
    # ========================================
    # STEP 1: Load Data
    # ========================================
    print("[1/5] Loading data...")
    try:
        data = load_all_data()
        print("✓ Data loaded successfully")
        
        # Debug: Print column names to understand structure
        print("\nData columns:")
        print(f"  STP Registry: {list(data['stp_registry'].columns)}")
        print(f"  Farm Locations: {list(data['farm_locations'].columns)}")
        print(f"  Daily Weather: {list(data['daily_weather'].columns)}")
        print(f"  Daily N Demand: {list(data['daily_n_demand'].columns)}")  # ADDED
        print(f"  Planting Schedule: {list(data['planting_schedule'].columns)}")  # ADDED
        
        # Validate - TEMPORARILY DISABLED TO SEE COLUMNS
        # valid, msg = check_data_integrity(data)
        # if not valid:
        #     print(f"✗ Data validation failed: {msg}")
        #     return
        
        print(f"\n  - STPs: {len(data['stp_registry'])}")
        print(f"  - Farms: {len(data['farm_locations'])}")
        print(f"  - Weather records: {len(data['daily_weather'])}")
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================
    # STEP 2: Initialize State
    # ========================================
    print("\n[2/5] Initializing system state...")
    try:
        state = SystemState(data)
        print("✓ State initialized")
        
        summary = state.get_state_summary()
        print(f"  - Start date: {summary['date']}")
        print(f"  - Total STP storage: {summary['total_stp_storage_kg']/1e6:.2f}M kg")
        print(f"  - Total farm demand: {summary['total_farm_demand_kg']/1e6:.2f}M kg")
        
    except Exception as e:
        print(f"✗ Error initializing state: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================
    # STEP 3: Train ML Model
    # ========================================
    print("\n[3/5] Training ML priority model...")
    try:
        priority_model = FarmPriorityModel()
        priority_model.train(state)
        
    except Exception as e:
        print(f"⚠ Error training model: {e}")
        print("  Continuing with expert heuristics...")
        priority_model = FarmPriorityModel()
    
    # ========================================
    # STEP 4: Solve Full Year
    # ========================================
    print("\n[4/5] Running optimization...")
    start_time = datetime.now()
    
    try:
        # Reset state for fresh run
        state = SystemState(data)
        
        # Run solver (modifies state.action_history)
        solve_full_year(state, priority_model)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✓ Optimization complete in {elapsed:.1f}s")
        
    except Exception as e:
        print(f"✗ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================
    # STEP 5: Export Solution
    # ========================================
    print("\n[5/5] Exporting solution...")
    try:
        # Load sample submission
        sample_path = DATA_DIR / "sample_submission.csv"
        submission = pd.read_csv(sample_path)
        
        print(f"  Sample submission shape: {submission.shape}")
        print(f"  Action history size: {len(state.action_history)}")
        
        # Convert action_history to DataFrame
        actions_df = pd.DataFrame(state.action_history)
        
        if len(actions_df) > 0:
            # Ensure date column is datetime
            actions_df["date"] = pd.to_datetime(actions_df["date"])
            submission["date"] = pd.to_datetime(submission["date"])
            
            # Merge actions into submission
            # First zero everything
            submission["tons_delivered"] = 0.0
            
            # Then update with our actions
            for _, action in actions_df.iterrows():
                mask = (
                    (submission["date"] == action["date"]) &
                    (submission["stp_id"] == action["stp_id"]) &
                    (submission["farm_id"] == action["farm_id"])
                )
                
                if mask.any():
                    submission.loc[mask, "tons_delivered"] = action["tons_delivered"]
        
        # Write to file
        output_path = OUTPUT_DIR / "solution.csv"
        submission.to_csv(output_path, index=False)
        print(f"✓ Solution written to {output_path}")
        
        # Statistics
        total_tons = submission["tons_delivered"].sum()
        nonzero = (submission["tons_delivered"] > 0).sum()
        
        print(f"\nSolution Statistics:")
        print(f"  - Total tons allocated: {total_tons:,.0f}")
        print(f"  - Non-zero entries: {nonzero:,} / {len(submission):,}")
        if nonzero > 0:
            print(f"  - Avg shipment size: {total_tons/nonzero:.2f} tons")
        
        # Final state check
        final_summary = state.get_state_summary()
        print(f"\nFinal State:")
        print(f"  - Remaining STP storage: {final_summary['total_stp_storage_kg']/1e6:.1f}M kg")
        print(f"  - Remaining farm demand: {final_summary['total_farm_demand_kg']/1e6:.1f}M kg")
        
    except Exception as e:
        print(f"✗ Error exporting solution: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE ✓")
    print("=" * 60)
    print(f"\nSubmit: {output_path}")
    print("\nNext steps:")
    print("1. Submit to Kaggle")
    print("2. Check score")
    print("3. If errors, share error messages")


if __name__ == "__main__":
    main()