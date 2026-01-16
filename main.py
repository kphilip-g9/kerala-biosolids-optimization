"""
Main Execution Pipeline for Biosolids Optimization
Integrated with Ground Truth Carbon Scoring
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import traceback

# Import project modules
from src.data_loader import load_all_data
from src.state import SystemState
from src.ml_models import FarmPriorityModel
from src.solver_greedy import solve_full_year
from src.utils import check_data_integrity
from src.scoring import CarbonScorer  # <--- The new scoring module

# Configuration
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    """Execute full optimization pipeline."""
    
    print("=" * 60)
    print("KERALA BIOSOLIDS OPTIMIZATION")
    print("ML-Enhanced Greedy Solver + Carbon Scoring")
    print("=" * 60)
    print()
    
    # ========================================
    # STEP 1: Load Data
    # ========================================
    print("[1/5] Loading data...")
    try:
        data = load_all_data()
        print("✓ Data loaded successfully")
        
        # Basic integrity check
        valid, msg = check_data_integrity(data)
        if not valid:
            print(f"✗ Data Integrity Error: {msg}")
            return
            
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        traceback.print_exc()
        return

    # ========================================
    # STEP 2: Initialize State
    # ========================================
    print("\n[2/5] Initializing System State...")
    try:
        state = SystemState(data)
        print(f"✓ State initialized for {state.current_date.date()}")
        print(f"  - {len(state.stp_registry)} STPs")
        print(f"  - {len(state.farm_locations)} Farms")
    except Exception as e:
        print(f"✗ State initialization failed: {e}")
        traceback.print_exc()
        return

    # ========================================
    # STEP 3: Train/Init ML Models
    # ========================================
    print("\n[3/5] Preparing ML Priority Model...")
    try:
        priority_model = FarmPriorityModel()
        
        # Check if we need training (or just use heuristics if model is simple)
        # Using the synthetic data generation from your ML module
        X_train, y_train = priority_model.generate_synthetic_training_data(state)
        priority_model.model.fit(X_train, y_train)
        priority_model.is_trained = True
        print(f"✓ Model trained on {len(X_train)} synthetic samples")
        
    except Exception as e:
        print(f"⚠ ML Model failed, falling back to heuristics: {e}")
        # Ensure model works even if training fails (heuristics fallback)
        priority_model.is_trained = False

    # ========================================
    # STEP 4: Run Simulation (Greedy Solver)
    # ========================================
    print("\n[4/5] Running Simulation (365 Days)...")
    try:
        solve_full_year(state, priority_model)
        print("✓ Simulation complete.")
    except Exception as e:
        print(f"✗ Simulation crashed: {e}")
        traceback.print_exc()
        return

    # ========================================
    # STEP 5: Export & Verify
    # ========================================
    print("\n[5/5] Exporting Solution & Scoring...")
    try:
        # 1. Convert action history to DataFrame
        if not state.action_history:
            print("⚠ WARNING: No actions were taken! Solution is empty.")
            return

        submission = pd.DataFrame(state.action_history)
        
        # Ensure correct column order for Kaggle/Submission
        cols = ["date", "stp_id", "farm_id", "tons_delivered"]
        submission = submission[cols]
        
        # Format date as YYYY-MM-DD
        submission["date"] = pd.to_datetime(submission["date"]).dt.strftime("%Y-%m-%d")
        
        # 2. Write CSV
        output_path = OUTPUT_DIR / "solution.csv"
        submission.to_csv(output_path, index=False)
        print(f"✓ Solution saved to: {output_path}")
        
        # 3. Statistics
        total_tons = submission["tons_delivered"].sum()
        print(f"\nStats:")
        print(f"  - Total Tons Delivered: {total_tons:,.0f}")
        print(f"  - Total Trips (approx): {len(submission):,}")

        # ========================================
        # STEP 6: VERIFY SCORE (Ground Truth)
        # ========================================
        # This uses the new src.scoring module we just built
        
        scorer = CarbonScorer(state)
        # We pass the submission DF we just created
        final_score = scorer.score_run(submission)
        
        # The scorer handles printing the beautiful report internally
        
    except Exception as e:
        print(f"✗ Error during export/scoring: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()