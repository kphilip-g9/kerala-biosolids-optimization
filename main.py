"""
Main Execution Pipeline
FIXED: Uses Improved Solver + Generates 91,250 Rows
"""
import pandas as pd
import numpy as np
from pathlib import Path
import itertools
import traceback

from src.data_loader import load_all_data
from src.state import SystemState
from src.utils import check_data_integrity
from src.scoring import CarbonScorer
from src.solver_improved import run_improved_solver # <--- IMPORTING NEW SOLVER

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    print("="*60 + "\nKERALA BIOSOLIDS OPTIMIZATION\n" + "="*60)
    
    # 1. Load
    print("[1/5] Loading data...")
    data = load_all_data()
    
    # 2. Init
    print("\n[2/5] Initializing State...")
    state = SystemState(data)
    
    # 3. Run Simulation (New Solver)
    print("\n[3/5] Running Improved Solver...")
    run_improved_solver(state)
    
    # 4. Export Submission (Dense Format)
    print("\n[4/5] Generating 91,250 Row Submission...")
    try:
        # Create full grid
        dates = pd.date_range("2025-01-01", "2025-12-31")
        farms = state.farm_locations["farm_id"].unique()
        skeleton = pd.DataFrame(list(itertools.product(dates, farms)), columns=["date_dt", "farm_id"])
        skeleton["date"] = skeleton["date_dt"].dt.strftime("%Y-%m-%d")
        
        # Merge actuals
        if state.action_history:
            actual = pd.DataFrame(state.action_history)
            actual["date"] = pd.to_datetime(actual["date"]).dt.strftime("%Y-%m-%d")
            actual = actual.groupby(["date", "farm_id"], as_index=False).agg({
                "tons_delivered": "sum", "stp_id": "first"
            })
            final = pd.merge(skeleton, actual, on=["date", "farm_id"], how="left")
        else:
            final = skeleton.copy()
            final["tons_delivered"] = 0
            final["stp_id"] = state.stp_registry.iloc[0]["stp_id"]

        # Fill NaNs
        final["tons_delivered"] = final["tons_delivered"].fillna(0)
        final["stp_id"] = final["stp_id"].fillna(state.stp_registry.iloc[0]["stp_id"])
        
        # Add ID
        final.sort_values(["date_dt", "farm_id"], inplace=True)
        final.insert(0, "id", range(len(final)))
        
        # Save
        out_file = OUTPUT_DIR / "solution.csv"
        final[["id", "date", "stp_id", "farm_id", "tons_delivered"]].to_csv(out_file, index=False)
        
        print(f"✓ Saved: {out_file}")
        print(f"  Rows: {len(final)} (Expected: 91250)")
        print(f"  Total Tons: {final['tons_delivered'].sum():,.0f}")
        
    except Exception as e:
        print(f"Export failed: {e}")
        traceback.print_exc()

    # 5. Score
    print("\n[5/5] Scoring...")
    scorer = CarbonScorer(state)
    # Score only active rows for speed
    scorer.score_run(final[final["tons_delivered"] > 0].copy())

if __name__ == "__main__":
    main()