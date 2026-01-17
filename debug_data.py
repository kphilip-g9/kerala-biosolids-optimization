"""
Debug data loading
"""
from src.data_loader import load_all_data
from src.state import SystemState
import pandas as pd

# Load data
data = load_all_data()

# Check daily_n_demand
print("="*60)
print("DAILY N DEMAND CHECK")
print("="*60)
demand = data['daily_n_demand']

print(f"\nShape: {demand.shape}")
print(f"Columns: {list(demand.columns)[:10]}...")  # First 10 columns
print(f"\nFirst few rows:")
print(demand.head())

print(f"\nColumn data types:")
print(demand.dtypes)

# Check totals
if 'date' in demand.columns:
    demand_copy = demand.drop('date', axis=1)
else:
    demand_copy = demand.copy()

print(f"\nTotal demand across all columns: {demand_copy.sum().sum():,.0f} kg")
print(f"Average per column: {demand_copy.sum().mean():,.0f} kg")

# Check farm_locations
print("\n" + "="*60)
print("FARM LOCATIONS CHECK")
print("="*60)
farms = data['farm_locations']
print(f"\nNumber of farms: {len(farms)}")
print(f"Farm IDs (first 10): {farms['farm_id'].head(10).tolist()}")

# Check if farm IDs match demand columns
farm_ids_set = set(farms['farm_id'].astype(str).str.strip())
demand_cols = set([str(c).strip() for c in demand.columns if c != 'date'])

matching = farm_ids_set & demand_cols
print(f"\nFarm IDs in farm_locations: {len(farm_ids_set)}")
print(f"Farm columns in daily_n_demand: {len(demand_cols)}")
print(f"Matching: {len(matching)}")

if len(matching) < len(farm_ids_set):
    print(f"\n⚠️ Missing {len(farm_ids_set) - len(matching)} farms in demand file!")
    missing = farm_ids_set - demand_cols
    print(f"Missing IDs: {list(missing)[:5]}...")

# Initialize state and check
print("\n" + "="*60)
print("STATE INITIALIZATION CHECK")
print("="*60)
state = SystemState(data)

print(f"\nFarm nitrogen remaining (first 10):")
for i in range(min(10, len(state.farm_n_remaining))):
    print(f"  Farm {i}: {state.farm_n_remaining[i]:,.2f} kg")

print(f"\nTotal nitrogen demand in state: {state.farm_n_remaining.sum():,.0f} kg")
print(f"Non-zero farms: {(state.farm_n_remaining > 0).sum()}")
print(f"Zero farms: {(state.farm_n_remaining == 0).sum()}")