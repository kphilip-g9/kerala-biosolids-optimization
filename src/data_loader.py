import pandas as pd
import json
from pathlib import Path

DATA_DIR = Path("data")

def load_all_data():
    data = {}

    # JSON config
    with open(DATA_DIR / "config.json", "r") as f:
        data["config"] = json.load(f)

    # CSV files
    data["stp_registry"] = pd.read_csv(DATA_DIR / "stp_registry.csv")
    data["farm_locations"] = pd.read_csv(DATA_DIR / "farm_locations.csv")
    data["daily_weather"] = pd.read_csv(DATA_DIR / "daily_weather_2025.csv")
    data["daily_n_demand"] = pd.read_csv(DATA_DIR / "daily_n_demand.csv")
    data["planting_schedule"] = pd.read_csv(DATA_DIR / "planting_schedule_2025.csv")

    return data
