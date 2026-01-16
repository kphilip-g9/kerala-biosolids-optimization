import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

def generate_baseline_solution():
    # Load sample submission to get correct format
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    # Baseline: do nothing (0 tons everywhere)
    sample["tons"] = 0.0

    out_path = OUT_DIR / "solution.csv"
    sample.to_csv(out_path, index=False)

    print(f"Baseline solution written to {out_path}")

if __name__ == "__main__":
    generate_baseline_solution()
