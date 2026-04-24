import yaml
from pathlib import Path
import subprocess
import os

scenarios = [
    "S5_SGS_Extreme_Aniso",
    "S6_SGS_Nested",
    "S7_SGS_Clustered",
    "S8_SGS_HighNugget"
]

config_path = Path("config.yaml")

for s in scenarios:
    print(f"\n>>> PROCESSING {s} ...")
    
    # Update config.yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    cfg["input"]["filepath"] = f"test_data/{s}.csv"
    cfg["input"]["ground_truth_filepath"] = f"test_data/{s}_ground_truth.csv"
    cfg["input"]["columns"] = {"x": "X", "y": "Y", "value": "Value"}
    cfg["geometry"]["resolution_m"] = 50.0
    
    # For nested or high nugget, maybe we need more restarts
    cfg["engine"]["gp"]["n_restarts"] = 10
    
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    
    # Run main.py
    subprocess.run(["conda", "run", "-n", "fafalab", "python", "main.py"], check=True)

print("\nDone. All scenarios processed.")
