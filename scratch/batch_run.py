import yaml
from pathlib import Path
import subprocess
import pandas as pd
import numpy as np

scenarios = [
    "S1_Isotropic",
    "S2_Anisotropic_45deg",
    "S3_Sparse_Aniso_120deg",
    "S4_HighNugget_Isotropic",
    "S5_SGS_Extreme_Aniso",
    "S6_SGS_Nested",
    "S7_SGS_Clustered",
    "S8_SGS_HighNugget"
]

config_path = Path("config.yaml")

# Storage for summary
all_results = []

def run_scenarios(mode):
    for s in scenarios:
        print(f"\n>>> PROCESSING {s} | MODE: {mode}")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        cfg["input"]["filepath"] = f"test_data/{s}.csv"
        cfg["input"]["ground_truth_filepath"] = f"test_data/{s}_ground_truth.csv"
        cfg["engine"]["mode"] = mode
        cfg["preprocessing"]["detrend"]["auto_detect"] = True
        
        # Ensure we have enough trials/restarts for batch quality
        if mode == "kriging":
            cfg["engine"]["kriging"]["n_trials"] = 150
        else:
            if "n_restarts" in cfg["engine"]["gp"]:
                del cfg["engine"]["gp"]["n_restarts"]
            cfg["engine"]["gp"]["n_optuna_trials"] = 300

        with open(config_path, "w") as f:
            yaml.dump(cfg, f)

        # Run main.py and capture output to extract metrics
        result = subprocess.run(
            ["conda", "run", "--no-capture-output", "-n", "fafalab", "python", "main.py"],
            capture_output=True, text=True
        )
        
        # Print the output so user can see progress
        print(result.stdout)
        
        # Parse metrics from stdout
        metrics = {"Scenario": s, "Mode": mode}
        for line in result.stdout.split('\n'):
            if "MAE  :" in line: metrics["MAE"] = float(line.split(":")[1].strip())
            if "RMSE :" in line: metrics["RMSE"] = float(line.split(":")[1].strip())
            if "R²   :" in line: metrics["R2"] = float(line.split(":")[1].strip())
            if "-> Significant trend detected" in line: metrics["Trend"] = "Yes"
            if "-> No significant trend" in line: metrics["Trend"] = "No"
            
        all_results.append(metrics)

# Run both modes
run_scenarios("kriging")
run_scenarios("gp")

# Create comparison table
df = pd.DataFrame(all_results)
print("\n" + "="*50)
print("FINAL COMPARISON SUMMARY")
print("="*50)
summary = df.pivot(index="Scenario", columns="Mode", values=["RMSE", "R2"])
print(summary)

summary.to_csv("comparison_summary.csv")
print("\nResults saved to comparison_summary.csv")
