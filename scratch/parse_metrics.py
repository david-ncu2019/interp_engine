import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from pathlib import Path

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

results = []
for s in scenarios:
    for mode in ["gp", "kriging"]:
        csv_path = Path(f"output/{s}/cv_results_{mode}.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            obs = df['Observed']
            pred = df['Predicted']
            rmse = np.sqrt(mean_squared_error(obs, pred))
            r2 = r2_score(obs, pred)
            results.append({"Scenario": s, "Mode": mode, "RMSE": rmse, "R2": r2})

df_res = pd.DataFrame(results)
if len(df_res) > 0:
    summary = df_res.pivot(index="Scenario", columns="Mode", values=["RMSE", "R2"])
    print(summary)
    summary.to_csv("comparison_summary.csv")
else:
    print("No CSVs found")
