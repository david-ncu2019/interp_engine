# Spatial Interpolation Engine

## What is this?

Imagine you are measuring temperatures, soil quality, or rainfall across a large field, but you can only take samples at 500 specific spots. You still need to know the values for the whole field. 

This **Spatial Interpolation Engine** is a smart tool that takes your scattered data points and fills in the blanks. It creates a continuous, high-quality map or surface out of your scattered measurements.

## Key Features

We have built this tool to be fully automatic and very robust. It does the hard mathematical work for you so you can focus on the results:

1. **Automatic Trend Detection:** 
   Sometimes, data has a clear overall direction (like getting hotter as you go South). The engine automatically checks if a big trend exists using statistical tests. If a trend is found, the engine removes it, focuses on mapping the small local details, and then adds the big trend back at the end.

2. **Two Powerful Brains (Engines):**
   You can choose between two different mathematical engines to create your map:
   * **Kriging Engine:** A classic, highly reliable method used heavily in Earth Sciences. It tests 9 different mathematical shapes (called variograms) to find the best way to connect your points.
   * **Gaussian Process (GP) Engine:** A modern machine learning method. We upgraded it so it automatically decides if your map should look very smooth or very rough and chaotic. It is also incredibly good at ignoring bad data or "noise" (known as the nugget effect).

3. **Handles Messy Real-World Data:**
   * **Fixes Bad Coordinates:** If two measurements are accidentally recorded at the exact same location with different values, the engine automatically merges them or slightly shifts them so the math doesn't crash.
   * **Normal-Score Transform (NST):** Real-world data is often skewed (meaning most values are small, but a few are extremely huge). This tool can temporarily transform your skewed data into a perfect "bell curve" so the math works perfectly, and then transform the final map back to the original numbers.

4. **Honest Accuracy Scoring (Cross-Validation):**
   The engine tests itself to prove its accuracy. It splits your map into diagonal strips. It hides one strip, uses the surrounding data to guess what is in the hidden strip, and then checks its guesses. This gives you a highly accurate "R-Squared" (R²) and "RMSE" error score, so you know exactly how much you can trust the final map.

## How to Use It

1. **Prepare your data:** Put your data in a CSV file with columns for X (longitude), Y (latitude), and Value.
2. **Configure:** Open the `config.yaml` file. You can choose which engine to use (`mode: gp` or `mode: kriging`) and tell it where your data file is.
3. **Run:** Open your terminal or command prompt and type:
   ```bash
   conda run -n fafalab python main.py
   ```
   *(Or just `python main.py` if your Python environment is already active).*

## What Output Will I Get?

The engine will create a new folder inside the `output/` directory with your dataset's name. Inside, you will find files sorted alphabetically (A, B, C...) so they tell a story of how the map was built:

*   **A_ground_truth.png:** A picture of the real map (if you provided test data).
*   **B_convex_hull.png:** Shows the boundary area where the engine is making guesses.
*   **C_trend_components.png:** Shows the big overall slope the engine found (if any).
*   **D & E (Variograms):** Math charts showing how points connect to each other over distance.
*   **F_anisotropy_ellipse.png:** A shape showing if your data stretches more in one direction (like a river) than another.
*   **G_prediction_surface.png:** **The final map!** This is the main result.
*   **H_comparison.png:** A side-by-side comparison of the guess vs. the real map.
*   **I_cv_dashboard.png & cv_results.csv:** Detailed accuracy reports.
*   **predicted.nc:** The raw map data saved in a scientific format (NetCDF) that you can open in other mapping software.
*   **parameters.txt / .json:** A record of the exact mathematical settings the engine discovered.

## Modification History

If you are interested in the technical details of how this engine was developed and improved, please read the included markdown reports:
* `modification_v1.md` and `modification_v2.md`: Detailed logs of code changes and bug fixes.
* `batch_run_report_v1.md` through `batch_run_report_v5_v2fixes.md`: Reports showing how the engine's accuracy improved with each update across 8 different synthetic test datasets.
