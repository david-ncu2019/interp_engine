# Interpolation Engine

A desktop app that turns a handful of scattered measurements into a smooth, colored map.

You give it a spreadsheet (CSV) where each row has a location (an X and a Y) and a measured
value at that spot — for example water level, ground height, or soil readings. The app fills in
the gaps between your points and draws a complete map. It also draws a second map that shows
where the estimate is trustworthy and where it's more of a guess.

---

## What you get

- **A filled-in map** built from your scattered points.
- **A confidence map** showing where the estimate is strong and where it's weak.
- **An optional accuracy check** you can switch on when you want a quality score.
- **A comparison tool** to test the result against values you already know.
- **Save and export** so you can keep your settings and your maps.

---

## What you need

- A **Windows** computer.
- **Miniconda** installed (a free tool that sets up everything the app needs in one step).
  Download it here: https://docs.conda.io/en/latest/miniconda.html

---

## Set it up

Open the **Anaconda Prompt** (installed with Miniconda) and run these one at a time:

```bash
git clone https://github.com/david-ncu2019/interp_engine.git
cd interp_engine
conda env create -f environment.yml
conda activate interp-engine
```

The third step downloads everything the app needs — it can take a few minutes the first time.

---

## Start the app

```bash
conda activate interp-engine
python -m ui_pyside.main_window
```

(You only need the first line if you've opened a new window since setting up.)

---

## Try it out

Two small example files come with the app so you can see it work right away:

1. Start the app using the command above.
2. In the **Data** section on the left, open `test_data/S1_Isotropic.csv`.
3. Tell it which columns to use: pick `X`, `Y`, and `Value`.
4. Click **Optimize Parameters** to let the app find good settings automatically.
5. Click **Run Interpolation** — your map and confidence map appear on the right.

Want to see a different pattern? Try `test_data/S2_Anisotropic_45deg.csv` the same way.

---

## What you can do

- **Use a second screen:** drag any map out of the window onto another monitor — handy for
  presenting or teaching. It keeps updating as you change settings.
- **Hide and show maps:** click the ✕ on a map to hide it, and bring it back from the
  **View → Panels** menu. **View → Reset Layout** puts everything back the way it started.
- **Faster or more thorough:** the **accuracy check** is off by default so results come quickly.
  Turn it on (the checkbox under Run Options) when you want quality scores — it's slower,
  especially with a lot of points.
- **Save your work:** save your settings to a file and load them again later.
- **Check against known answers:** if you have a file of correct values, the app can compare its
  results to them and tell you how close it got.

---

## If something goes wrong

- **The app won't start.** Make sure you ran `conda activate interp-engine` first — the app only
  works inside that environment.
- **It feels slow.** Leave the **accuracy check** turned off, and keep your point count
  reasonable (a few thousand points at most).
