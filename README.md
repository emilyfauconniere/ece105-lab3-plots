# ece105-lab3-plots

## Overview

This repository contains generate_plots.py, a small utility to generate synthetic temperature readings for two sensors and produce example plots (scatter, overlapping histograms, and side-by-side boxplots) using Matplotlib. The data are reproducible via a random seed.

## Requirements

- An active Conda environment named `ece105` (or another environment of your choice).
- NumPy and Matplotlib installed in that environment.

## Installation

1. Activate the environment:

```powershell
conda activate ece105
```

2. Install dependencies (use either conda or mamba):

```powershell
conda install -c conda-forge numpy matplotlib
# or with mamba
mamba install -c conda-forge numpy matplotlib
```

## Usage

Run the example script which generates synthetic data and displays a figure with three subplots (scatter, histogram, boxplot):

```powershell
python generate_plots.py
```

The script's `main()` function accepts an optional seed (default `'0707'`) for reproducible data; running the module directly uses the default seed.

## Output

By default the script opens a Matplotlib interactive window showing the example figure. It does not write output files unless you modify the script to save figures (e.g., with `plt.savefig(...)`).

## Files

- `generate_plots.py` — Generates synthetic data and provides plotting helpers: `plot_scatter`, `plot_histogram`, `plot_boxplot`, and a `main()` demonstration.

## AI tools used and disclosure

[Placeholder] Describe any AI tools used during development and any relevant disclosure here. Replace this paragraph with your project-specific statement.

## License

(If applicable) Add license information here.
