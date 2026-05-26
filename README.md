# ML_FinalProject

Benchmark of regression and feedforward neural network (FNN) models for detecting drug–protein interactions in simulated chemoproteomics data. Final project for CS6140 (Machine Learning) at Northeastern University.

## Overview

Chemoproteomics measures interactions between small molecules and the proteome, but there is no standardized statistical approach for calling drug–protein interactions from dose–response data. This project builds a simulation framework that mimics realistic chemoproteomics experiments — class imbalance, technical variability, outliers, and varying replicate / dose designs — and benchmarks three methods on the resulting data:

- **Isotonic Regression (IR)** — non-parametric monotonic (non-increasing) fit on log-intensities via `sklearn.isotonic`, with an F-test against a constant null model.
- **Four-Parameter Logistic Regression (4PL)** — sigmoidal dose–response curve fit on DMSO-normalized intensities via `lmfit`, with an F-test against a constant null model.
- **Feedforward Neural Network (FNN)** — single hidden layer with ReLU + sigmoid output (PyTorch), trained on per-replicate dose vectors with binary cross-entropy loss and Adam.

Each simulated protein follows one of three interaction profiles (strong, weak, non-interactor) across DMSO + 8 drug doses. Six datasets vary replicate count, technical variance, outlier rate, and dose design to probe robustness.

**Conclusion:** Isotonic regression was the most robust across conditions and is the recommended method for chemoproteomics dose–response analysis. The FNN matched IR/4PL only under low-noise, high-replicate conditions and failed in sparse or noisy regimes. See the final report PDF for full methods and results.

## Repository contents

| File | Purpose |
|------|---------|
| `Full_Analysis.ipynb` | **Main entry point.** End-to-end notebook that builds the six simulated datasets, fits IR / 4PL / FNN, and produces all figures and tables. The PyTorch FNN architecture and the accuracy/precision/recall/F1 calculations live in this notebook. |
| `data_generation_simulation.py` | `simulate_chemo_proteinlevel_nonparametric(...)` — simulates protein-level dose–response data with configurable proteins, class proportions (`TP`, `TW`, `TN`), replicates, technical variance, dose set, and outlier rate. |
| `bio_template1.csv` | Real-data-derived template log-intensities for each interaction type (strong / weak / no interaction), used by the simulator to set per-dose base means. |
| `FourPL_model_fit.py` | `four_pl_fit_logscale(...)` — fits the 4PL curve to DMSO-normalized relative intensities and returns MSE, F-statistic, and p-value vs. a constant null. |
| `ir_model_fit_logscale.py` | `isotonic_regression_fit_logscale(...)` — fits a non-increasing isotonic regression to log-intensities and returns SSE, F-statistic, and p-value vs. a constant null. |
| `Model_eval_IR_4PL.py` | `evaluate_model_fit(data, method=...)` — runs IR or 4PL across every protein in a dataset, applies BH/FDR correction, and reports counts of significant hits per interaction group. |
| `FNN_reformat_data.py` | `prepare_fnn_dataset(...)` — reshapes simulated long-format data into per-replicate dose vectors (length-9) for FNN training. (The notebook also defines an equivalent function inline.) |
| `plot_simulation_curve.py` | Plots a single protein's simulated dose–response curve, highlighting outliers and optionally overlaying the 4PL or IR fit. |
| `ir_visualize.py` | Plots an isotonic regression fit for a single protein (relative-intensity scale). |
| `plot_confusion_matrix.py` | Confusion matrix plot (matplotlib only). |
| `plot_recall.py` | Recall-by-protein-group grouped bar chart across all datasets and models (Figure 1 in the report). |
| `plot_roc_curve.py` | ROC curves and AUC for the trained PyTorch FNN on train / test sets. |
| `protein_group_recall.pdf` | Saved version of the recall figure. |

## How to run

```bash
git clone https://github.com/sszvetecz/ML_FinalProject.git
cd ML_FinalProject
pip install numpy pandas scipy scikit-learn lmfit torch matplotlib jupyter
jupyter notebook Full_Analysis.ipynb
```

Run `Full_Analysis.ipynb` top-to-bottom. All `.py` files are helper modules imported by the notebook — they are not standalone scripts.

## Simulated datasets

| Dataset | Replicates | σ²_tech | Outliers | Doses |
|---------|-----------|---------|----------|-------|
| SimData1 | 3 | 0.25 | 0% | DMSO + 8 doses (1, 3, 10, 30, 100, 300, 1000, 3000 nM) |
| SimData2 | 3 | 0.25 | 5% | DMSO + 8 doses |
| SimData3 | 3 | 0.80 | 0% | DMSO + 8 doses |
| SimData4 | 3 | 0.25 | 0% | DMSO, 1 nM, 1000 nM, 3000 nM |
| SimData5 | 2 | 0.80 | 0% | DMSO + 8 doses |
| SimData6 | 1 | 0.80 | 0% | DMSO + 8 doses |

Each dataset contains 5,000 proteins with the class proportions 0.1% strong / 0.9% weak / 99% non-interactor, reflecting the class imbalance typical of chemoproteomics screens.

## Author

Sarah Szvetecz — CS6140, Northeastern University.
