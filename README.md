# Bayesian Model Averaging & Prospect-Theory Portfolio Optimisation  
**Master’s Thesis – University of Copenhagen, 2025**

---

## Overview
This repository contains all code, data references, cached results, and figures for our thesis on combining Bayesian Model Averaging (BMA) with Prospect Theory preferences.



* **Main script:** `results.py`  
  Runs the full back-test pipeline and regenerates BMA models and backtests as reported in the thesis.
* **Core functions:** everything lives in `functions/`; the optimisation logic sits in `prospect_optimizer.py`.
* **Analysis:** side analyses and plots are collected under `results/`.
**Avramov 2023:** ConditionalAssetPricingLogmarginalLikelood.py, ConditionalAssetPricingCommonFunction.py, CommonFunctions.py, GammaFunctions.py, ConditionalAssetPricingOOSPredictions.py and tictoc.py as well as the csv files in data are from Avramov 2023 and are used to generate the results in the thesis. We have modified the original code to run with our own functionalities so it is not a direct copy of the original code. The original code can be found at https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13226 under "supporting information". 

---

