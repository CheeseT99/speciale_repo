#Requirements
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, levene, f_oneway
import prospect_optimizer as po
import os
import sys
import time
import pickle as pkl

toc = time.time()

parent_dir = os.getcwd() # speciale_repo

### Input parameters
r_hat = 0.0  # Reference return
lambda_ = 2  # Base loss aversion coefficient
strategy = ["conservative","aggressive"]  # You can switch between "aggressive" or "conservative"
lambda_values = [2.5]
gamma_values = [0.5]

#Indlæser returnsdata
datapath = os.path.join(parent_dir+'/data/returns_data.csv')

returns = pd.read_csv(datapath, index_col='Date')

# Fetch r_s from BMA framework
# Based on:
# Factors ['Mkt-RF', 'HML', 'CMA', 'SMB', 'MMOM', 'RMW']
# Predictors ['b.m', 'ep', 'de', 'ntis', 'dy', 'svar', 'dp']

factors_path = os.path.join(parent_dir, "conditional_dump_models_MMax_7_OOS_new1.pkl")

# Open and load the pickle file
# with open(factors_path, 'rb') as file:
#     data = pkl.load(file)
# r_s = pd.read_csv(os.path.join(parent_dir, "data", "conditional_dump_models_MMax_7_OOS_new1.pkl"))  # Replace with actual path to your data


# factors_result = po.optimize_portfolio(r_s, r_hat, lambda_, strategy)


returns_result = po.optimize_portfolio(returns, r_hat, lambda_, strategy)
print("Results are generating")
results = po.resultgenerator(lambda_values, gamma_values, returns)
print("Done")
aggressive_results =results[0]
conservative_results = results[1]



tic = time.time()
print(f"Elapsed time: {(tic-toc):.2f}")