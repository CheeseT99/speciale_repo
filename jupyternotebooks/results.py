#Requirements
import os
import sys
import time
# Construct and print the path
functions_path = os.path.abspath(os.path.join(os.getcwd(), 'functions'))

# Add to sys.path if not already there
if functions_path not in sys.path:
    sys.path.append(functions_path)

import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, levene, f_oneway
import prospect_optimizer as po
import visualisation as vis

import pickle as pkl

toc = time.time()

parent_dir = os.getcwd() # speciale_repo

### Input parameters for single run
r_hat = 0.0  # Reference return
lambda_ = 2  # Base loss aversion coefficient
strategies = ["conservative","aggressive"]  # You can switch between "aggressive" or "conservative"
lambda_values = [1.5, 1.75, 1.99, 2.25, 2.5]
gamma_values = [0.12, 0.2, 0.25, 0.35, 0.5]

n_predictors_to_use = 2 

# strategies = ["conservative","aggressive"]
# lambda_values = [1.5, 1.75, 1.99, 2.25, 2.5]
# gamma_values = [0.12, 0.2, 0.25, 0.35, 0.5]

#Indlæser returnsdata
#datapath = os.path.join(parent_dir+'/data/returns_data.csv')

# Short data of market returns (original from before BMA)
datapath = os.path.join(parent_dir+'/data/returns_data_short.csv')
returns = pd.read_csv(datapath, index_col='Date')


# Initiate BMA framework
bma_pickle_path = 'bma_returns.pkl'
start_date = '1997-01-01'
end_date = '2007-12-01'
date_tag = f"{start_date}_{end_date}"
# 11 years of data
# min_obs = 120

bma_returns = po.rolling_bma_returns(parent_dir, n_predictors_to_use=2, start_date=start_date, end_date=end_date)
tic = time.time()
print(f"Elapsed time: {(tic-toc):.2f}")

# # Check if the file exists
# if os.path.exists(bma_pickle_path):
#     print("BMA File exists, loading data from pickle.")
#     # Load the data from the pickle file
#     with open(bma_pickle_path, 'rb') as file:
#         bma_returns = pkl.load(file)
# else:
#     pass

# bma_returns = po.rolling_bma_returns(parent_dir, n_predictors_to_use, start_date, end_date, min_obs=120, tau = 1.1)

#bma_returns = po.initialize_bma_returns(bma_pickle_path, parent_dir, n_predictors_to_use, start_date, end_date)

results_dict = po.resultgenerator_bma(
    lambda_values=lambda_values,
    gamma_values=gamma_values,
    bma_returns=bma_returns,
    strategies=strategies,
    date_tag=date_tag
)

summary_df = po.summarize_backtest_results(results_dict)
print(summary_df.head(10))  # top 10 strategies by Sharpe


market = po.load_market_benchmark(parent_dir, start_date=start_date, end_date=end_date)

plot_strats_vs_market = vis.plot_strategies_vs_market(
    summary_dict=results_dict,
    market_cumulative=market,
    parent_dir=parent_dir
)
print(f"Plot saved to: {plot_strats_vs_market}")


best_strat = vis.plot_best_strategy_vs_market(
    summary_df=summary_df,
    results_dict=results_dict,
    market_cumulative=market,
    parent_dir=parent_dir,
    metric="Sharpe Ratio"  # or "Final Wealth"
)
print(f"📈 Saved best strategy vs market plot: {best_strat}")


plot_best_vs_worst_strat = vis.plot_best_and_worst_strategy_vs_market(
    summary_df=summary_df,
    results_dict=results_dict,
    market_cumulative=market,
    parent_dir=parent_dir,
    metric="Sharpe Ratio"  # or "Final Wealth"
)
print(f"📉 Best vs worst strategy plot saved to: {plot_best_vs_worst_strat}")
# Plot Sharpe ratio heatmaps

metrics_to_plot = ["Sharpe Ratio", "Final Wealth", "Max Drawdown"]

plot_paths = vis.plot_all_heatmaps(
    summary_df=summary_df,
    metrics=metrics_to_plot,
    parent_dir=parent_dir
)

print("📊 All heatmaps saved:")
for path in plot_paths:
    print(f" - {path}")

# final_value = results_df['Compounded Returns'].iloc[-1]
# sr = results_df['Portfolio Returns'].mean() / results_df['Portfolio Returns'].std()

# print(f"Final portfolio value: {final_value:.3f}")
# print(f"Sharpe Ratio: {sr:.2f}")

# bma_returns_result = po.resultgenerator(lambda_values, gamma_values, bma_returns, strategies)


# result_naive_single_inputs = po.optimize_portfolio(returns, r_hat, lambda_, strategies)
# print("Results are generating")
# results_naive = po.resultgenerator(lambda_values, gamma_values, returns, strategies)
# print("Done")



# # Create returns_conservative
# returns_conservative = {
#     key: value for key, value in results_naive.items() if "conservative" in key
# }

# # Create returns_aggressive
# returns_aggressive = {
#     key: value for key, value in results_naive.items() if "aggressive" in key
# }


# print("Returns aggressive:\n", returns_aggressive.values())
# print("Returns conservative:\n", returns_conservative.values())

tic = time.time()
print(f"Elapsed time: {(tic-toc):.2f}")


