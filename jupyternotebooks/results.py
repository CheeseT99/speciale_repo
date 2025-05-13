# System imports
import os
import sys
import time
from scipy.optimize import Bounds  
# Construct and print the path
functions_path = os.path.abspath(os.path.join(os.getcwd(), 'functions'))

# Add to sys.path if not already there
if functions_path not in sys.path:
    sys.path.append(functions_path)

# Imports
import pandas as pd

# Own libraries
import prospect_optimizer as po
import visualisation as vis
import weight_analysis as wa
import evaluation as ev
import other_optimizations as oo


toc = time.time()

parent_dir = os.getcwd() # speciale_repo

### INPUT PARAMETERS

#strategies = ["conservative","aggressive"]  # You can switch between "aggressive" or "conservative"
#lambda_values = [1.5, 1.75, 1.99, 2.25, 2.5]
#gamma_values = [0.12, 0.2, 0.25, 0.35, 0.5]
# Investor profile parameters


################Rigtige input parametre#######################

#Sstrategies = ["conservative"]
lambda_values = [1.5, 1.75, 1.99]
gamma_values = [0.12, 0.2, 0.25]
#############################################################

#TESTING PARAMETERS
#strategies = ["conservative"]
#lambda_values = [1.99]
#gamma_values = [ 0.2]


#############Test dates#################
#start_date = '1977-06-01'
#end_date = '1991-02-01'


########################################




#####################This is the True start and end date#################\\

start_date = '1977-06-01'
end_date = '2016-12-01'

# Latest data: '2016-12-01'
# Date range for the analysis
##end_date = '2016-12-01'
########################################################################
# Static parameter
date_tag = f"{start_date}_{end_date}"
# 11 years of data
# min_obs = 120

# Changeable?
r_hat = 0.0  # Reference return
lambda_ = 2  # Base loss aversion coefficient
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

bma_returns = po.rolling_bma_returns(parent_dir, n_predictors_to_use=2, start_date=start_date, end_date=end_date)
tic = time.time()
print(f"Elapsed time: {(tic-toc):.2f}")
# Example to access a selected key: bma_returns[pd.Timestamp('2007-12-01')]


# Cache folder
cache_dir = "./bma_cache"

# Fetch historical returns
historical_returns = po.load_historical_returns(
    parent_dir=parent_dir,
    start_date=start_date,
    end_date=end_date
)

results_dict_bma = po.resultgenerator_bma(
    lambda_values=lambda_values,
    gamma_values=gamma_values,
    bma_returns=bma_returns,
    true_returns=historical_returns,
    strategies=strategies,
    date_tag=date_tag,
    cache_dir=cache_dir
)


results_dict_bma_maxsharpe = po.resultgenerator_bma_maxsharpe(
    lambda_values=lambda_values,
    gamma_values=gamma_values,
    bma_returns=bma_returns,
    true_returns=historical_returns,
    strategies=strategies,
    date_tag=date_tag,
    cache_dir=cache_dir
)
# Keys in results_dict are tuples of (strategy, lambda, gamma)
# Example to extract a single strategy result: summary_df['conservative_1.5_0.12']

summary_df = po.summarize_backtest_results(results_dict_bma)
print(summary_df.head(10))  # top 10 strategies by Sharpe

# You can now compute CE as usual:
ce_df = ev.compute_certainty_equivalents(results_dict_bma)


# Historical Mean
results_dict_historical_mean = po.resultgenerator_historical_mean(
    lambda_values=lambda_values,
    gamma_values=gamma_values,
    historical_returns=historical_returns,
    strategies=strategies,
    date_tag=date_tag
)

# MVP
results_dict_mvp = po.resultgenerator_mvp(
    lambda_values=lambda_values,
    gamma_values=gamma_values,
    historical_returns=historical_returns,
    strategies=strategies,
    date_tag=date_tag
)
#Max Sharpe

results_dict_max_sharpe = po.resultgenerator_max_sharpe(
    lambda_values=lambda_values,
    gamma_values=gamma_values,
    historical_returns=historical_returns,
    strategies=strategies,
    date_tag=date_tag
)

# Fama-French model
# Load data

test_assets_returns, factor_returns = po.load_test_assets_and_factors(
    parent_dir=parent_dir,
    start_date=start_date,
    end_date=end_date
)


# Slice matched time period
aligned_returns = test_assets_returns.join(factor_returns, how="inner")

# # Estimate expected returns
# fama_french_returns = po.estimate_factor_model_expected_returns(
#     asset_returns=aligned_returns[test_assets_returns.columns],
#     factor_returns=aligned_returns[factor_returns.columns],
#     lookback_window=120
# )

# print(fama_french_returns)

results_dict_factor_model = po.resultgenerator_factor_model(
    lambda_values=lambda_values,
    gamma_values=gamma_values,
    test_assets_returns=test_assets_returns,
    factor_returns=factor_returns,
    strategies=strategies,
    date_tag=date_tag
)


### DYNAMIC REFERENCE RETURNS:

# Load reference returns: MKT-RF, and SPY
# market_returns = po.load_reference_returns(parent_dir, ref_type="market")
# spy_returns = po.load_reference_returns(parent_dir, ref_type="spy")


# reference_rule = "market_return"  # or 'spy_returns', 'prev_portfolio', 'fixed_zero', 'rolling_avg'


# # Run across all reference rules
# results_by_reference = po.evaluate_reference_robustness(
#     lambda_values=lambda_values,
#     gamma_values=gamma_values,
#     bma_returns=bma_returns,
#     strategies=strategies,
#     date_tag=date_tag,
#     parent_dir=parent_dir
# )

# # Then extract the one you want to work with:
# results_dict = results_by_reference["market_return"]

# summary_all_refs = []

# for ref_rule, results_dict in results_by_reference.items():
#     summary_df = po.summarize_backtest_results(results_dict)
#     summary_df["Reference_Rule"] = ref_rule  # Add column to track source
#     summary_all_refs.append(summary_df.reset_index())

# # Combine into a single DataFrame
# summary_combined = pd.concat(summary_all_refs).set_index("Strategy_Key")

results_by_method = {
    "BMA": results_dict_bma,
    "Historical Mean": results_dict_historical_mean,
    "MVP": results_dict_mvp,
    "FF Model": results_dict_factor_model,
    "Max sharpe BMA": results_dict_bma_maxsharpe,
    "Max Sharpe": results_dict_max_sharpe
}
comparison_df = po.compare_methods(results_by_method)
print(comparison_df.head())

ce_combined_df = ev.compare_certainty_equivalents(results_by_method, reference=0.0)
print(ce_combined_df.head())

comparison_df = ev.merge_ce_and_performance(ce_combined_df, comparison_df)
print(comparison_df.head())
vis.plot_ce_and_sharpe_comparison(comparison_df, save_path="./plots/ce_sharpe_comparison.png")


summary_ce_df = ev.summarize_certainty_equivalents(ce_combined_df)

print(summary_ce_df)

performance_summary_df = ev.build_performance_summary_by_method(
    ce_combined_df=ce_combined_df,
    comparison_df=comparison_df,
    results_by_method={
        "BMA": results_dict_bma,
        "Historical Mean": results_dict_historical_mean,
        "MVP": results_dict_mvp,
        "FF Model": results_dict_factor_model
    }
)

print(performance_summary_df)

## Forecast accuracy analysis - No MVP since it doesn't have a forecast
forecast_accuracy_df = ev.evaluate_forecast_accuracy({
    "BMA": results_dict_bma,
    "Historical Mean": results_dict_historical_mean,
    "FF Model": results_dict_factor_model
})

print(forecast_accuracy_df)


# 1. Backtest using Mean-Variance Optimizer
results_dict_mvo = oo.backtest_portfolio_bma_mvo(
    bma_returns=bma_returns,
    risk_aversion=3.0  # Default moderate risk aversion
)

# Convert
df_mvo = oo.mvo_results_to_dataframe(results_dict_mvo)

# # Store the MVO results in the same format
# results_dict_for_eval = {
#     "PT": results_dict_bma,     # Prospect Theory
#     "MVO": {"mvo_benchmark": df_mvo}  # Mean-Variance
# }

### Comparison of optimisation methods ### 
# It's done using only the first strategy from PT. 
# Hence it's df structure and not dictionaries. 
# Fetch the first strategy, lambda, and gamma values
df_pt = results_dict_bma[f"{strategies[0]}_{lambda_values[0]}_{gamma_values[0]}"]
df_naive = oo.backtest_bma_naive_df(bma_returns)

methods = {
    "Prospect Theory": df_pt,
    "Mean-Variance": df_mvo,
    "Naive Equal-Weight": df_naive,
    # later you can add Risk Parity, FF Model, etc.
}

summary_comparison_df = oo.summarize_methods_comparison(methods, lambda_=1.5, gamma=0.5)
print(summary_comparison_df)


sensitivity_df = oo.sensitivity_analysis_ce(methods, lambda_values=[1.5, 2.0, 2.5, 3.0], gamma=0.5)
print(sensitivity_df)

vis.plot_returns_vs_pt_value(df_pt, lambda_=1.5, gamma=0.5, method_name="Prospect Theory")



# Breakpoint before plots
breakpoint = 1

## Plots

market = po.load_market_benchmark(parent_dir, start_date=start_date, end_date=end_date)

plot_strats_vs_market = vis.plot_strategies_vs_market(
    summary_dict=results_dict_bma,
    market_cumulative=market,
    parent_dir=parent_dir
)
print(f"Plot saved to: {plot_strats_vs_market}")


best_strat = vis.plot_best_strategy_vs_market(
    summary_df=summary_df,
    results_dict=results_dict_bma,
    market_cumulative=market,
    parent_dir=parent_dir,
    metric="Sharpe Ratio"  # or "Final Wealth"
)
print(f"📈 Saved best strategy vs market plot: {best_strat}")


plot_best_vs_worst_strat = vis.plot_best_and_worst_strategy_vs_market(
    summary_df=summary_df,
    results_dict=results_dict_bma,
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


avg_weights = wa.average_weights(results_dict_bma)
vol_turnover = wa.weight_volatility_and_turnover(results_dict_bma)
heatmap_paths = wa.save_weight_heatmaps(results_dict_bma, parent_dir)
top_holdings = wa.top_assets_per_strategy(results_dict_bma)
hhi_df, pivot_cons, pivot_agg = wa.herfindahl_index(results_dict_bma)
wa.save_hhi_heatmaps(pivot_cons, pivot_agg, parent_dir)

print("📊 Per-strategy summary:")
print(hhi_df.head())

print("\n📈 Avg HHI — Conservative:")
print(pivot_cons)

print("\n📈 Avg HHI — Aggressive:")
print(pivot_agg)

plot_path = wa.plot_sharpe_vs_hhi(summary_df, hhi_df, parent_dir)
print(f"📈 Sharpe vs HHI plot saved to: {plot_path}")

# Step 1: Compute CE values
ce_df = ev.compute_certainty_equivalents(results_dict_bma, reference=0.0)

# Step 2: Plot heatmap
plot_path = vis.plot_certainty_equivalent_heatmap(ce_df, parent_dir)

# Step 2: View top-performing strategies by CE
print(ce_df.head())


vis.plot_certainty_equivalent_comparison(
    ce_df=ce_combined_df,
    methods=['BMA', 'Historical Mean', 'MVP'],
    save_path="./plots/ce_comparison_barplot.png"
)



breakpoint = 1

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


