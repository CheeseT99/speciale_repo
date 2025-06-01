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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
from glob import glob

# Own libraries
import prospect_optimizer as po
import visualisation as vis
import weight_analysis as wa
import evaluation as ev
import other_optimizations as oo


toc = time.time()

parent_dir = os.getcwd() # speciale_repo


#INPUT PARAMETERS
strategies = ["conservative", "aggressive"]	
lambda_values = [1.99, 2.5]
gamma_values = [0.12, 0.2]
start_date = '1977-06-01'
end_date = '2016-12-01'
date_tag = f"{start_date}_{end_date}"
datapath = os.path.join(parent_dir+'/data/returns_data_short.csv')
returns = pd.read_csv(datapath, index_col='Date')

# Initiate BMA framework
bma_pickle_path = 'bma_returns.pkl'

bma_returns = po.rolling_bma_returns(parent_dir, n_predictors_to_use=2, start_date=start_date, end_date=end_date)
tic = time.time()
print(f"Elapsed time: {(tic-toc):.2f}")

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

results_dict_historical_mean = po.resultgenerator_historical_mean(
    lambda_values=lambda_values,
    gamma_values=gamma_values,
    historical_returns=historical_returns,
    strategies=strategies,
    date_tag=date_tag
)

test_assets_returns, factor_returns = po.load_test_assets_and_factors(
    parent_dir=parent_dir,
    historical_returns=historical_returns,
    start_date=start_date,
    end_date=end_date
)
results_dict_factor_model = po.resultgenerator_factor_model(
    lambda_values=lambda_values,
    gamma_values=gamma_values,
    test_assets_returns=test_assets_returns,
    factor_returns=factor_returns,
    reference_series=0.0,
    strategies=strategies,
    date_tag=date_tag
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

results_by_pt_method = {
    "PT by BMA": results_dict_bma,
    "PT by Historical Mean": results_dict_historical_mean,
    "PT by FF Model": results_dict_factor_model,
}


# Define the exact desired columns
desired_columns = [
    'Portfolio Returns',
    'Compounded Returns',
    'Portfolio Weights',
    'Expected Portfolio Returns',
    'Reference Return'
]

all_2008_df_list = []

for pt_method, df in results_by_pt_method.items():
    for strat_key, inner_df in df.items():
        inner_df_2008 = inner_df[inner_df.index.year == 2008].copy()

        # Keep only desired columns if they exist
        existing_cols = [col for col in desired_columns if col in inner_df_2008.columns]
        inner_df_2008 = inner_df_2008[existing_cols]

        # Add metadata
        inner_df_2008['PT Method'] = pt_method
        inner_df_2008['Strategy Key'] = strat_key

        all_2008_df_list.append(inner_df_2008)

# Combine all rows
all_2008_df = pd.concat(all_2008_df_list)
final_column_order = desired_columns + ['PT Method', 'Strategy Key']
all_2008_df = all_2008_df[final_column_order]

# print(all_2008_df)

# Now filter for PT BMA and compute average Portfolio Returns across Strategy Keys
pt_bma_df = all_2008_df[all_2008_df['PT Method'] == 'PT by BMA']
avg_returns_by_date = pt_bma_df.groupby(pt_bma_df.index)[['Portfolio Returns', 'Expected Portfolio Returns']].mean()

# Ensure the weights are converted from list-strings to proper numpy arrays
weights_extracted = all_2008_df[['PT Method', 'Portfolio Weights']].copy()
weights_extracted['Portfolio Weights'] = weights_extracted['Portfolio Weights'].apply(lambda x: np.array(x))

# Group by PT Method and compute average weights
grouped_weights = weights_extracted.groupby('PT Method')['Portfolio Weights'].apply(lambda x: np.mean(np.stack(x.values), axis=0))

# Convert to DataFrame
avg_weights_df = pd.DataFrame(grouped_weights.tolist(), index=grouped_weights.index)


# Assign factor names
factor_names = ['Mkt-RF', 'SMB', 'CMA', 'PEAD', 'QMJ', 'MGMT', 'PERF', 'LIQ', 'IFCR']
avg_weights_df.columns = factor_names
print(avg_weights_df)

# Group by PT Method and date
grouped = all_2008_df.groupby(['PT Method', all_2008_df.index])

# Mean over all strategy keys within each PT Method
agg_df = grouped[['Portfolio Returns', 'Expected Portfolio Returns']].mean().reset_index()

print(agg_df)

# Group by (Date, PT Method) to compute average weights per method per month
grouped = all_2008_df.groupby([all_2008_df.index, 'PT Method'])['Portfolio Weights']
avg_weights_by_period_method = grouped.apply(lambda x: np.mean(np.stack(x.values), axis=0))

# Convert to DataFrame with labeled columns
avg_weights_by_period_method_df = pd.DataFrame(avg_weights_by_period_method.tolist(), 
                                               index=avg_weights_by_period_method.index,
                                               columns=factor_names)

print(avg_weights_by_period_method_df.round(4))

# Get unique PT Methods
methods = agg_df['PT Method'].unique()

# Ensure the output folder exists
os.makedirs("deepdive-plots", exist_ok=True)

# Setup subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot: Realized Portfolio Returns
for method in agg_df['PT Method'].unique():
    method_df = agg_df[agg_df['PT Method'] == method]
    axs[0].plot(method_df['level_1'], method_df['Portfolio Returns'], label=method)
axs[0].set_title("2008: Realized Portfolio Returns by PT Method")
axs[0].set_ylabel("Return")
axs[0].grid(True)
axs[0].legend()

# Plot: Expected Portfolio Returns
for method in agg_df['PT Method'].unique():
    method_df = agg_df[agg_df['PT Method'] == method]
    axs[1].plot(method_df['level_1'], method_df['Expected Portfolio Returns'], '--', label=method)
axs[1].set_title("2008: Expected Portfolio Returns by PT Method")
axs[1].set_xlabel("Date")
axs[1].set_ylabel("Return")
axs[1].grid(True)
axs[1].legend()

# Formatting and save
plt.xticks(rotation=45)
plt.tight_layout()

save_path = os.path.join("deepdive-plots", "returns_vs_expected_subplots_PT_methods_2008.pdf")
plt.savefig(save_path, format='pdf')
plt.close()


folder = "./bma_cache2.0"
file_pattern = os.path.join(folder, "bma_pred_2008-*.pkl")
pkl_files = sorted(glob(file_pattern))

# Prepare storage
monthly_returns = {}
monthly_covariances = {}

# Load data
for filepath in pkl_files:
    with open(filepath, "rb") as f:
        data = pickle.load(f)
        filename = os.path.basename(filepath)
        month_label = filename.replace("bma_pred_", "").replace(".pkl", "")

        # Extract the first (and only) row of OOS return and covariance
        monthly_returns[month_label] = data['returns_OOS'].flatten()
        monthly_covariances[month_label] = data['covariance_matrix_OOS'][0]

# Convert returns to DataFrame
bma_predictions = pd.DataFrame.from_dict(monthly_returns, orient='index')
bma_predictions.index = pd.to_datetime(bma_predictions.index, format='%Y-%m')
bma_predictions.columns = ['Mkt-RF', 'SMB', 'CMA', 'PEAD', 'QMJ', 'MGMT', 'PERF', 'LIQ', 'IFCR']


# Fetch true returns:
realized_returns_df = po.load_historical_returns(parent_dir, start_date="2008-01-01", end_date="2008-12-31")



# Parameters
valid_months = sorted(monthly_returns.keys())
n_samples = 10000
weights = np.ones(9) / 9  # equal-weight portfolio
n_rows, n_cols = 6, 2

# Ensure avg_returns_by_date has a proper DatetimeIndex
avg_returns_by_date.index = pd.to_datetime(avg_returns_by_date.index)

# Regenerate the plot with expected returns taken from avg_returns_by_date['Expected Portfolio Returns']
fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, 18), sharex=False)
axs = axs.flatten()

for i, month in enumerate(valid_months):
    mu = monthly_returns[month]
    Sigma = monthly_covariances[month]
    samples = np.random.multivariate_normal(mu, Sigma, size=n_samples)
    portfolio_returns = samples @ weights

    # Retrieve both realized and expected returns from avg_returns_by_date
    realized_port_return = avg_returns_by_date.loc[month, 'Portfolio Returns']
    expected_port_return = avg_returns_by_date.loc[month, 'Expected Portfolio Returns']

    axs[i].hist(portfolio_returns, bins=40, density=True, alpha=0.7, color='skyblue', label='Simulated Returns')
    axs[i].axvline(expected_port_return.values[0], color='blue', linestyle='--', label='Expected Return')
    axs[i].axvline(realized_port_return.values[0], color='red', linestyle='-', label='Realized Return')
    axs[i].set_title(f"{month} - Portfolio Return Dist.")
    axs[i].legend()

# Hide unused subplots
for j in range(len(valid_months), len(axs)):
    axs[j].axis('off')

plt.tight_layout()
plt.savefig("./deepdive-plots/bma_portfolio_return_distributions_2008_grid.pdf", format="pdf")
plt.close()




realized_returns_df_full = po.load_historical_returns(parent_dir, start_date=start_date, end_date=end_date)

# Highlight the 2008 return values in each histogram
factors = ['Mkt-RF', 'SMB', 'CMA', 'PEAD', 'QMJ', 'MGMT', 'PERF', 'LIQ', 'IFCR']
# Filter 2008 observations
returns_2008 = realized_returns_df_full.loc["2008"]

# Adjust histogram layout to 3x3 instead of 1x9
n_factors = realized_returns_df_full.shape[1]
n_rows, n_cols = 3, 3

fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, 12), sharex=False)
axs = axs.flatten()

for i, col in enumerate(factors):
    all_values = realized_returns_df_full[col].dropna()
    values_2008 = returns_2008[col].dropna()
    mean_val = all_values.mean()

    axs[i].hist(all_values, bins=20, color='lightgrey', alpha=0.6, label="All Years")
    axs[i].hist(values_2008, bins=20, color='skyblue', alpha=0.9, label="2008")
    axs[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean = {mean_val:.4f}')

    axs[i].set_title(f"Distribution of Returns: {col}")
    axs[i].set_xlabel("Return")
    axs[i].set_ylabel("Frequency")
    axs[i].grid(True)
    axs[i].legend()

# Hide any unused subplots
for j in range(len(factors), len(axs)):
    axs[j].axis('off')

plt.tight_layout()
plt.savefig("./deepdive-plots/realized_return_distributions_by_factor_with_2008_overlay.pdf", format='pdf')
plt.close()


# Corrected plotting loop using 'Date' as x-axis
df_plot = avg_weights_by_period_method_df.reset_index()
df_plot['Date'] = pd.to_datetime(df_plot['level_0']).drop(columns='level_0')

# Create one plot per PT Method with proper labels and file names
for method in df_plot['PT Method'].unique():
    method_df = df_plot[df_plot['PT Method'] == method].copy()
    method_df = method_df.set_index('Date')  # Set the index to the date column
    method_df = method_df.drop(columns=['PT Method', 'level_0'])  # Drop the PT Method and level_0 columns
    ax = method_df[factor_names].plot(kind='bar', figsize=(12, 6))
    ax.set_ylabel("Weight")
    ax.set_xlabel("Month")
    ax.legend(title="Test asset", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(method_df.index.strftime('%b'), rotation=0)

    filename = f"./deepdive-plots/weights_{method.replace(' ', '_').replace('by_', 'by')}.pdf"
    plt.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.close()



breakpoint = 1
