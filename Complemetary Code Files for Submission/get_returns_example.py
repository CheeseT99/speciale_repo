from conditionalAssetPricingLogMarginalLikelihoodTauClass import Model
import pandas as pd
# Load your data

# === Load your datasets ===
returns = pd.read_csv(r"C:\Users\Tor Osted\OneDrive\Dokumenter\GitHub\speciale_repo\Complemetary Code Files for Submission\Data\Simulated_Returns_Data.csv")
factors = pd.read_csv(r"C:\Users\Tor Osted\OneDrive\Dokumenter\GitHub\speciale_repo\Complemetary Code Files for Submission\Data\factors-20.csv")
predictors = pd.read_csv(r'C:\Users\Tor Osted\OneDrive\Dokumenter\GitHub\speciale_repo\Complemetary Code Files for Submission\Data\Z - 197706.csv')

# === Step 1: Convert 'Date' columns to datetime format ===
returns['Date'] = pd.to_datetime(returns['Date'])
factors['Date'] = pd.to_datetime(factors['Date'])
predictors['Date'] = pd.to_datetime(predictors['Date'])

# === Step 2: Find common date range ===
start_date = max(returns['Date'].min(), factors['Date'].min(), predictors['Date'].min())
end_date = min(returns['Date'].max(), factors['Date'].max(), predictors['Date'].max())

print("Common date range:", start_date.date(), "to", end_date.date())

# === Step 3: Filter all datasets to this common range ===
returns = returns[(returns['Date'] >= start_date) & (returns['Date'] <= end_date)].reset_index(drop=True)
factors = factors[(factors['Date'] >= start_date) & (factors['Date'] <= end_date)].reset_index(drop=True)
predictors = predictors[(predictors['Date'] >= start_date) & (predictors['Date'] <= end_date)].reset_index(drop=True)
# Assume preprocessing done: 'Date' column in each, proper alignment, etc.

# Define inputs
Tau = 1.5
significant_predictors = list(range(predictors.shape[1] - 1))  # skip 'Date'
index_end_of_estimation = 246  # Optional

# Instantiate the model
model = Model(rr=returns, ff=factors, zz=predictors,
              significantPredictors=significant_predictors,
              Tau=Tau,
              indexEndOfEstimation=index_end_of_estimation,
              key_demean_predictors=True)

# Run the expected return calculation using the single most probable model
#result = model.conditionalAssetPricingSingleOOSTauNumba(single_top_model=1)
result = model.conditionalAssetPricingOOSTauNumba()

# Extract the expected return vector for the next period
expected_returns = result[0]  # returns_OOS

