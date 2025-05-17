import os
import pickle as pkl
import pandas as pd
import numpy as np

# Example: Load a specific estimation file
cache_dir = './bma_cache2.0'
month_str = '2006-06'  # Example date
file_path = os.path.join(cache_dir, f"bma_init_{month_str}.pkl")

with open(file_path, "rb") as f:
    bma_dict = pkl.load(f)


# Extract posterior probabilities and log marginal likelihoods
CML = bma_dict["CMLCombined"]
CLMLU = bma_dict["CLMLU"]
CLMLR = bma_dict["CLMLR"]

# Generate a DataFrame of models
df_models = pd.DataFrame({
    "Model Index": np.arange(len(CML)),
    "Log Marg Lik": np.concatenate([CLMLU, CLMLR]),
    "Posterior Prob": CML,
    "Model Type": ["Unrestricted"] * len(CLMLU) + ["Restricted"] * len(CLMLR)
})

# Sort by posterior probability
top_models = df_models.sort_values("Posterior Prob", ascending=False).reset_index(drop=True)

# Display top 10 models
print(top_models.head(10))


# Add cumulative sum
top_models["Cumulative Mass"] = top_models["Posterior Prob"].cumsum()

# How many models capture 90% of total probability?
n_top = (top_models["Cumulative Mass"] < 0.90).sum() + 1
print(f"Top {n_top} models account for 90% of posterior mass")


prob_mispricing = np.sum(CML[:len(CLMLU)])
prob_no_mispricing = np.sum(CML[len(CLMLU):])
print(f"Probability of mispricing    = {prob_mispricing:.4f}")
print(f"Probability of no mispricing = {prob_no_mispricing:.4f}")

results = []

for file in sorted(os.listdir(cache_dir)):
    if file.startswith("bma_init_") and file.endswith(".pkl"):
        date_str = file.replace("bma_init_", "").replace(".pkl", "")
        with open(os.path.join(cache_dir, file), "rb") as f:
            bma_dict = pkl.load(f)

        CML = bma_dict["CMLCombined"]
        CLMLU = bma_dict["CLMLU"]
        prob_mispricing = np.sum(CML[:len(CLMLU)])

        results.append((pd.to_datetime(date_str), prob_mispricing))

df = pd.DataFrame(results, columns=["Date", "ProbMispricing"]).set_index("Date")
df.sort_index(inplace=True)
df.plot(title="Probability of Mispricing Over Time")

print(df)

import os
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing cached BMA init files
cache_dir = "./bma_cache2.0"
plot_dir = "./bma_analysis_plots"
os.makedirs(plot_dir, exist_ok=True)

# Prepare storage
records = []
factor_inclusion_time = []
predictor_inclusion_time = []

# Discover all relevant files in the directory
uploaded_files = [
    os.path.join(cache_dir, f)
    for f in os.listdir(cache_dir)
    if f.startswith("bma_init_") and f.endswith(".pkl")
]

# Loop through uploaded files
for file_path in sorted(uploaded_files):
    date_str = os.path.basename(file_path).replace("bma_init_", "").replace(".pkl", "")
    date = pd.to_datetime(date_str)

    try:
        with open(file_path, "rb") as f:
            bma_dict = pkl.load(f)

        # Unpack contents
        CML = bma_dict["CMLCombined"]
        CLMLU = bma_dict["CLMLU"]
        CLMLR = bma_dict["CLMLR"]
        factorsProb = bma_dict.get("factorsProbability", None)
        predictorsProb = bma_dict.get("predictorsProbability", None)

        # Posterior stats
        entropy = -np.sum(CML * np.log(CML + 1e-12))
        top_prob = np.max(CML)
        cum_mass = np.sort(CML)[::-1].cumsum()
        n90 = np.searchsorted(cum_mass, 0.9) + 1
        prob_mispricing = np.sum(CML[:len(CLMLU)])

        # Record
        records.append({
            "Date": date,
            "Entropy": entropy,
            "TopProb": top_prob,
            "TopN90Mass": n90,
            "ProbMispricing": prob_mispricing,
        })

        # Store inclusion probabilities
        if factorsProb is not None and predictorsProb is not None:
            factor_inclusion_time.append(pd.Series(factorsProb, name=date))
            predictor_inclusion_time.append(pd.Series(predictorsProb, name=date))

    except Exception as e:
        print(f"Skipping {file_path} due to error: {e}")

# Create dataframes
df_post = pd.DataFrame(records).set_index("Date").sort_index()
factor_df = pd.DataFrame(factor_inclusion_time).sort_index()
predictor_df = pd.DataFrame(predictor_inclusion_time).sort_index()

# Plot time series
fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
df_post['Entropy'].plot(ax=axs[0], title='Posterior Entropy Over Time')
df_post['TopProb'].plot(ax=axs[1], title='Top Model Posterior Probability Over Time')
df_post['TopN90Mass'].plot(ax=axs[2], title='Number of Models for 90% Posterior Mass')
df_post['ProbMispricing'].plot(ax=axs[3], title='Posterior Probability of Mispricing (Unrestricted Models)')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "posterior_summary_plots.png"))
plt.close()



# Heatmap: Factor inclusion probabilities
if not factor_df.empty:
    plt.figure(figsize=(14, 8))
    plt.imshow(factor_df.T, aspect='auto', interpolation='none', cmap='viridis')
    plt.colorbar(label="Inclusion Probability")
    plt.yticks(range(factor_df.shape[1]), labels=factor_df.columns, fontsize=8)
    plt.xticks(range(len(factor_df.index)), labels=factor_df.index.strftime('%Y-%m'), rotation=90, fontsize=6)
    plt.title("Factor Inclusion Probabilities Over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "factor_inclusion_heatmap.png"))
    plt.close()

# Heatmap: Predictor inclusion probabilities
if not predictor_df.empty:
    plt.figure(figsize=(14, 8))
    plt.imshow(predictor_df.T, aspect='auto', interpolation='none', cmap='plasma')
    plt.colorbar(label="Inclusion Probability")
    plt.yticks(range(predictor_df.shape[1]), labels=predictor_df.columns, fontsize=8)
    plt.xticks(range(len(predictor_df.index)), labels=predictor_df.index.strftime('%Y-%m'), rotation=90, fontsize=6)
    plt.title("Predictor Inclusion Probabilities Over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "predictor_inclusion_heatmap.png"))
    plt.close()


# Plot factor inclusion probabilities over time
if not factor_df.empty:
    plt.figure(figsize=(14, 8))
    for col in factor_df.columns:
        plt.plot(factor_df.index, factor_df[col], label=col)
    plt.title("Factor Inclusion Probabilities Over Time")
    plt.xlabel("Date")
    plt.ylabel("Inclusion Probability")
    plt.legend(loc='upper right', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "factor_inclusion_timeseries.png"))
    plt.close()

# Plot predictor inclusion probabilities over time
if not predictor_df.empty:
    plt.figure(figsize=(14, 8))
    for col in predictor_df.columns:
        plt.plot(predictor_df.index, predictor_df[col], label=col)
    plt.title("Predictor Inclusion Probabilities Over Time")
    plt.xlabel("Date")
    plt.ylabel("Inclusion Probability")
    plt.legend(loc='upper right', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "predictor_inclusion_timeseries.png"))
    plt.close()

# Compute summary stats for reporting
factor_summary = pd.DataFrame({
    'Mean': factor_df.mean(),
    'StdDev': factor_df.std()
}).sort_values("StdDev", ascending=False)

predictor_summary = pd.DataFrame({
    'Mean': predictor_df.mean(),
    'StdDev': predictor_df.std()
}).sort_values("StdDev", ascending=False)

# Save as CSV for reference or LaTeX export
factor_summary.to_csv(os.path.join(plot_dir, "factor_inclusion_summary.csv"))
predictor_summary.to_csv(os.path.join(plot_dir, "predictor_inclusion_summary.csv"))

# Define explicit variable names
factor_names = ['Mkt-RF', 'SMB', 'CMA', 'PEAD', 'QMJ', 'MGMT', 'PERF', 'LIQ', 'IFCR']
predictor_names = ['dp', 'dy', 'ep', 'de', 'svar', 'ntis', 'lbl', 'lty', 'dfy']

# Assign names to the columns if the shape matches
if factor_df.shape[1] == len(factor_names):
    factor_df.columns = factor_names

if predictor_df.shape[1] == len(predictor_names):
    predictor_df.columns = predictor_names

# Save renamed versions and re-generate plots
# Heatmap: Factor inclusion probabilities
if not factor_df.empty:
    plt.figure(figsize=(14, 8))
    plt.imshow(factor_df.T, aspect='auto', interpolation='none', cmap='viridis')
    plt.colorbar(label="Inclusion Probability")
    plt.yticks(range(factor_df.shape[1]), labels=factor_df.columns, fontsize=8)
    plt.xticks(range(len(factor_df.index)), labels=factor_df.index.strftime('%Y-%m'), rotation=90, fontsize=6)
    plt.title("Factor Inclusion Probabilities Over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "factor_inclusion_heatmap_named.png"))
    plt.close()

# Heatmap: Predictor inclusion probabilities
if not predictor_df.empty:
    plt.figure(figsize=(14, 8))
    plt.imshow(predictor_df.T, aspect='auto', interpolation='none', cmap='plasma')
    plt.colorbar(label="Inclusion Probability")
    plt.yticks(range(predictor_df.shape[1]), labels=predictor_df.columns, fontsize=8)
    plt.xticks(range(len(predictor_df.index)), labels=predictor_df.index.strftime('%Y-%m'), rotation=90, fontsize=6)
    plt.title("Predictor Inclusion Probabilities Over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "predictor_inclusion_heatmap_named.png"))
    plt.close()

# Regenerate only the named time series plots for inclusion probabilities

# Factor time series with names
plt.figure(figsize=(14, 8))
for col in factor_df.columns:
    plt.plot(factor_df.index, factor_df[col], label=col)
plt.title("Factor Inclusion Probabilities Over Time")
plt.xlabel("Date")
plt.ylabel("Inclusion Probability")
plt.legend(loc='upper right', fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "factor_inclusion_timeseries_named.png"))
plt.close()

# Predictor time series with names
plt.figure(figsize=(14, 8))
for col in predictor_df.columns:
    plt.plot(predictor_df.index, predictor_df[col], label=col)
plt.title("Predictor Inclusion Probabilities Over Time")
plt.xlabel("Date")
plt.ylabel("Inclusion Probability")
plt.legend(loc='upper right', fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "predictor_inclusion_timeseries_named.png"))
plt.close()


# Save updated versions of the summary stats
factor_summary_named = pd.DataFrame({
    'Mean': factor_df.mean(),
    'StdDev': factor_df.std()
}).sort_values("StdDev", ascending=False)

predictor_summary_named = pd.DataFrame({
    'Mean': predictor_df.mean(),
    'StdDev': predictor_df.std()
}).sort_values("StdDev", ascending=False)

factor_summary_named.to_csv(os.path.join(plot_dir, "factor_inclusion_summary_named.csv"))
predictor_summary_named.to_csv(os.path.join(plot_dir, "predictor_inclusion_summary_named.csv"))
