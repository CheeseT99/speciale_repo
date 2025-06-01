import os
import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Directory containing cached BMA init files
cache_dir = "./bma_cache2.0"
plot_dir = "./bma_analysis_plots"
os.makedirs(plot_dir, exist_ok=True)
# Prepare storage for each cutoff group
cutoffs = [10, 100, 1000] 
cutoff_masses = {n: [] for n in cutoffs}
mispricing_probs = []
n90_list = []  # ← NEW: track # of models for 90% posterior mass
top1_probs = []

# Loop through all BMA files
for file in sorted(os.listdir(cache_dir)):
    if file.startswith("bma_init_") and file.endswith(".pkl"):
        month_str = file.replace("bma_init_", "").replace(".pkl", "")
        file_path = os.path.join(cache_dir, file)

        try:
            with open(file_path, "rb") as f:
                bma_dict = pkl.load(f)

            # Extract posterior probabilities and log marginal likelihoods
            CML = bma_dict["CMLCombined"]
            CLMLU = bma_dict["CLMLU"]
            CLMLR = bma_dict["CLMLR"]

            df_models = pd.DataFrame({
                "Model Index": np.arange(len(CML)),
                "Log Marg Lik": np.concatenate([CLMLU, CLMLR]),
                "Posterior Prob": CML,
                "Model Type": ["Unrestricted"] * len(CLMLU) + ["Restricted"] * len(CLMLR)
            })

            # Sort and compute cumulative mass
            top_models = df_models.sort_values("Posterior Prob", ascending=False).reset_index(drop=True)
            top_models["Cumulative Mass"] = top_models["Posterior Prob"].cumsum()

            # Track cumulative masses for specific cutoffs
            for n in cutoffs:
                if n <= len(top_models):
                    mass = top_models.loc[n - 1, "Cumulative Mass"]
                    cutoff_masses[n].append(mass)

            # NEW: how many models to reach 90%?
            n90 = (top_models["Cumulative Mass"] < 0.90).sum() + 1
            n90_list.append(n90)

            # Sort and compute cumulative mass
            top_models = df_models.sort_values("Posterior Prob", ascending=False).reset_index(drop=True)
            top_models["Cumulative Mass"] = top_models["Posterior Prob"].cumsum()

            # Store posterior prob of the top 1 model
            top1_probs.append(top_models.loc[0, "Posterior Prob"])

            # Record mispricing probability
            prob_mispricing = np.sum(CML[:len(CLMLU)])
            mispricing_probs.append(prob_mispricing)

        except Exception as e:
            print(f"Skipping {file} due to error: {e}")

# ========================
# Print the average results
# ========================
print("\n=== Average Posterior Mass Across All Months ===")
for n in cutoffs:
    avg_mass = np.mean(cutoff_masses[n])
    print(f"Top {n:>3} models account for {avg_mass:.2%} of posterior mass on average")

avg_n90 = np.mean(n90_list)
print(f"\nAverage number of models needed to reach 90% posterior mass: {avg_n90:.0f}")

avg_top1 = np.mean(top1_probs)
print(f"\nAverage posterior probability of the top 1 model: {avg_top1:.4%}")


avg_mispricing = np.mean(mispricing_probs)
avg_nomispricing = 1 - avg_mispricing
print(f"\nAverage Probability of Mispricing:    {avg_mispricing:.4f}")
print(f"Average Probability of No Mispricing: {avg_nomispricing:.4f}")


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

# Re-run the loop to construct the mispricing time series and plot it
cache_dir = "./bma_cache2.0"
results = []

try:
    for file in sorted(os.listdir(cache_dir)):
        if file.startswith("bma_init_") and file.endswith(".pkl"):
            date_str = file.replace("bma_init_", "").replace(".pkl", "")
            with open(os.path.join(cache_dir, file), "rb") as f:
                bma_dict = pkl.load(f)

            CML = bma_dict["CMLCombined"]
            CLMLU = bma_dict["CLMLU"]
            prob_mispricing = np.sum(CML[:len(CLMLU)])

            results.append((pd.to_datetime(date_str), prob_mispricing))

    # Create DataFrame and plot
    df = pd.DataFrame(results, columns=["Date", "ProbMispricing"]).set_index("Date")
    df.sort_index(inplace=True)
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["ProbMispricing"], label="Unrestricted Models")
    plt.plot(df.index, 1 - df["ProbMispricing"], label="Restricted Models", linestyle='--')
    plt.xlabel("Date")
    plt.ylabel("Posterior Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "posterior_mispricing_distribution.png"))
    plt.close()

except FileNotFoundError:
    print("Folder './bma_cache2.0' not found. Please upload or specify the correct path.")






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
fig, axs = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
df_post['Entropy'].plot(ax=axs[0], title='Posterior Entropy Over Time')
df_post['TopProb'].plot(ax=axs[1], title='Top Model Posterior Probability Over Time')
df_post['TopN90Mass'].plot(ax=axs[2], title='Number of Models for 90% Posterior Mass')
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

# Plot PEAD inclusion probability over time
if not factor_df.empty and 3 in factor_df.columns:
    plt.figure(figsize=(14, 6))
    plt.plot(factor_df.index, factor_df[3], label="PEAD", color="tab:blue")
    plt.title("Inclusion Probability of PEAD Over Time")
    plt.xlabel("Date")
    plt.ylabel("Inclusion Probability")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "pead_inclusion_timeseries.png"))
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
    'Mean Probability': factor_df.mean(),
    'Std. Dev.': factor_df.std()
}).round(2).sort_values("Mean Probability", ascending=False)

predictor_summary_named = pd.DataFrame({
    'Mean Probability': predictor_df.mean(),
    'Std. Dev.': predictor_df.std()
}).round(2).sort_values("Mean Probability", ascending=False)

print(factor_summary_named)
print(predictor_summary_named)

factor_summary_named.to_csv(os.path.join(plot_dir, "factor_inclusion_summary_named.csv"))
predictor_summary_named.to_csv(os.path.join(plot_dir, "predictor_inclusion_summary_named.csv"))
