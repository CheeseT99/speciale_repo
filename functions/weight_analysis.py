import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def average_weights(results_dict):
    avg_weights_dict = {}

    for key, df in results_dict.items():
        weights_df = pd.DataFrame(df['Portfolio Weights'].tolist(), index=df.index)
        avg_weights_dict[key] = weights_df.mean()

    return pd.DataFrame(avg_weights_dict).T  # rows = strategy, columns = assets




def weight_volatility_and_turnover(results_dict):
    """
    Calculate average standard deviation of weights and turnover for each strategy.
    Hence, 
    Avg Std Dev (weights): "How much do the weights of each asset fluctuate over time?"
    Avg Turnover: "How much does the portfolio change from one month to the next?" 
    
    Low turnover: strategy trades infrequently → lower transaction costs
    High turnover: strategy trades frequently → may exploit time-varying opportunities

    It achieves: 
    Emotionally reactive the investor is (based on λ, γ)
    Model uncertainty translates into allocation behavior
    """

    stats = []

    for key, df in results_dict.items():
        weights_df = pd.DataFrame(df['Portfolio Weights'].tolist(), index=df.index)

        std_per_asset = weights_df.std().mean()  # mean std across assets
        turnover = weights_df.diff().abs().sum(axis=1)
        avg_turnover = turnover.mean()

        stats.append({
            'Strategy_Key': key,
            'Avg Std Dev (weights)': std_per_asset,
            'Avg Turnover': avg_turnover
        })

    return pd.DataFrame(stats).set_index('Strategy_Key')

def herfindahl_index(results_dict):
    """
    Calculate Herfindahl-Hirschman Index (HHI) for each strategy and return:
    - Per-strategy summary (Avg HHI and Max HHI)
    - Pivot tables showing Avg HHI per (λ, γ) for conservative and aggressive

    Returns:
        hhi_df (pd.DataFrame): Strategy-level HHI summary
        pivot_cons (pd.DataFrame): Table of Avg HHI (conservative) [λ x γ]
        pivot_agg (pd.DataFrame): Table of Avg HHI (aggressive) [λ x γ]
    
    It's basically asking:
    "How concentrated is the portfolio in just a few assets?"

    Interpretation:
    * Low HHI (~1/N): well-diversified, equal-weighted portfolio
    * High HHI (~1.0): concentrated in a few assets
    * Maximum HHI = 1.0: all weight in a single asset
    * Minimum HHI = 1/N: perfectly equal weights
    """


    hhi_rows = []

    for key, df in results_dict.items():
        weights_df = pd.DataFrame(df['Portfolio Weights'].tolist(), index=df.index)
        hhi = (weights_df ** 2).sum(axis=1)

        # Parse strategy, lambda, gamma
        parts = key.split('_')
        strategy = parts[0]
        lam = float(parts[1])
        gamma = float(parts[2])

        hhi_rows.append({
            'Strategy_Key': key,
            'Strategy': strategy,
            'Lambda': lam,
            'Gamma': gamma,
            'Avg HHI': hhi.mean(),
            'Max HHI': hhi.max()
        })

    hhi_df = pd.DataFrame(hhi_rows).set_index('Strategy_Key')

    # Create pivot tables: one for each strategy
    pivot_cons = hhi_df[hhi_df['Strategy'] == 'conservative'].pivot_table(
        index='Lambda', columns='Gamma', values='Avg HHI'
    ).sort_index()

    pivot_agg = hhi_df[hhi_df['Strategy'] == 'aggressive'].pivot_table(
        index='Lambda', columns='Gamma', values='Avg HHI'
    ).sort_index()

    return hhi_df, pivot_cons, pivot_agg



def plot_sharpe_vs_hhi(summary_df, hhi_df, parent_dir, metric='Sharpe Ratio'):
    """
    Plots Sharpe Ratio (or other metric) vs Average HHI with a regression line,
    colored by strategy type. Saves plot to disk.

    Args:
        summary_df (pd.DataFrame): Output from summarize_backtest_results()
        hhi_df (pd.DataFrame): Output from herfindahl_index(), with Strategy_Key index
        parent_dir (str): Folder to save the plot
        metric (str): Performance metric to compare against HHI (default = 'Sharpe Ratio')

    Returns:
        str: Path to saved plot

    This function basically asks:
        "Are more concentrated portfolios delivering better risk-adjusted returns?"
    """

    # Merge summary and HHI data
    merged = summary_df.join(hhi_df[['Avg HHI']], how='inner')
    merged['Strategy'] = merged.index.str.split('_').str[0]
    merged['Lambda'] = merged.index.str.split('_').str[1]
    merged['Gamma'] = merged.index.str.split('_').str[2]

    # Plot
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=merged,
        x='Avg HHI',
        y=metric,
        hue='Strategy',
        style='Strategy',
        s=100
    )

    # Regression line
    sns.regplot(
        data=merged,
        x='Avg HHI',
        y=metric,
        scatter=False,
        color='gray',
        line_kws={'linewidth': 2, 'label': 'OLS Fit'}
    )

    # Annotate ONLY conservative strategies
    for _, row in merged.iterrows():
        if row['Strategy'] == 'conservative':
            jitter_x = row['Avg HHI'] + np.random.uniform(-0.0015, 0.0015)
            jitter_y = row[metric] + np.random.uniform(-0.5, 0.5)
            label = f"λ={row['Lambda']}, γ={row['Gamma']}"
            plt.text(jitter_x, jitter_y, label, fontsize=7, alpha=0.7)

    plt.title(f"{metric} vs Portfolio Concentration (HHI)")
    plt.xlabel("Average HHI (Portfolio Concentration)")
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plots_dir = os.path.join(parent_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    file_path = os.path.join(plots_dir, f"{metric.replace(' ', '_')}_vs_HHI.png")
    plt.savefig(file_path)
    plt.close()

    return file_path




def save_hhi_heatmaps(pivot_cons, pivot_agg, parent_dir):
    """
    Saves HHI heatmaps (λ x γ) for conservative and aggressive strategies.
    """

    plots_dir = os.path.join(parent_dir, "plots", "hhi")
    os.makedirs(plots_dir, exist_ok=True)
    
    for pivot, label in [(pivot_cons, "conservative"), (pivot_agg, "aggressive")]:
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="coolwarm", cbar_kws={"label": "Avg HHI"})
        plt.title(f"Herfindahl Index Heatmap — {label.title()}")
        plt.xlabel("Gamma")
        plt.ylabel("Lambda")
        plt.tight_layout()

        file_path = os.path.join(plots_dir, f"hhi_heatmap_{label}.png")
        plt.savefig(file_path)
        plt.close()

        print(f"✅ Saved: {file_path}")



def save_weight_heatmaps(results_dict, parent_dir):
    plot_dir = os.path.join(parent_dir, "plots", "weights")
    os.makedirs(plot_dir, exist_ok=True)

    saved = []
    for key, df in results_dict.items():
        weights_df = pd.DataFrame(df['Portfolio Weights'].tolist(), index=df.index)
        plt.figure(figsize=(10, 6))
        sns.heatmap(weights_df.T, cmap="coolwarm", center=0)
        plt.title(f"Weight Heatmap: {key}")
        plt.xlabel("Time")
        plt.ylabel("Assets")
        path = os.path.join(plot_dir, f"heatmap_weights_{key}.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        saved.append(path)
    return saved


def top_assets_per_strategy(results_dict, top_n=3):
    top_assets = {}

    for key, df in results_dict.items():
        weights_df = pd.DataFrame(df['Portfolio Weights'].tolist(), index=df.index)
        mean_weights = weights_df.mean()
        top_assets[key] = mean_weights.sort_values(ascending=False).head(top_n)

    return pd.DataFrame(top_assets).T  # strategies x top assets
