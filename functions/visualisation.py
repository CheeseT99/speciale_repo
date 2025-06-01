import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_strategies_vs_market(summary_dict: dict, market_cumulative: pd.Series, parent_dir: str, title="Strategy vs Market") -> str:
    """
    Saves a line plot of cumulative strategy returns vs market benchmark.

    Args:
        summary_dict (dict): Dictionary of backtest results keyed by strategy names
        market_cumulative (pd.Series): Benchmark cumulative returns (e.g., Mkt-RF)
        parent_dir (str): Root directory where 'plots' folder will be created
        title (str): Plot title

    Returns:
        str: Path to saved plot
    """
    # Create plot folder
    plots_dir = os.path.join(parent_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot
    plt.figure(figsize=(10, 6))
    
    for key, df in summary_dict.items():
        compounded = df['Compounded Returns']
        plt.plot(compounded.index, compounded, label=key)

    # Align market index
    aligned_market = market_cumulative.loc[
        market_cumulative.index >= compounded.index.min()
    ]
    aligned_market = aligned_market.loc[
        aligned_market.index <= compounded.index.max()
    ]
    plt.plot(aligned_market.index, aligned_market, label='Market (Mkt-RF)', color='black', linewidth=2, linestyle='--')

    # Decorate
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Wealth")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save
    file_path = os.path.join(plots_dir, "strategy_vs_market.png")
    plt.savefig(file_path)
    plt.close()

    return file_path


def plot_best_strategy_vs_market(
    summary_df: pd.DataFrame,
    results_dict: dict,
    market_cumulative: pd.Series,
    parent_dir: str,
    metric: str = 'Final Wealth'
) -> str:
    """
    Finds and plots the best-performing strategy profile vs the market, 
    including the metric value in the legend.

    Returns:
        str: Path to saved PNG file
    """
    if metric not in summary_df.columns:
        raise ValueError(f"Metric '{metric}' not found in summary DataFrame.")

    # Identify best strategy
    best_key = summary_df[metric].idxmax()
    best_row = summary_df.loc[best_key]
    best_df = results_dict[best_key]

    strategy = best_row['Strategy']
    lam = best_row['Lambda']
    gamma = best_row['Gamma']
    value = best_row[metric]

    plots_dir = os.path.join(parent_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(
        best_df.index,
        best_df['Compounded Returns'],
        label=f"Best Strategy ({strategy}, λ={lam}, γ={gamma})\n{metric} = {value:.4f}",
        linewidth=2
    )

    aligned_market = market_cumulative.loc[
        market_cumulative.index >= best_df.index.min()
    ]
    aligned_market = aligned_market.loc[
        aligned_market.index <= best_df.index.max()
    ]
    plt.plot(
        aligned_market.index,
        aligned_market,
        label='Market (Mkt-RF)',
        color='black',
        linestyle='--',
        linewidth=2
    )

    plt.title(f"Best Strategy vs Market — by {metric}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Wealth")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    file_path = os.path.join(plots_dir, f"best_strategy_vs_market_{metric.replace(' ', '_')}.png")
    plt.savefig(file_path)
    plt.close()

    return file_path


def plot_best_and_worst_strategy_vs_market(
    summary_df: pd.DataFrame,
    results_dict: dict,
    market_cumulative: pd.Series,
    parent_dir: str,
    metric: str = 'Final Wealth'
) -> str:
    """
    Plots and saves a comparison of the best and worst strategy vs the market
    using the given performance metric.

    Returns:
        str: Path to saved PNG file
    """
    if metric not in summary_df.columns:
        raise ValueError(f"Metric '{metric}' not found in summary DataFrame.")

    # Identify best and worst
    best_key = summary_df[metric].idxmax()
    worst_key = summary_df[metric].idxmin()

    best_row = summary_df.loc[best_key]
    worst_row = summary_df.loc[worst_key]

    best_df = results_dict[best_key]
    worst_df = results_dict[worst_key]

    # Market alignment
    start_idx = min(best_df.index.min(), worst_df.index.min())
    end_idx = max(best_df.index.max(), worst_df.index.max())
    aligned_market = market_cumulative.loc[start_idx:end_idx]

    # Prepare figure
    plots_dir = os.path.join(parent_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # --- Best strategy ---
    ax[0].plot(best_df.index, best_df['Compounded Returns'], label=f"{best_key}\n{metric}: {best_row[metric]:.4f}", linewidth=2)
    ax[0].plot(aligned_market.index, aligned_market, label='Market (Mkt-RF)', color='black', linestyle='--')
    ax[0].set_title("Best Strategy")
    ax[0].legend()
    ax[0].grid(True)
    ax[0].set_xlabel("Date")
    ax[0].set_ylabel("Cumulative Wealth")

    # --- Worst strategy ---
    ax[1].plot(worst_df.index, worst_df['Compounded Returns'], label=f"{worst_key}\n{metric}: {worst_row[metric]:.4f}", linewidth=2, color='tomato')
    ax[1].plot(aligned_market.index, aligned_market, label='Market (Mkt-RF)', color='black', linestyle='--')
    ax[1].set_title("Worst Strategy")
    ax[1].legend()
    ax[1].grid(True)
    ax[1].set_xlabel("Date")

    plt.suptitle(f"Best vs. Worst Strategy — by {metric}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    file_path = os.path.join(plots_dir, f"best_vs_worst_{metric.replace(' ', '_')}.png")
    plt.savefig(file_path)
    plt.close()

    return file_path


def plot_all_heatmaps(summary_df: pd.DataFrame, metrics: list[str], parent_dir='./') -> list[str]:
    """
    Generates and saves heatmaps for multiple performance metrics across (λ, γ) per strategy.

    Args:
        summary_df (pd.DataFrame): Output from summarize_backtest_results()
        metrics (list of str): List of columns to plot (e.g., ['Sharpe Ratio', 'Final Wealth'])
        parent_dir (str): Where to create the 'plots' folder

    Returns:
        List[str]: Paths to all saved plots
    """
    saved_paths = []
    plots_dir = os.path.join(parent_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    strategies = summary_df['Strategy'].unique()

    for strategy in strategies:
        data = summary_df[summary_df['Strategy'] == strategy]

        for metric in metrics:
            if metric not in data.columns:
                print(f"⚠️ Metric '{metric}' not found in summary — skipping.")
                continue

            pivot = data.pivot(index='Gamma', columns='Lambda', values=metric)
            pivot = pivot.sort_index(ascending=True)

            plt.figure(figsize=(8, 6))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", cbar_kws={"label": metric})
            plt.title(f"{strategy.title()} — {metric}")
            plt.xlabel("Lambda (Loss Aversion)")
            plt.ylabel("Gamma (Risk Aversion)")
            plt.tight_layout()

            file_name = f"{strategy}_{metric.replace(' ', '_')}.png"
            save_path = os.path.join(plots_dir, file_name)
            plt.savefig(save_path)
            plt.close()

            saved_paths.append(save_path)

    return saved_paths



def plot_certainty_equivalent_heatmap(ce_df: pd.DataFrame, parent_dir: str) -> str:
    """
    Plots a heatmap of Certainty Equivalent across (λ, γ) grid.

    Args:
        ce_df (pd.DataFrame): Output from compute_certainty_equivalents()
        parent_dir (str): Path to your project root for saving the plot

    Returns:
        str: File path to saved heatmap
    """
    # Pivot to matrix: rows = Lambda, columns = Gamma
    pivot = ce_df.reset_index().pivot_table(
        index='Lambda',
        columns='Gamma',
        values='Certainty Equivalent'
    ).sort_index(ascending=True)

    # Plot setup
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={"label": "Certainty Equivalent"})
    plt.title("Certainty Equivalent Utility — (λ, γ) Grid")
    plt.xlabel("Gamma (Risk Aversion)")
    plt.ylabel("Lambda (Loss Aversion)")
    plt.tight_layout()

    # Save
    plots_dir = os.path.join(parent_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    file_path = os.path.join(plots_dir, "certainty_equivalent_heatmap.png")
    plt.savefig(file_path)
    plt.close()

    return file_path


def plot_certainty_equivalent_comparison(ce_df: pd.DataFrame, methods=('BMA', 'Historical Mean', 'MVP'), save_path: str = None):
    """
    Plots Certainty Equivalent comparison across methods for each strategy.

    Args:
        ce_df: DataFrame with columns ['Strategy_Key', 'Strategy', 'Lambda', 'Gamma', 'Method', 'Certainty Equivalent']
        methods: list of method names to include (must match 'Method' column)
        save_path: optional path to save figure
    """
    # Filter for relevant methods
    ce_filtered = ce_df[ce_df['Method'].isin(methods)]

    # Set up plot
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=ce_filtered,
        x='Strategy_Key',
        y='Certainty Equivalent',
        hue='Method',
        dodge=True
    )

    plt.xticks(rotation=90)
    plt.xlabel("Strategy (Lambda, Gamma)")
    plt.ylabel("Certainty Equivalent")
    plt.title("Certainty Equivalent Comparison Across Methods")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"✅ Saved plot to {save_path}")
    else:
        plt.show()



def plot_ce_and_sharpe_comparison(merged_df: pd.DataFrame, save_path: str = None):
    """
    Creates a side-by-side barplot of Certainty Equivalent and Sharpe Ratio across methods.

    Args:
        merged_df: DataFrame containing 'Method', 'Certainty Equivalent', and 'Sharpe Ratio'.
        save_path: optional path to save the plot.
    """

    # Group by Method and calculate mean CE and Sharpe
    ce_by_method = merged_df.groupby('Method')['Certainty Equivalent'].mean()
    sharpe_by_method = merged_df.groupby('Method')['Sharpe Ratio'].mean()

    methods = ce_by_method.index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # === Left: Certainty Equivalent
    sns.barplot(x=ce_by_method.values, y=methods, ax=axes[0], orient='h', palette="Blues_d")
    axes[0].set_title("Average Certainty Equivalent by Method")
    axes[0].set_xlabel("Certainty Equivalent")
    axes[0].set_ylabel("Method")
    axes[0].grid(True)

    # === Right: Sharpe Ratio
    sns.barplot(x=sharpe_by_method.values, y=methods, ax=axes[1], orient='h', palette="Greens_d")
    axes[1].set_title("Average Sharpe Ratio by Method")
    axes[1].set_xlabel("Sharpe Ratio")
    axes[1].set_ylabel("")

    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"✅ Saved plot to: {save_path}")
    else:
        plt.show()


