import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


"""
Overview of functions:
Computes the Certainty Equivalent (CE) using Prospect Theory for a given set of strategies.
1. prospect_theory_value
2. calculate_certainty_equivalent
3. compute_certainty_equivalents
"""

def prospect_theory_value(r: float, reference: float, lambda_: float, gamma: float) -> float:
    delta = r - reference
    if delta >= 0:
        return (delta ** (1 - gamma)) / (1 - gamma) if gamma != 1 else np.log(1 + delta)
    else:
        return -lambda_ * ((-delta) ** (1 - gamma)) / (1 - gamma) if gamma != 1 else -lambda_ * np.log(1 - delta)

def calculate_certainty_equivalent(returns: pd.Series, lambda_: float, gamma: float, reference: float = 0.0) -> float:
    utilities = [
        prospect_theory_value(r, reference=reference, lambda_=lambda_, gamma=gamma)
        for r in returns
    ]
    avg_utility = np.mean(utilities)

    # Invert utility function to get certainty equivalent
    if avg_utility >= 0:
        ce = ((1 - gamma) * avg_utility) ** (1 / (1 - gamma)) if gamma != 1 else np.exp(avg_utility) - 1
    else:
        ce = -(((1 - gamma) * (-avg_utility)) / lambda_) ** (1 / (1 - gamma)) if gamma != 1 else -(np.exp(-avg_utility / lambda_) - 1)

    return ce


def compute_certainty_equivalents(results_dict: dict[str, pd.DataFrame], reference: float = 0.0) -> pd.DataFrame:
    """
    Applies the PT CE calculation to each strategy in results_dict.

    Returns a DataFrame with welfare rankings (CE) per strategy.
    """
    ce_list = []

    for key, df in results_dict.items():
        # Extract λ and γ from the key: "strategy_lambda_gamma"
        _, lam_str, gamma_str = key.split('_')
        lam = float(lam_str)
        gamma = float(gamma_str)
        returns = df['Portfolio Returns']

        ce = calculate_certainty_equivalent(returns, lambda_=lam, gamma=gamma, reference=reference)

        ce_list.append({
            'Strategy_Key': key,
            'Lambda': lam,
            'Gamma': gamma,
            'Certainty Equivalent': ce
        })

    ce_df = pd.DataFrame(ce_list).set_index("Strategy_Key").sort_values(by="Certainty Equivalent", ascending=False)
    return ce_df


def compare_certainty_equivalents(results_by_method: dict[str, dict[str, pd.DataFrame]], reference: float = 0.0) -> pd.DataFrame:
    """
    Compares Certainty Equivalents across multiple return estimation methods.

    Args:
        results_by_method: dictionary where
            key = method name (e.g., 'BMA', 'Historical Mean', 'MVP')
            value = results_dict from that method
        reference: reference return level for CE calculation (default = 0)

    Returns:
        pd.DataFrame with Strategy_Key, Lambda, Gamma, Method, Certainty Equivalent
    """
    ce_rows = []

    # Assume all methods have roughly the same strategy keys
    all_strategy_keys = set.intersection(*[set(r.keys()) for r in results_by_method.values()])

    for key in all_strategy_keys:
        lam = float(key.split("_")[1])
        gamma = float(key.split("_")[2])
        strategy = key.split("_")[0]

        for method_name, results_dict in results_by_method.items():
            df = results_dict[key]

            ce = calculate_certainty_equivalent(df['Portfolio Returns'], lambda_=lam, gamma=gamma, reference=reference)

            ce_rows.append({
                'Strategy_Key': key,
                'Strategy': strategy,
                'Lambda': lam,
                'Gamma': gamma,
                'Method': method_name,
                'Certainty Equivalent': ce
            })

    ce_df = pd.DataFrame(ce_rows).set_index('Strategy_Key')
    return ce_df.sort_values(by=["Lambda", "Gamma", "Strategy", "Method"])


def summarize_certainty_equivalents(ce_combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes total, mean, std, and median Certainty Equivalents per method.

    Args:
        ce_combined_df: DataFrame containing 'Method' and 'Certainty Equivalent' columns.

    Returns:
        pd.DataFrame: Summary table indexed by Method.
    """
    ce_sum_by_method = ce_combined_df.groupby('Method')['Certainty Equivalent'].sum()
    ce_mean_by_method = ce_combined_df.groupby('Method')['Certainty Equivalent'].mean()
    ce_std_by_method = ce_combined_df.groupby('Method')['Certainty Equivalent'].std()
    ce_median_by_method = ce_combined_df.groupby('Method')['Certainty Equivalent'].median()

    summary_df = pd.DataFrame({
        'CE Sum': ce_sum_by_method,
        'CE Mean': ce_mean_by_method,
        'CE Std Dev': ce_std_by_method,
        'CE Median': ce_median_by_method
    })

    return summary_df.sort_values(by='CE Mean', ascending=False)


def merge_ce_and_performance(ce_combined_df: pd.DataFrame, comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges CE results and performance metrics into one DataFrame for analysis.

    Args:
        ce_combined_df: DataFrame containing Certainty Equivalents (indexed by Strategy_Key)
        comparison_df: DataFrame containing Sharpe Ratio, Mean Return, etc. (indexed by Strategy_Key)

    Returns:
        pd.DataFrame: Combined summary table
    """
    # Ensure proper indexes
    ce_combined_df = ce_combined_df.reset_index()
    comparison_df = comparison_df.reset_index()

    # Merge on Strategy_Key and Method
    merged_df = pd.merge(
        comparison_df,
        ce_combined_df[['Strategy_Key', 'Method', 'Certainty Equivalent']],
        on=['Strategy_Key', 'Method'],
        how='inner'
    )

    return merged_df.set_index('Strategy_Key')



def build_performance_summary_by_method(
    ce_combined_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    results_by_method: dict[str, dict[str, pd.DataFrame]]
) -> pd.DataFrame:
    """
    Builds final summary table for each Method: CE, Sharpe, Max Drawdown, Mean Return, Std Dev, HHI.
    
    Args:
        ce_combined_df: DataFrame with Certainty Equivalent results.
        comparison_df: DataFrame with Sharpe, Mean Return, Max Drawdown, Std Dev.
        results_by_method: dict of method name -> results_dict (each strategy -> DataFrame).
    
    Returns:
        pd.DataFrame: Final summary per Method.
    """

    # --- Certainty Equivalent Aggregates ---
    ce_summary = ce_combined_df.groupby('Method')['Certainty Equivalent'].agg(
        CE_Sum='sum',
        CE_Mean='mean',
        CE_Std_Dev='std',
        CE_Median='median'
    )

    # --- Performance Aggregates ---
    perf_summary = comparison_df.groupby('Method').agg({
        'Mean Return': 'mean',
        'Std Dev': 'mean',              # Added standard deviation here
        'Sharpe Ratio': 'mean',
        'Max Drawdown': 'mean',
        'Final Wealth': 'mean'
    })

    # --- HHI Aggregates ---
    hhi_rows = []
    for method, results_dict in results_by_method.items():
        all_hhis = []
        for df in results_dict.values():
            weights_df = pd.DataFrame(df['Portfolio Weights'].tolist(), index=df.index)
            hhi = (weights_df ** 2).sum(axis=1)
            all_hhis.append(hhi)
        combined_hhi = pd.concat(all_hhis)
        avg_hhi = combined_hhi.mean()
        hhi_rows.append({'Method': method, 'Avg HHI': avg_hhi})

    hhi_summary = pd.DataFrame(hhi_rows).set_index('Method')

    # --- Merge all summaries ---
    final_summary = pd.concat([ce_summary, perf_summary, hhi_summary], axis=1)

    return final_summary.sort_values(by='CE_Mean', ascending=False)

def evaluate_forecast_accuracy(results_by_method: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Evaluates forecast accuracy (MSE, MAE, Bias) for expected vs realized portfolio returns
    across multiple return estimation methods.

    Args:
        results_by_method: dict of method name -> results_dict (each strategy -> DataFrame).

    Returns:
        pd.DataFrame: Summary table per Method.
    """

    rows = []

    for method, results_dict in results_by_method.items():
        method_errors = []

        for key, df in results_dict.items():
            if 'Expected Portfolio Returns' not in df.columns:
                print(f"⚠️ Warning: Method {method} Strategy {key} has no Expected Portfolio Return. Skipping...")
                continue

            expected_returns = df['Expected Portfolio Returns'].values
            realized_returns = df['Portfolio Returns'].values

            errors = expected_returns - realized_returns
            method_errors.extend(errors)

        method_errors = np.array(method_errors)

        mse = np.mean(method_errors**2)
        mae = np.mean(np.abs(method_errors))
        bias = np.mean(method_errors)

        rows.append({
            'Method': method,
            'MSE': mse,
            'MAE': mae,
            'Bias': bias
        })

    return pd.DataFrame(rows).set_index('Method')
