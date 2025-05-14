from scipy.optimize import minimize
import numpy as np
import pandas as pd


def optimize_mean_variance(expected_returns, cov_matrix, risk_aversion=3.0):
    """
    Solves the classical mean-variance optimization problem:
        max_w E[r]^T w - (risk_aversion / 2) * w^T Σ w
        subject to: sum(w) = 1, w_i >= 0

    Args:
        expected_returns: 1D np.array of expected returns (K,)
        cov_matrix: 2D np.array of asset covariances (K, K)
        risk_aversion: gamma coefficient (default = 3.0)

    Returns:
        Optimal weight vector (np.array of shape (K,))
    """

    K = len(expected_returns)
    init_weights = np.ones(K) / K

    def objective(w):
        mean_term = np.dot(w, expected_returns)
        risk_term = 0.5 * risk_aversion * np.dot(w, cov_matrix @ w)
        return - (mean_term - risk_term)

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # sum to 1
    ]

    bounds = [(0, 1) for _ in range(K)]  # no short-selling

    result = minimize(objective, init_weights, method='SLSQP', constraints=constraints, bounds=bounds)
    return result.x


def backtest_portfolio_bma_mvo(bma_returns: dict, risk_aversion: float = 3.0, seed: int = 42) -> dict:
    """
    Backtest using Mean-Variance optimization on BMA simulated returns,
    drawing a single realistic outcome from the posterior predictive distribution.

    Args:
        bma_returns: dict[date] -> (simulated_returns, expected_returns, covariance)
        risk_aversion: Risk aversion coefficient (γ)
        seed: Random seed for reproducibility

    Returns:
        results_dict: dict[date] -> dict with realized return, expected return, weights
    """
    np.random.seed(seed)
    results = {}

    dates = list(bma_returns.keys())
    cumulative_return = 1.0

    for t, current_date in enumerate(dates[:-1]):
        sim_draws, expected_returns, covariance = bma_returns[current_date]

        mu = expected_returns.values
        cov = covariance

        # Optional: regularize covariance matrix to avoid instability
        cov += 1e-4 * np.eye(len(cov))

        # Step 1: Optimize portfolio weights
        weights = optimize_mean_variance(mu, cov, risk_aversion)

        # Step 2: Sample ONE realistic outcome from predictive distribution
        next_draws, _, _ = bma_returns[dates[t + 1]]
        random_index = np.random.randint(next_draws.shape[0])
        realized_returns = next_draws[random_index]  # shape: (K,)

        # Step 3: Compute return and track results
        portfolio_return = np.dot(realized_returns, weights)
        cumulative_return *= (1 + portfolio_return)
        expected_portfolio_return = np.dot(mu, weights)

        results[current_date] = {
            'Portfolio Return': portfolio_return,
            'Compounded Return': cumulative_return,
            'Portfolio Weights': weights,
            'Expected Portfolio Return': expected_portfolio_return
        }

    return results



def mvo_results_to_dataframe(results_dict):
    """
    Converts MVO results into a clean DataFrame.
    """
    rows = []

    for date, values in results_dict.items():
        rows.append({
            'Portfolio Returns': values['Portfolio Return'],
            'Compounded Returns': values['Compounded Return'],
            'Portfolio Weights': values['Portfolio Weights'],
            'Expected Portfolio Returns': values['Expected Portfolio Return']
        })

    df = pd.DataFrame(rows, index=list(results_dict.keys()))
    return df


# The next 3 functions are for calculating important metrics for comparison of methods
def compute_certainty_equivalent(df, lambda_: float, gamma: float, reference: float = 0.0) -> float:
    """
    Computes the Certainty Equivalent for a portfolio based on Prospect Theory.

    Args:
        df: DataFrame with 'Portfolio Returns'
        lambda_: Loss aversion parameter (λ)
        gamma: Risk aversion curvature (γ)
        reference: Reference return (default 0)

    Returns:
        Certainty Equivalent (float)
    """
    returns = df['Portfolio Returns']
    ce_sum = 0
    S = len(returns)

    for r in returns:
        gain_term = max(0, r - reference)
        loss_term = max(0, reference - r)

        ce_sum += (gain_term ** (1 - gamma)) / (1 - gamma) - lambda_ * (loss_term ** (1 - gamma)) / (1 - gamma)

    avg_utility = ce_sum / S

    if avg_utility >= 0:
        ce = ((1 - gamma) * avg_utility) ** (1 / (1 - gamma))
    else:
        ce = -(((1 - gamma) * (-avg_utility)) / lambda_) ** (1 / (1 - gamma))

    return ce


def summarize_backtest(df) -> dict:
    """
    Summarizes key risk and return metrics from a backtest.

    Args:
        df: DataFrame with 'Portfolio Returns' and 'Compounded Returns'

    Returns:
        Dictionary with Mean Return, Std Dev, Sharpe Ratio, Max Drawdown
    """

    returns = df['Portfolio Returns']
    compounded = df['Compounded Returns']

    mean_return = returns.mean()
    std_return = returns.std()
    sharpe_ratio = mean_return / std_return if std_return != 0 else np.nan

    # Max drawdown based on Compounded Returns
    running_max = compounded.cummax()
    drawdowns = 1 - compounded / running_max
    max_drawdown = drawdowns.max()

    return {
        'Mean Return': mean_return,
        'Std Dev': std_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }

def calculate_average_hhi(df) -> float:
    """
    Calculates the average Herfindahl-Hirschman Index (HHI) over the backtest period.

    Args:
        df: DataFrame with 'Portfolio Weights' column

    Returns:
        Average HHI (float)
    """
    import pandas as pd

    weights_df = pd.DataFrame(df['Portfolio Weights'].tolist(), index=df.index)
    hhi_series = (weights_df ** 2).sum(axis=1)

    return hhi_series.mean()


def summarize_methods_comparison(methods: dict, lambda_: float = 1.0, gamma: float = 0.5) -> pd.DataFrame:
    """
    Summarizes Certainty Equivalent, Sharpe, Max Drawdown, Mean Return, Std Dev, and Avg HHI
    across different optimization methods.

    Args:
        methods: dict {Method Name: DataFrame}
        lambda_: PT loss aversion parameter (for CE calculation)
        gamma: PT risk aversion parameter (for CE calculation)

    Returns:
        pd.DataFrame with all metrics, indexed by Method.
    """
    rows = []

    for method_name, df in methods.items():
        # Certainty Equivalent
        ce = compute_certainty_equivalent(df, lambda_=lambda_, gamma=gamma)

        # Sharpe, Max Drawdown, Mean, Std
        summary = summarize_backtest(df)

        # HHI
        avg_hhi = calculate_average_hhi(df)

        row = {
            'Certainty Equivalent': ce,
            'Mean Return': summary['Mean Return'],
            'Std Dev': summary['Std Dev'],
            'Sharpe Ratio': summary['Sharpe Ratio'],
            'Max Drawdown': summary['Max Drawdown'],
            'Avg HHI': avg_hhi
        }

        rows.append((method_name, row))

    final_df = pd.DataFrame(dict(rows)).T
    return final_df

def sensitivity_analysis_ce(methods: dict, 
                             lambda_values: list = [1.5, 2.0, 2.5, 3.0], 
                             gamma: float = 0.5) -> pd.DataFrame:
    """
    Runs sensitivity analysis by varying λ (loss aversion) and holding γ fixed.

    Args:
        methods: dict {Method Name: DataFrame}
        lambda_values: List of λ values to test
        gamma: Fixed γ risk curvature

    Returns:
        pd.DataFrame with CE_PT, CE_MVO, Difference, indexed by λ
    """
    rows = []

    df_pt = methods['Prospect Theory']
    df_mvo = methods['Mean-Variance']

    for lam in lambda_values:
        ce_pt = compute_certainty_equivalent(df_pt, lambda_=lam, gamma=gamma)
        ce_mvo = compute_certainty_equivalent(df_mvo, lambda_=lam, gamma=gamma)

        rows.append({
            'λ': lam,
            'CE_PT': ce_pt,
            'CE_MVO': ce_mvo,
            'Difference (PT - MVO)': ce_pt - ce_mvo
        })

    return pd.DataFrame(rows).set_index('λ')


def backtest_bma_naive_df(bma_returns: dict, seed: int = 42) -> pd.DataFrame:
    """
    Backtests the 1/N naive equal-weighted portfolio using one realistic return draw per period.

    Args:
        bma_returns: OrderedDict[date] -> (sim_draws, expected_returns, covariance)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with:
        ['Portfolio Returns', 'Compounded Returns', 'Portfolio Weights', 'Expected Portfolio Returns']
    """
    np.random.seed(seed)
    rows = []
    dates = list(bma_returns.keys())
    num_assets = bma_returns[dates[0]][0].shape[1]

    equal_weights = np.ones(num_assets) / num_assets
    cumulative_return = 1.0

    for t in range(len(dates) - 1):
        current_date = dates[t]
        next_date = dates[t + 1]

        # Expected returns at current date
        _, expected_returns, _ = bma_returns[current_date]

        # Simulate one realized path
        next_draws, _, _ = bma_returns[next_date]
        random_index = np.random.randint(next_draws.shape[0])
        realized_returns = next_draws[random_index]  # shape (num_assets,)

        portfolio_return = np.dot(realized_returns, equal_weights)
        expected_return = np.dot(expected_returns.values, equal_weights)

        cumulative_return *= (1 + portfolio_return)

        rows.append({
            'Date': current_date,
            'Portfolio Returns': portfolio_return,
            'Compounded Returns': cumulative_return,
            'Portfolio Weights': equal_weights.copy(),
            'Expected Portfolio Returns': expected_return
        })

    return pd.DataFrame(rows).set_index('Date')


import numpy as np
import pandas as pd
from typing import Callable
from scipy.stats import t
import matplotlib.pyplot as plt

def compute_metrics(returns: np.ndarray, lambda_=2.25, gamma=0.5, r_hat=0.0):
    """
    Compute mean, std, sharpe, certainty equivalent for a return series.
    """
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    sharpe = mean / std if std > 0 else np.nan

    # Certainty Equivalent (Prospect Theory-style)
    ce_sum = 0
    S = len(returns)
    for r in returns:
        gain = max(0, r - r_hat)
        loss = max(0, r_hat - r)
        ce_sum += (gain ** (1 - gamma)) / (1 - gamma) - lambda_ * (loss ** (1 - gamma)) / (1 - gamma)
    avg_util = ce_sum / S
    if avg_util >= 0:
        ce = ((1 - gamma) * avg_util) ** (1 / (1 - gamma))
    else:
        ce = -(((1 - gamma) * (-avg_util)) / lambda_) ** (1 / (1 - gamma))

    return {
        'Mean Return': mean,
        'Std Dev': std,
        'Sharpe Ratio': sharpe,
        'Certainty Equivalent': ce
    }

def bootstrap_backtest(strategy_func: Callable, bma_returns: dict, n_runs: int = 1000) -> pd.DataFrame:
    """
    Bootstraps a backtest by running the strategy n_runs times with different seeds.
    Returns a DataFrame with metrics for each run.
    """
    records = []
    for i in range(n_runs):
        df = strategy_func(bma_returns, seed=i)
        returns = df['Portfolio Returns'].dropna().values
        metrics = compute_metrics(returns)
        records.append(metrics)

    return pd.DataFrame(records)

def backtest_portfolio_bma_mvo_df(bma_returns, risk_aversion=3.0, seed=42):
    """
    Wraps the MVO dictionary output as a clean DataFrame with expected columns.
    Ensures compatibility with bootstrap_backtest.
    """
    result_dict = backtest_portfolio_bma_mvo(bma_returns, risk_aversion=risk_aversion, seed=seed)

    # Convert to DataFrame
    df = pd.DataFrame(result_dict).T  # Transpose so timestamps are index

    # Standardize column naming
    df = df.rename(columns={
        'Portfolio Return': 'Portfolio Returns'
    })

    df.index.name = 'Date'
    return df

def backtest_bma_naive_df(bma_returns, seed=42):
    """
    Updated naive backtest to return expected DataFrame structure with realistic simulation.
    """
    np.random.seed(seed)
    rows = []
    dates = list(bma_returns.keys())
    num_assets = bma_returns[dates[0]][0].shape[1]

    equal_weights = np.ones(num_assets) / num_assets
    cumulative_return = 1.0

    for t in range(len(dates) - 1):
        current_date = dates[t]
        next_date = dates[t + 1]

        _, expected_returns, _ = bma_returns[current_date]
        next_draws, _, _ = bma_returns[next_date]

        random_index = np.random.randint(next_draws.shape[0])
        realized_returns = next_draws[random_index]

        portfolio_return = np.dot(realized_returns, equal_weights)
        expected_return = np.dot(expected_returns.values, equal_weights)

        cumulative_return *= (1 + portfolio_return)

        rows.append({
            'Date': current_date,
            'Portfolio Returns': portfolio_return,
            'Compounded Returns': cumulative_return,
            'Portfolio Weights': equal_weights.copy(),
            'Expected Portfolio Returns': expected_return
        })

    return pd.DataFrame(rows).set_index('Date')



def summarize_metrics(df: pd.DataFrame):
    summary = df.describe(percentiles=[0.025, 0.5, 0.975]).T[['mean', 'std', '2.5%', '50%', '97.5%']]
    return summary.rename(columns={'mean': 'Avg', 'std': 'SD'})

def plot_metric_distributions(df: pd.DataFrame, strategy_name: str):
    plt.figure(figsize=(14, 6))
    for i, col in enumerate(df.columns):
        plt.subplot(1, len(df.columns), i + 1)
        plt.hist(df[col], bins=30, alpha=0.7)
        plt.title(f"{strategy_name} - {col}")
        plt.axvline(df[col].mean(), color='r', linestyle='--', label='Mean')
        plt.legend()
    plt.tight_layout()
    plt.show()


