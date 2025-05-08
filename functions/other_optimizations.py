from scipy.optimize import minimize
import numpy as np


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


def backtest_portfolio_bma_mvo(bma_returns: dict, risk_aversion: float = 3.0) -> dict:
    """
    Backtest using Mean-Variance optimization on BMA simulated returns.

    Args:
        bma_returns: dict[date] -> (simulated_returns, expected_returns)
        risk_aversion: Risk aversion coefficient (default = 3.0) - 	DeMiguel, Garlappi, and Uppal (2009) often use γ = 3.

    Returns:
        results_dict: dict[date] -> dict with realized return, expected return, weights
    """
    results = {}

    dates = list(bma_returns.keys())
    cumulative_return = 1.0

    for t, current_date in enumerate(dates[:-1]):
        sim_draws, expected_returns = bma_returns[current_date]

        mean_return = expected_returns.values
        cov_matrix = np.cov(sim_draws.T)

        # Optimize
        weights = optimize_mean_variance(mean_return, cov_matrix, risk_aversion)

        # Next period returns
        next_draws, _ = bma_returns[dates[t + 1]]
        realized_returns = next_draws.mean(axis=0)  # average of simulated returns next period

        portfolio_return = np.dot(realized_returns, weights)

        cumulative_return *= (1 + portfolio_return)

        # Expected portfolio return
        expected_portfolio_return = np.dot(mean_return, weights)

        results[current_date] = {
            'Portfolio Return': portfolio_return,
            'Compounded Return': cumulative_return,
            'Portfolio Weights': weights,
            'Expected Portfolio Return': expected_portfolio_return
        }

    return results

import pandas as pd

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

    certainty_equivalent = (ce_sum / S) ** (1 / (1 - gamma))

    return certainty_equivalent - 1  # Normalize so CE is excess over 0

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


def backtest_bma_naive_df(bma_returns: dict) -> pd.DataFrame:
    """
    Backtests the 1/N naive equal-weighted portfolio directly on BMA-simulated returns.

    Args:
        bma_returns: OrderedDict[date] -> (simulated_draws, expected_returns)

    Returns:
        DataFrame with columns:
        ['Portfolio Returns', 'Compounded Returns', 'Portfolio Weights', 'Expected Portfolio Returns']
    """

    rows = []
    dates = list(bma_returns.keys())
    num_assets = bma_returns[dates[0]][0].shape[1]

    equal_weights = np.ones(num_assets) / num_assets
    cumulative_return = 1.0

    for t in range(len(dates) - 1):
        current_date = dates[t]
        next_date = dates[t + 1]

        # Get current expected returns and next realized returns
        _, expected_returns = bma_returns[current_date]
        next_draws, _ = bma_returns[next_date]

        realized_returns = next_draws.mean(axis=0)
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


