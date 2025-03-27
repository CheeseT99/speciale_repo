import os
import itertools
import numpy as np
import pandas as pd
from numpy.linalg import inv, slogdet
from scipy.special import gammaln
import math
import time

toc = time.time()

##############################################################################
# 1) HELPER FUNCTIONS FOR BMA
##############################################################################

def log_marginal_likelihood(X, y, alpha):
    """
    Computes a simplified log of marginal likelihood under a Zellner's g-prior approach.

    Parameters
    ----------
    X : np.ndarray, shape (n_obs, k_vars)
        Design matrix (including intercept if desired).
    y : np.ndarray, shape (n_obs,)
        Dependent variable (returns).
    alpha : float
        'g' in the g-prior, controlling prior shrinkage.

    Returns
    -------
    float
        Log of marginal likelihood (model evidence).
    """
    n, k = X.shape
    # Add small ridge for numerical stability if needed:
    V = X.T @ X  # + 1e-9 * np.eye(k)

    # OLS estimates
    try:
        beta_ols = inv(V) @ (X.T @ y)
    except np.linalg.LinAlgError:
        # Singularity or ill-conditioned matrix; return a very low likelihood
        return -1e10

    RSS_ols = np.sum((y - X @ beta_ols)**2)

    # Basic approximate log-likelihood ignoring constants:
    #   ll_ols ~ -(n/2) * log(RSS_ols)
    ll_ols = -0.5 * n * np.log(RSS_ols + 1e-12)  # add small constant

    # Penalty for model size (Zellner's g-prior)
    penalty = -0.5 * k * np.log(1 + alpha)

    # Combine
    log_ev = ll_ols + penalty
    return log_ev

def posterior_predictive_mean(X, y, Xnew, alpha):
    """
    A simple posterior predictive mean under the same g-prior.
    Here we do OLS-based estimate for demonstration.
    """
    V = X.T @ X
    try:
        beta_ols = inv(V) @ (X.T @ y)
    except np.linalg.LinAlgError:
        # If inversion fails, fallback
        return 0.0
    # Posterior mean of beta can be slightly shrunk, but we’ll just use OLS
    beta_post = beta_ols
    ypred = Xnew @ beta_post
    return ypred

##############################################################################
# 2) MAIN BMA FUNCTION
##############################################################################

def run_bma_on_csv(csv_path, ret_col='ret_excess', macro_start='dp', ff_factors=None,
                   max_factors=3, alpha_prior=10.0, print_top=10):
    """
    Reads a CSV, enumerates subsets of factors (columns) up to a given size,
    computes marginal likelihood for each subset, and performs BMA.

    Parameters
    ----------
    csv_path : str
        Full path to your data file (CSV).
    ret_col : str
        Column name for the dependent variable (e.g., 'ret_excess').
    macro_start : str
        Column from which all subsequent columns are macro predictors.
    ff_factors : list or None
        List of Fama-French factor columns. If None, uses empty list.
    max_factors : int
        Maximum subset size for demonstration (can be 1..N).
    alpha_prior : float
        Hyperparameter (g) for the Zellner's g-prior approach.
    print_top : int
        How many top models to print.

    Returns
    -------
    df_results : pd.DataFrame
        DataFrame of models, their log marginal likelihood, and posterior probability.
    """
    # Load data
    df = pd.read_csv(csv_path, index_col='date', parse_dates=True)
    df = df.drop(columns=['month','month:1','month:2'])
    print("Data has been loaded")
    # Make sure the ret_col is in the DataFrame
    if ret_col not in df.columns:
        raise ValueError(f"Dependent variable '{ret_col}' not found in DataFrame columns.")

    # Dependent variable
    y = df[ret_col].values
    nobs = len(y)

    # Identify macro columns (from macro_start onward)
    all_cols = df.columns.tolist()

    if macro_start in all_cols:
        start_idx = all_cols.index(macro_start)
        macro_cols = all_cols[start_idx:]
    else:
        macro_cols = []

    # Build a list of candidate columns = macro + optional Fama-French
    ff_factors = ff_factors if ff_factors is not None else []
    candidate_cols = macro_cols + ff_factors

    # But remove the dependent variable if it accidentally appears
    candidate_cols = [c for c in candidate_cols if c != ret_col]

    # Prepare to store results
    results = []
    print("Running log-marginal-likelihoods")
    # We always add an intercept, so we do not treat it as a "factor"
    # Enumerate subsets up to max_factors in size
    for k in range(1, max_factors + 1):
        for subset in itertools.combinations(candidate_cols, k):
            Xsubset = df[list(subset)].values  # shape (nobs, k)

            # Add intercept
            intercept = np.ones((nobs, 1))
            X = np.hstack([intercept, Xsubset])  # shape (nobs, k+1)

            # Compute log marginal likelihood
            ll = log_marginal_likelihood(X, y, alpha_prior)

            # Save
            model_desc = f"Intercept + {subset}"
            results.append((model_desc, ll, subset))

    df_results = pd.DataFrame(results, columns=['Model', 'LogMargLik', 'Predictors'])

    # Convert logMargLik -> posterior via exp, assume uniform prior over models
    max_ll = df_results['LogMargLik'].max()
    # for numerical stability
    weights_unnorm = np.exp(df_results['LogMargLik'] - max_ll)
    posterior_probs = weights_unnorm / np.sum(weights_unnorm)

    df_results['PostProb'] = posterior_probs
    df_results.sort_values('PostProb', ascending=False, inplace=True, ignore_index=True)

    # Print top models
    print(f"\n=== Top {print_top} Models by Posterior Probability ===")
    print(df_results.head(print_top))

    # Simple BMA forecast for the LAST observation (in-sample demonstration)
    # We'll do a 1 x (k+1) design row
    # Weighted combination of each model's posterior predictive
    pred_bma = 0.0
    for i in range(len(df_results)):
        subset = df_results.loc[i, 'Predictors']
        w_i = df_results.loc[i, 'PostProb']

        # Build training X
        Xsubset = df[list(subset)].values
        intercept = np.ones((nobs, 1))
        Xtrain = np.hstack([intercept, Xsubset])

        # Build Xnew for the last row
        # shape => (1, k+1)
        last_row = df.iloc[-1][list(subset)]  # shape => (k,)
        Xnew = np.hstack([[1.0], last_row.values])

        y_pred_i = posterior_predictive_mean(Xtrain, y, Xnew.reshape(1, -1), alpha_prior)
        pred_bma += w_i * y_pred_i

    print(f"\nBMA (in-sample) predictive mean for the final row of dependent variable '{ret_col}': {pred_bma[0]:.4f}")

    return df_results

##############################################################################
# 3) EXAMPLE USAGE
##############################################################################

if __name__ == "__main__":
    # Construct path
    parent_dir = os.path.dirname(os.getcwd())
    csv_path = os.path.join(parent_dir, 'speciale_repo', 'data', 'data_1st_BMA.csv')
    # We know your DataFrame has columns like 'ret', 'ret_excess', 'dp', ... 'rf'
    # We'll pick 'ret_excess' as the dependent variable, 'dp' as start of macro
    # and some Fama-French factors
    ff_factors_example = ['mkt_excess', 'smb', 'hml', 'rf']

    # Run BMA on up to 3-factor subsets
    df_bma_results = run_bma_on_csv(
        csv_path=csv_path,
        ret_col='ret_excess',
        macro_start='dp',
        ff_factors=ff_factors_example,
        max_factors=3,
        alpha_prior=10.0,
        print_top=10
    )
    tic = time.time()
    print(f"The program was run in {(tic-toc):.4f} seconds.")
