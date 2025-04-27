import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, levene, f_oneway
from conditionalAssetPricingLogMarginalLikelihoodTauClass import Model, printFactorsAndPredictorsProbabilities
import pickle as pkl
import os
from collections import OrderedDict
from sklearn.linear_model import LinearRegression

import evaluation as ev

import time

def load_historical_returns(parent_dir: str, start_date=None, end_date=None) -> pd.DataFrame:
    """
    Loads historical factor returns to be used as investable returns
    for historical mean backtesting.

    Args:
        parent_dir: Path to folder containing 'factors-20.csv'
        start_date: Optional (YYYY-MM-DD) string
        end_date: Optional (YYYY-MM-DD) string

    Returns:
        pd.DataFrame: [Date x Factors] monthly returns (as decimals)
    """

    factors_path = os.path.join(parent_dir, "Complemetary Code Files for Submission", "Data", "factors-20.csv")
    factors = pd.read_csv(factors_path)

    # Parse date
    factors['Date'] = pd.to_datetime(factors['Date'].astype(str), format='%Y%m')
    factors.set_index('Date', inplace=True)
    factors.sort_index(inplace=True)

    # Drop non-investable columns
    drop_cols = ['MKTRF', 'SMB*', 'MKT', 'CON', 'IA', 'ROE', 'ME']
    factors = factors.drop(columns=[col for col in drop_cols if col in factors.columns], errors='ignore')

    # Optional slicing
    if start_date:
        factors = factors.loc[start_date:]
    if end_date:
        factors = factors.loc[:end_date]

    return factors


def load_reference_returns(parent_dir: str, ref_type: str = "market", start_date=None, end_date=None) -> dict:
    """
    Loads reference returns for use in PT evaluations.

    Args:
        parent_dir (str): Path to root data folder
        ref_type (str): 'market' (Mkt-RF) or 'spy' (MKT)
        start_date (str): Optional start filter (YYYY-MM-DD)
        end_date (str): Optional end filter (YYYY-MM-DD)

    Returns:
        dict[pd.Timestamp, float]: reference returns (monthly)
    """

    factors_path = os.path.join(parent_dir, "Complemetary Code Files for Submission", "Data", "factors-20.csv")
    factors = pd.read_csv(factors_path)

    factors['Date'] = pd.to_datetime(factors['Date'].astype(str), format='%Y%m')
    factors.set_index('Date', inplace=True)
    factors.sort_index(inplace=True)

    # Map available references
    col_map = {
        "market": "Mkt-RF",
        "spy": "MKT"
    }

    if ref_type not in col_map:
        raise ValueError(f"ref_type must be one of {list(col_map.keys())}")

    col = col_map[ref_type]

    if col not in factors.columns:
        raise KeyError(f"Column '{col}' not found in factors-20.csv")

    returns = factors[col] / 100  # convert from percent to decimal

    if start_date:
        returns = returns.loc[start_date:]
    if end_date:
        returns = returns.loc[:end_date]

    return returns.to_dict()


def load_test_assets_and_factors(parent_dir: str, start_date=None, end_date=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads both test asset returns and regression factor returns
    from 'factors-20.csv'.

    Args:
        parent_dir: path to folder containing factors-20.csv
        start_date: optional start filter (YYYY-MM-DD)
        end_date: optional end filter (YYYY-MM-DD)

    Returns:
        (test_assets_df, factor_returns_df): tuple of two DataFrames
            - test_assets_df: returns of SMB, HML, RMW, CMA, MMOM
            - factor_returns_df: returns of Mkt-RF, IA, ROE
    """

    factors_path = os.path.join(parent_dir, "Complemetary Code Files for Submission", "Data", "factors-20.csv")

    factors = pd.read_csv(factors_path)

    # Parse date
    factors['Date'] = pd.to_datetime(factors['Date'].astype(str), format='%Y%m')
    factors.set_index('Date', inplace=True)
    factors.sort_index(inplace=True)

    # Select test asset columns
    test_asset_columns = ['SMB', 'HML', 'RMW', 'CMA', 'MMOM']
    test_assets = factors[test_asset_columns]

    # Select regression factor columns
    factor_columns = ['Mkt-RF', 'IA', 'ROE']
    factor_returns = factors[factor_columns]

    # Optional slicing
    if start_date:
        test_assets = test_assets.loc[start_date:]
        factor_returns = factor_returns.loc[start_date:]
    if end_date:
        test_assets = test_assets.loc[:end_date]
        factor_returns = factor_returns.loc[:end_date]

    return test_assets, factor_returns



def load_market_benchmark(parent_dir: str, start_date=None, end_date=None) -> pd.Series:
    factors_path = os.path.join(parent_dir, "Complemetary Code Files for Submission", "Data", "factors-20.csv")
    
    factors = pd.read_csv(factors_path)
    factors['Date'] = pd.to_datetime(factors['Date'].astype(str), format='%Y%m')
    factors.set_index('Date', inplace=True)
    factors.sort_index(inplace=True)

    # Extract Mkt-RF or other benchmark (assumed in percentage points, so divide by 100 if needed)
    market_returns = factors['Mkt-RF'] 

    # Optional slicing
    if start_date:
        market_returns = market_returns.loc[start_date:]
    if end_date:
        market_returns = market_returns.loc[:end_date]

    # Compute cumulative returns
    cumulative_market = (1 + market_returns).cumprod()
    return cumulative_market

# Bemærk: bruges ikke ved BMA
def prospect_value(weights, r_s, r_hat, lambda_, gamma=0.1, strategy="conservative", r_tminus1=0.0, r_tminus2=0.0):
    """
    Calculate the prospect value function with dynamic lambda for conservative or aggressive investors.

    Parameters:
    weights (array): The weights for each asset in the portfolio.
    r_s (array): The returns of each asset.
    r_hat (float): The reference return.
    lambda_ (float): The base loss aversion sensitivity coefficient.
    gamma (float): The risk aversion coefficient (default is 0.1).
    strategy (str): The risk strategy of the investor ('conservative' or 'aggressive').

    Returns:
    float: The prospect value.
    """
    # Calculate portfolio returns based on weights
    portfolio_returns = np.dot(r_s, weights) #Used to calculate expected returns
    # er mean return for hver måned ved første kørsel af minimizer (lige vægte)

    # Calculate zt based on these returns
    zt = calculate_zt(np.mean(portfolio_returns), r_tminus1)

    # Dynamically adjust lambda based on strategy and portfolio performance
    if strategy == "conservative":
        lambda_dynamic = calculate_conservative_lambda(r_tminus1, r_tminus2, zt, lambda_)
    elif strategy == "aggressive":
        lambda_dynamic = calculate_aggressive_lambda(r_tminus1, r_tminus2, zt, lambda_)
    else:
        lambda_dynamic = lambda_

    # Calculate prospect value
    S = len(r_s)  # Number of periods
    prospect_value_sum = 0

    for s in range(S):
        r_sx = np.dot(r_s[s], weights)  # Portfolio return for time period s
        
        gain_term = max(0, r_sx - r_hat)
        loss_term = max(0, r_hat - r_sx)

        prospect_value_sum += (gain_term ** (1 - gamma)) / (1 - gamma) - lambda_dynamic * (loss_term ** (1 - gamma)) / (1 - gamma)
        

    return -prospect_value_sum / S  # Negative sign for maximization


def calculate_conservative_lambda(last_return, second_last_return, zt, lambda_=0.1):
    """
    Calculate conservative lambda based on the returns.
    """
    if last_return >= second_last_return:
        return lambda_
    else:
        return lambda_ + (zt - 1)

def calculate_aggressive_lambda(last_return, second_last_return, zt, lambda_=0.1):
    """
    Calculate aggressive lambda based on the returns.
    """
    if last_return >= second_last_return:
        return lambda_
    else:
        return lambda_ + ((1 / zt) - 1)

def calculate_zt(expected_return, last_return):
    """
    Calculate zt based on expected and last return.
    """
    return (1 + last_return) / (1 + expected_return)


def bma_initialization(ff, zz, tau, index_of_estimation, n_predictors_to_use=2):
    """
    Step 1: Initialize the conditional BMA model and compute marginal likelihoods.

    Returns:
        A dictionary containing model object, posterior probabilities, and diagnostics.
    """


    significantPredictors = np.arange(n_predictors_to_use)

    model = Model(
        rr=pd.DataFrame({'': []}),
        ff=ff,
        zz=zz,
        significantPredictors=significantPredictors,
        Tau=tau,
        indexEndOfEstimation=index_of_estimation,
        key_demean_predictors=True
    )

    (CLMLU, factorsNames, factorsProbabilityU, predictorsNames, predictorsProbabilityU,
     T0IncreasedFraction, T0Max, T0Min, T0Avg, T0_div_T0_plus_TAvg, T_div_T0_plus_TAvg,
     CLMLR, factorsProbabilityR, predictorsProbabilityR) = model.conditionalAssetPricingLogMarginalLikelihoodTauNumba()

    CLMLCombined = np.concatenate((CLMLU, CLMLR))
    CMLCombined = np.exp(CLMLCombined - np.max(CLMLCombined))
    CMLCombined /= np.sum(CMLCombined)

    factorsProbability = (
        np.sum(CMLCombined[:len(CLMLU)]) * factorsProbabilityU +
        np.sum(CMLCombined[len(CLMLU):]) * factorsProbabilityR
    )
    predictorsProbability = (
        np.sum(CMLCombined[:len(CLMLU)]) * predictorsProbabilityU +
        np.sum(CMLCombined[len(CLMLU):]) * predictorsProbabilityR
    )

    print(f"Probability of mispricing    = {np.sum(CMLCombined[:len(CLMLU)]):.4f}")
    print(f"Probability of no mispricing = {np.sum(CMLCombined[len(CLMLU):]):.4f}")

    return {
        'model': model,
        'CMLCombined': CMLCombined,
        'CLMLU': CLMLU,
        'CLMLR': CLMLR,
        'factorsProbability': factorsProbability,
        'predictorsProbability': predictorsProbability,
        'T0IncreasedFraction': T0IncreasedFraction,
        'T0Max': T0Max,
        'T0Min': T0Min,
        'T0Avg': T0Avg,
        'T0divT0plusTAvg': T0_div_T0_plus_TAvg,
        'TdivT0plusTAvg': T_div_T0_plus_TAvg
    }

def bma_predictions(bma_dict, number_of_sim=1000):
    """
    Step 2: From the initialized model and posterior, compute predictive means and covariances.

    Returns:
        Dictionary with 'returns_OOS' and 'covariance_matrix_OOS'.
    """
    model = bma_dict['model']
    CMLCombined = bma_dict['CMLCombined']

    nModelsInPrediction = min(number_of_sim, 1000)

    (returns_OOS, _, _, 
     covariance_matrix_OOS, _, 
     _, _, _, _, _, _) = model.conditionalAssetPricingOOSTauNumba(CMLCombined, nModelsInPrediction)

    return {
        'returns_OOS': returns_OOS,
        'covariance_matrix_OOS': covariance_matrix_OOS
    }

def calculate_bma_returns(prediction_dict, number_of_sim=10000):
    """
    Step 3: Simulate BMA returns from predicted means and covariances.

    Returns:
        Simulated return draws (np.ndarray of shape (number_of_sim, K))
    """

    means = prediction_dict['returns_OOS'][0]
    cov = prediction_dict['covariance_matrix_OOS'][0]

    simulated_returns = np.random.multivariate_normal(
        means,
        cov,
        number_of_sim
    )
    # # Add noise to the simulated returns
    # noise_std = 0.01  # Example: add 1% standard deviation noise
    # noise = np.random.normal(0, noise_std, size=simulated_returns.shape)
    # simulated_returns_noisy = simulated_returns + noise


    # return simulated_returns_noisy
    return simulated_returns

def run_bma_pipeline(ff_slice, zz_slice, tau, pickle_path_init, pickle_path_pred, number_of_sim=10000):
    """
    Runs the full BMA pipeline with caching for both:
    - Step 1: BMA initialization (heavy)
    - Step 2: Predictive mean/covariance (moderate)
    - Step 3: Return simulation (lightweight)

    Returns:
        np.ndarray of simulated returns.
    """

    # === Step 1: Load or compute and save model initialization ===
    if os.path.exists(pickle_path_init):
        print(f"Loading cached BMA initialization from: {pickle_path_init}")
        with open(pickle_path_init, "rb") as f:
            bma_dict = pkl.load(f)
    else:
        print(f"Pickle not found. Initializing and saving to: {pickle_path_init}")
        index_of_estimation = len(ff_slice) - 10
        bma_dict = bma_initialization(ff_slice, zz_slice, tau, index_of_estimation)

        with open(pickle_path_init, "wb") as f:
            pkl.dump(bma_dict, f)

    # === Step 2: Load or compute and save predictive mean/covariance ===
    if os.path.exists(pickle_path_pred):
        print(f"Loading cached BMA predictions from: {pickle_path_pred}")
        with open(pickle_path_pred, "rb") as f:
            pred_dict = pkl.load(f)
    else:
        print(f"Pickle not found. Computing BMA predictions and saving to: {pickle_path_pred}")
        pred_dict = bma_predictions(bma_dict, number_of_sim=number_of_sim)

        with open(pickle_path_pred, "wb") as f:
            pkl.dump(pred_dict, f)

    # === Step 3: Simulate returns ===
    sim_draws = calculate_bma_returns(pred_dict, number_of_sim=number_of_sim)

    return sim_draws


def rolling_bma_returns(parent_dir, n_predictors_to_use, start_date, end_date, min_obs=120, tau=1.1, number_of_sim=10000) -> OrderedDict[pd.Timestamp, np.ndarray]:
    """
    Rolling BMA simulation:
    - For each month, estimate model (or load from cache)
    - Simulate predictive returns using run_bma_pipeline()
    - Store 10,000 simulated draws in an OrderedDict indexed by date
    """
    # --- Load data ---
    factors_path = os.path.join(parent_dir, "Complemetary Code Files for Submission", "Data", "factors-20.csv")
    predictors_path = os.path.join(parent_dir, "Complemetary Code Files for Submission", "Data", "Z - 197706.csv")

    factors = pd.read_csv(factors_path).drop(columns=['MKTRF', 'SMB*', 'MKT', 'CON', 'IA', 'ROE', 'ME'], errors='ignore')
    predictors = pd.read_csv(predictors_path).drop(columns=['Unnamed: 0'], errors='ignore')

    # --- Parse and align dates ---
    factors['Date'] = pd.to_datetime(factors['Date'].astype(str), format='%Y%m')
    predictors['Date'] = pd.to_datetime(predictors['Date'], format='%m/%d/%Y')

    factors.set_index('Date', inplace=True)
    predictors.set_index('Date', inplace=True)
    factors.sort_index(inplace=True)
    predictors.sort_index(inplace=True)

    # --- Slice by user-defined date range ---
    factors = factors.loc[start_date:end_date]
    predictors = predictors.loc[start_date:end_date]

    # --- Merge and align both datasets ---
    merged = pd.concat([factors, predictors], axis=1, join='inner')
    factor_cols = factors.columns
    predictor_cols = predictors.columns

    # --- Output container ---
    bma_predictions = OrderedDict()
    all_dates = merged.index[min_obs:]

    # --- Ensure cache directory exists ---
    cache_dir = os.path.join(parent_dir, "bma_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # --- Rolling loop ---
    for current_date in all_dates:
        print(f"Processing {current_date.strftime('%Y-%m')}")

        f_slice = merged.loc[:current_date, factor_cols]
        z_slice = merged.loc[:current_date, predictor_cols]

        f_reset = f_slice.reset_index().rename(columns={'index': 'Date'})
        z_reset = z_slice.reset_index().rename(columns={'index': 'Date'})

        date_label = current_date.strftime('%Y-%m')
        pickle_path_init = os.path.join(cache_dir, f"bma_init_{date_label}.pkl")
        pickle_path_pred = os.path.join(cache_dir, f"bma_pred_{date_label}.pkl")

        sim_draws = run_bma_pipeline(
            ff_slice=f_reset,
            zz_slice=z_reset,
            tau=tau,
            pickle_path_init=pickle_path_init,
            pickle_path_pred=pickle_path_pred,
            number_of_sim=number_of_sim
        )

        bma_predictions[current_date] = sim_draws

    return bma_predictions


def prospect_value_BMA(weights, r_s, r_hat, lambda_, gamma=0.1, strategy="conservative", r_tminus1=0.0, r_tminus2=0.0):
    """
    Calculate the prospect value function with dynamic lambda for conservative or aggressive investors.

    Parameters:
    weights (array): The weights for each asset in the portfolio.
    r_s (array): The returns of each asset.
    r_hat (float): The reference return.
    lambda_ (float): The base loss aversion sensitivity coefficient.
    gamma (float): The risk aversion coefficient (default is 0.1).
    strategy (str): The risk strategy of the investor ('conservative' or 'aggressive').

    Returns:
    float: The prospect value.
    """
    # Calculate portfolio returns based on weights
    portfolio_returns = np.dot(r_s, weights) #Used to calculate expected returns
    # er mean return for hver måned ved første kørsel af minimizer (lige vægte)

    # Calculate zt based on these returns
    zt = calculate_zt(np.mean(portfolio_returns), r_tminus1)

    # Dynamically adjust lambda based on strategy and portfolio performance
    if strategy == "conservative":
        lambda_dynamic = calculate_conservative_lambda(r_tminus1, r_tminus2, zt, lambda_)
    elif strategy == "aggressive":
        lambda_dynamic = calculate_aggressive_lambda(r_tminus1, r_tminus2, zt, lambda_)
    else:
        lambda_dynamic = lambda_

    # Calculate prospect value
    S = len(r_s)  # Number of periods
    prospect_value_sum = 0

    for s in range(S):
        r_sx = np.dot(r_s[s], weights)  # Portfolio return for time period s
        
        gain_term = max(0, r_sx - r_hat)
        loss_term = max(0, r_hat - r_sx)

        prospect_value_sum += (gain_term ** (1 - gamma)) / (1 - gamma) - lambda_dynamic * (loss_term ** (1 - gamma)) / (1 - gamma)
        

    return -prospect_value_sum / S  # Negative sign for maximization


def optimize_portfolio(r_s, r_hat, lambda_, strategy="conservative", gamma=0.1, r_tminus1=0.0, r_tminus2=0.0, BMA=False):
    num_assets = len(r_s.T)  # Number of assets in the portfolio

    initial_weights = np.ones(num_assets) / num_assets

    # Convert the equalities and bounds to inequalities for COBYLA:
    constraints = []

    # Equality constraint: sum(x) = 1 transformed into two inequalities:
    # sum(x) - 1 ≥ 0 and 1 - sum(x) ≥ 0
    constraints.append({'type': 'ineq', 'fun': lambda x: np.sum(x) - 1})
    constraints.append({'type': 'ineq', 'fun': lambda x: 1 - np.sum(x)})

    # Bounds: 0 ≤ x[i] ≤ 1 becomes:
    # x[i] ≥ 0  => 'fun': lambda x: x[i]
    # 1 - x[i] ≥ 0 => 'fun': lambda x: 1 - x[i]
    for i in range(num_assets):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[i]})

    if BMA == True:
        # Use the BMA returns
        result = minimize(prospect_value_BMA, initial_weights, 
                          args=(r_s, r_hat, lambda_, gamma, strategy, r_tminus1, r_tminus2),
                          method='COBYLA', constraints=constraints,
                          options={'maxiter': 10000, 'tol': 1e-6})
    else:
        result = minimize(prospect_value, initial_weights, 
                        args=(r_s, r_hat, lambda_, gamma, strategy, r_tminus1, r_tminus2),
                        method='COBYLA', constraints=constraints,
                        options={'maxiter': 1000, 'tol': 1e-6})

    return result

def backtest_portfolio_bma(
    bma_returns: OrderedDict,
    strategy="conservative",
    lambda_=1,
    gamma=0.9,
    pickle_path_backtest: str = None
) -> pd.DataFrame:
    """
    Backtest portfolio using prospect theory on simulated BMA return draws.
    Will cache to pickle if pickle_path_backtest is provided.

    Returns:
        pd.DataFrame with ['Portfolio Returns', 'Compounded Returns', 'Portfolio Weights', 'r_hat']
    """

    if pickle_path_backtest is not None and os.path.exists(pickle_path_backtest):
        print(f"Loading cached backtest from: {pickle_path_backtest}")
        with open(pickle_path_backtest, "rb") as f:
            return pkl.load(f)

    dates = list(bma_returns.keys())
    num_assets = bma_returns[dates[0]].shape[1]

    portfolio_returns = []
    compounded_returns = []
    portfolio_weights = []
    r_hat_values = []

    cumulative_return = 1.0
    r_tminus1 = 0.0
    r_tminus2 = 0.0

    for t, current_date in enumerate(dates[:-1]):
        print(f"Backtesting {current_date.strftime('%Y-%m')}")

        r_s = bma_returns[current_date]
        r_hat = r_tminus1
        r_hat_values.append(r_hat)

        result = optimize_portfolio(
            r_s=r_s,
            r_hat=r_hat,
            lambda_=lambda_,
            strategy=strategy,
            gamma=gamma,
            r_tminus1=r_tminus1,
            r_tminus2=r_tminus2,
            BMA=True
        )
        weights = [round(w, 4) for w in result.x]
        portfolio_weights.append(weights)

        next_date = dates[t + 1]
        next_r_s = bma_returns[next_date]
        portfolio_return_simulated = np.dot(next_r_s, weights)
        portfolio_return = np.mean(portfolio_return_simulated)

        portfolio_returns.append(portfolio_return)
        cumulative_return *= (1 + portfolio_return)
        compounded_returns.append(cumulative_return)

        r_tminus2 = r_tminus1
        r_tminus1 = portfolio_return

    result_df = pd.DataFrame({
        'Portfolio Returns': portfolio_returns,
        'Compounded Returns': compounded_returns,
        'Portfolio Weights': portfolio_weights,
        'r_hat': r_hat_values
    }, index=dates[1:])

    if pickle_path_backtest is not None:
        with open(pickle_path_backtest, "wb") as f:
            pkl.dump(result_df, f)

    return result_df

def resultgenerator_bma(lambda_values, gamma_values, bma_returns, strategies, date_tag: str, cache_dir="./bma_cache"):
    """
    Runs backtests over a grid of (lambda, gamma, strategy) using BMA-simulated returns.
    Each result is cached to a .pkl and returned in a dictionary.
    """
    import os, time
    results_dict = {}
    os.makedirs(cache_dir, exist_ok=True)

    for lambdas in lambda_values:
        print(f"\nλ = {lambdas}")
        for gammas in gamma_values:
            print(f"  γ = {gammas}")
            for strategy in strategies:
                key = f"{strategy}_{lambdas}_{gammas}"
                print(f"    Strategy: {key}")

                pickle_path_bt = os.path.join(
                    cache_dir,
                    f"backtest_{strategy}_lambda{lambdas}_gamma{gammas}_{date_tag}.pkl"
                )

                toc = time.time()
                results = backtest_portfolio_bma(
                    bma_returns=bma_returns,
                    strategy=strategy,
                    lambda_=lambdas,
                    gamma=gammas,
                    pickle_path_backtest=pickle_path_bt
                )
                tic = time.time()
                print(f"    → Completed in {(tic - toc):.2f}s")

                results_dict[key] = results

    return results_dict


def backtest_portfolio_historical_mean(
    historical_returns: pd.DataFrame,
    strategy="conservative",
    lambda_=1,
    gamma=0.9,
    lookback_period: int = 120,  # 10 years of monthly returns
    pickle_path_backtest: str = None
) -> pd.DataFrame:
    """
    Backtest portfolio using historical mean return estimates.
    Uses a rolling lookback window to estimate expected returns.
    """

    if pickle_path_backtest is not None and os.path.exists(pickle_path_backtest):
        print(f"Loading cached historical mean backtest from: {pickle_path_backtest}")
        with open(pickle_path_backtest, "rb") as f:
            return pkl.load(f)

    dates = historical_returns.index
    num_assets = historical_returns.shape[1]

    portfolio_returns = []
    compounded_returns = []
    portfolio_weights = []
    r_hat_values = []

    cumulative_return = 1.0
    r_tminus1 = 0.0
    r_tminus2 = 0.0

    for t in range(lookback_period, len(dates) - 1):
        current_date = dates[t]
        print(f"Backtesting {current_date.strftime('%Y-%m')}")

        # Lookback window
        past_returns = historical_returns.iloc[t - lookback_period:t]

        # Forecast = historical mean
        expected_returns = past_returns.mean()

        # Realized next period returns
        next_returns = historical_returns.iloc[t + 1]

        # Optimize
        result = optimize_portfolio(
            r_s=np.random.multivariate_normal(expected_returns.values, past_returns.cov(), size=10000),
            r_hat=r_tminus1,
            lambda_=lambda_,
            strategy=strategy,
            gamma=gamma,
            r_tminus1=r_tminus1,
            r_tminus2=r_tminus2,
            BMA=False
        )
        weights = [round(w, 4) for w in result.x]
        portfolio_weights.append(weights)

        portfolio_return = np.dot(next_returns.values, weights)
        portfolio_returns.append(portfolio_return)

        cumulative_return *= (1 + portfolio_return)
        compounded_returns.append(cumulative_return)

        r_hat_values.append(r_tminus1)
        r_tminus2 = r_tminus1
        r_tminus1 = portfolio_return

    result_df = pd.DataFrame({
        'Portfolio Returns': portfolio_returns,
        'Compounded Returns': compounded_returns,
        'Portfolio Weights': portfolio_weights,
        'r_hat': r_hat_values
    }, index=dates[lookback_period+1:])

    if pickle_path_backtest is not None:
        with open(pickle_path_backtest, "wb") as f:
            pkl.dump(result_df, f)

    return result_df



def resultgenerator_historical_mean(lambda_values, gamma_values, historical_returns, strategies, date_tag: str, cache_dir="./historical_mean_cache"):
    """
    Runs backtests over a grid of (lambda, gamma, strategy) using Historical Sample Mean returns.
    """
    results_dict = {}
    os.makedirs(cache_dir, exist_ok=True)

    for lambdas in lambda_values:
        print(f"\nλ = {lambdas}")
        for gammas in gamma_values:
            print(f"  γ = {gammas}")
            for strategy in strategies:
                key = f"{strategy}_{lambdas}_{gammas}"
                print(f"    Strategy: {key}")

                pickle_path_bt = os.path.join(
                    cache_dir,
                    f"backtest_historical_mean_{strategy}_lambda{lambdas}_gamma{gammas}_{date_tag}.pkl"
                )

                toc = time.time()
                results = backtest_portfolio_historical_mean(
                    historical_returns=historical_returns,
                    strategy=strategy,
                    lambda_=lambdas,
                    gamma=gammas,
                    pickle_path_backtest=pickle_path_bt
                )
                tic = time.time()
                print(f"    → Completed in {(tic - toc):.2f}s")

                results_dict[key] = results

    return results_dict


def backtest_portfolio_mvp(
    historical_returns: pd.DataFrame,
    strategy="conservative",
    lambda_=1,
    gamma=0.9,
    lookback_period: int = 120,
    pickle_path_backtest: str = None
) -> pd.DataFrame:
    """
    Backtest Minimum Variance Portfolio (MVP) without expected returns.
    NOTE: We don't use optimize_portfolio() here, since returns doesn't matter in MVP, ONLY minimized variance.
    """

    if pickle_path_backtest is not None and os.path.exists(pickle_path_backtest):
        print(f"Loading cached MVP backtest from: {pickle_path_backtest}")
        with open(pickle_path_backtest, "rb") as f:
            return pkl.load(f)

    dates = historical_returns.index
    num_assets = historical_returns.shape[1]

    portfolio_returns = []
    compounded_returns = []
    portfolio_weights = []
    r_hat_values = []

    cumulative_return = 1.0
    r_tminus1 = 0.0
    r_tminus2 = 0.0

    for t in range(lookback_period, len(dates) - 1):
        current_date = dates[t]
        print(f"Backtesting {current_date.strftime('%Y-%m')}")

        # Lookback returns
        past_returns = historical_returns.iloc[t - lookback_period:t]
        cov_matrix = past_returns.cov()

        # === Optimize MVP ===
        def portfolio_variance(weights):
            return weights.T @ cov_matrix.values @ weights

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(num_assets)]
        initial_weights = np.ones(num_assets) / num_assets

        result = minimize(portfolio_variance, initial_weights, bounds=bounds, constraints=constraints)

        weights = [round(w, 4) for w in result.x]
        portfolio_weights.append(weights)

        # === Next period realized returns
        next_returns = historical_returns.iloc[t + 1]

        portfolio_return = np.dot(next_returns.values, weights)
        portfolio_returns.append(portfolio_return)

        cumulative_return *= (1 + portfolio_return)
        compounded_returns.append(cumulative_return)

        r_hat_values.append(r_tminus1)
        r_tminus2 = r_tminus1
        r_tminus1 = portfolio_return

    result_df = pd.DataFrame({
        'Portfolio Returns': portfolio_returns,
        'Compounded Returns': compounded_returns,
        'Portfolio Weights': portfolio_weights,
        'r_hat': r_hat_values
    }, index=dates[lookback_period+1:])

    if pickle_path_backtest is not None:
        with open(pickle_path_backtest, "wb") as f:
            pkl.dump(result_df, f)

    return result_df


def resultgenerator_mvp(lambda_values, gamma_values, historical_returns, strategies, date_tag: str, cache_dir="./mvp_cache"):
    """
    Runs MVP backtests over (lambda, gamma, strategy) grid.
    """
    results_dict = {}
    os.makedirs(cache_dir, exist_ok=True)

    for lambdas in lambda_values:
        print(f"\nλ = {lambdas}")
        for gammas in gamma_values:
            print(f"  γ = {gammas}")
            for strategy in strategies:
                key = f"{strategy}_{lambdas}_{gammas}"
                print(f"    Strategy: {key}")

                pickle_path_bt = os.path.join(
                    cache_dir,
                    f"backtest_mvp_{strategy}_lambda{lambdas}_gamma{gammas}_{date_tag}.pkl"
                )

                toc = time.time()
                results = backtest_portfolio_mvp(
                    historical_returns=historical_returns,
                    strategy=strategy,
                    lambda_=lambdas,
                    gamma=gammas,
                    pickle_path_backtest=pickle_path_bt
                )
                tic = time.time()
                print(f"    → Completed in {(tic - toc):.2f}s")

                results_dict[key] = results

    return results_dict

### Fama-French 5-Factor Model

def estimate_factor_model_expected_returns(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    lookback_window: int = 120
) -> pd.Series:
    """
    Estimates expected returns for assets using a linear factor model.

    Args:
        asset_returns: DataFrame of asset returns (e.g., SMB, HML, RMW)
        factor_returns: DataFrame of factor returns (e.g., Mkt-RF, IA, ROE)
        lookback_window: number of months to look back for estimation

    Returns:
        pd.Series: Expected returns per asset
    """

    # Restrict to lookback window
    asset_returns = asset_returns.iloc[-lookback_window:]
    factor_returns = factor_returns.iloc[-lookback_window:]

    expected_returns = {}

    for asset in asset_returns.columns:
        y = asset_returns[asset].values
        X = factor_returns.values

        # Fit OLS regression
        model = LinearRegression().fit(X, y)
        betas = model.coef_

        # Estimate expected factor returns
        expected_factor_premia = factor_returns.mean().values

        # Implied expected return for the asset
        implied_return = np.dot(betas, expected_factor_premia)
        expected_returns[asset] = implied_return

    return pd.Series(expected_returns)

def backtest_portfolio_factor_model(
    test_assets_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    strategy="conservative",
    lambda_=1,
    gamma=0.9,
    lookback_period: int = 120,
    pickle_path_backtest: str = None
) -> pd.DataFrame:
    """
    Backtest using expected returns derived from a factor model (modified Fama-French structure).
    """

    if pickle_path_backtest is not None and os.path.exists(pickle_path_backtest):
        print(f"Loading cached factor model backtest from: {pickle_path_backtest}")
        with open(pickle_path_backtest, "rb") as f:
            return pkl.load(f)

    dates = test_assets_returns.index
    num_assets = test_assets_returns.shape[1]

    portfolio_returns = []
    compounded_returns = []
    portfolio_weights = []
    r_hat_values = []

    cumulative_return = 1.0
    r_tminus1 = 0.0
    r_tminus2 = 0.0

    for t in range(lookback_period, len(dates) - 1):
        current_date = dates[t]
        print(f"Backtesting {current_date.strftime('%Y-%m')}")

        # Lookback window
        past_asset_returns = test_assets_returns.iloc[t - lookback_period:t]
        past_factor_returns = factor_returns.iloc[t - lookback_period:t]

        # === Step 1: Estimate Betas and Expected Returns ===
        expected_returns = {}

        for asset in past_asset_returns.columns:
            y = past_asset_returns[asset].values
            X = past_factor_returns.values

            model = LinearRegression().fit(X, y)
            betas = model.coef_

            expected_factor_premia = past_factor_returns.mean().values
            implied_return = np.dot(betas, expected_factor_premia)

            expected_returns[asset] = implied_return

        expected_returns = pd.Series(expected_returns)

        # === Step 2: Simulate returns ===
        cov_matrix = past_asset_returns.cov()
        simulated_returns = np.random.multivariate_normal(expected_returns.values, cov_matrix.values, size=10000)

        # === Step 3: Optimize portfolio ===
        result = optimize_portfolio(
            r_s=simulated_returns,
            r_hat=r_tminus1,
            lambda_=lambda_,
            strategy=strategy,
            gamma=gamma,
            r_tminus1=r_tminus1,
            r_tminus2=r_tminus2,
            BMA=False
        )
        weights = [round(w, 4) for w in result.x]
        portfolio_weights.append(weights)

        # === Step 4: Realized next-period return
        next_returns = test_assets_returns.iloc[t + 1]

        portfolio_return = np.dot(next_returns.values, weights)
        portfolio_returns.append(portfolio_return)

        cumulative_return *= (1 + portfolio_return)
        compounded_returns.append(cumulative_return)

        r_hat_values.append(r_tminus1)
        r_tminus2 = r_tminus1
        r_tminus1 = portfolio_return

    result_df = pd.DataFrame({
        'Portfolio Returns': portfolio_returns,
        'Compounded Returns': compounded_returns,
        'Portfolio Weights': portfolio_weights,
        'r_hat': r_hat_values
    }, index=dates[lookback_period+1:])

    if pickle_path_backtest is not None:
        with open(pickle_path_backtest, "wb") as f:
            pkl.dump(result_df, f)

    return result_df

def resultgenerator_factor_model(
    lambda_values,
    gamma_values,
    test_assets_returns,
    factor_returns,
    strategies,
    date_tag: str,
    cache_dir="./factor_model_cache"
) -> dict[str, pd.DataFrame]:
    """
    Runs factor model-based backtests over (lambda, gamma, strategy) grid.
    """

    results_dict = {}
    os.makedirs(cache_dir, exist_ok=True)

    for lambdas in lambda_values:
        print(f"\nλ = {lambdas}")
        for gammas in gamma_values:
            print(f"  γ = {gammas}")
            for strategy in strategies:
                key = f"{strategy}_{lambdas}_{gammas}"
                print(f"    Strategy: {key}")

                pickle_path_bt = os.path.join(
                    cache_dir,
                    f"backtest_factor_model_{strategy}_lambda{lambdas}_gamma{gammas}_{date_tag}.pkl"
                )

                toc = time.time()
                results = backtest_portfolio_factor_model(
                    test_assets_returns=test_assets_returns,
                    factor_returns=factor_returns,
                    strategy=strategy,
                    lambda_=lambdas,
                    gamma=gammas,
                    pickle_path_backtest=pickle_path_bt
                )
                tic = time.time()
                print(f"    → Completed in {(tic - toc):.2f}s")

                results_dict[key] = results

    return results_dict



def backtest_portfolio_dynamic_rhat(
    bma_returns: OrderedDict,
    strategy="conservative",
    lambda_=1,
    gamma=0.9,
    reference_rule: str = "prev_portfolio",  # NEW: dynamic reference point
    market_returns: dict = None,             # optional, if using market as reference
    cache_dir: str = "./bma_cache"
) -> pd.DataFrame:
    """
    Backtest portfolio using prospect theory on simulated BMA return draws,
    allowing for different reference point definitions and automatic caching.

    Args:
        bma_returns: OrderedDict[date, np.ndarray] of simulated returns
        strategy: 'conservative' or 'aggressive'
        lambda_: loss aversion
        gamma: risk aversion
        reference_rule: 'prev_portfolio', 'fixed_zero', 'market_return', 'rolling_avg'
        market_returns: optional dict[pd.Timestamp, float]
        cache_dir: folder to store cached backtest pickle

    Returns:
        pd.DataFrame with ['Portfolio Returns', 'Compounded Returns', 'Portfolio Weights', 'r_hat']
    """

    # === Create cache path ===
    os.makedirs(cache_dir, exist_ok=True)
    filename = f"bt_{strategy}_lambda{lambda_}_gamma{gamma}_{reference_rule}.pkl"
    cache_path = os.path.join(cache_dir, filename)

    # === Load from cache if exists ===
    if os.path.exists(cache_path):
        print(f"📂 Loading cached backtest from: {cache_path}")
        with open(cache_path, "rb") as f:
            return pkl.load(f)

    # === Begin computation ===
    dates = list(bma_returns.keys())
    num_assets = bma_returns[dates[0]].shape[1]

    portfolio_returns = []
    compounded_returns = []
    portfolio_weights = []
    r_hat_values = []

    cumulative_return = 1.0
    r_tminus1 = 0.0
    r_tminus2 = 0.0

    for t, current_date in enumerate(dates[:-1]):
        print(f"🔄 Backtesting {current_date.strftime('%Y-%m')}")

        r_s = bma_returns[current_date]  # shape: (num_sim, num_assets)

        # === Define r_hat based on reference rule ===
        if reference_rule == "prev_portfolio":
            r_hat = r_tminus1
        elif reference_rule == "fixed_zero":
            r_hat = 0.0
        elif reference_rule == "market_return":
            r_hat = market_returns.get(current_date, 0.0) if market_returns else 0.0
        elif reference_rule == "rolling_avg":
            r_hat = np.mean(portfolio_returns[-3:]) if len(portfolio_returns) >= 3 else 0.0
        else:
            raise ValueError(f"Unknown reference rule: {reference_rule}")

        r_hat_values.append(r_hat)

        # === Optimize ===
        result = optimize_portfolio(
            r_s=r_s,
            r_hat=r_hat,
            lambda_=lambda_,
            strategy=strategy,
            gamma=gamma,
            r_tminus1=r_tminus1,
            r_tminus2=r_tminus2,
            BMA=True
        )
        weights = [round(w, 4) for w in result.x]
        portfolio_weights.append(weights)

        # === Next period returns ===
        next_date = dates[t + 1]
        next_r_s = bma_returns[next_date]
        portfolio_return_simulated = np.dot(next_r_s, weights)
        portfolio_return = np.mean(portfolio_return_simulated)

        portfolio_returns.append(portfolio_return)
        cumulative_return *= (1 + portfolio_return)
        compounded_returns.append(cumulative_return)

        r_tminus2 = r_tminus1
        r_tminus1 = portfolio_return

    # === Build result DataFrame ===
    result_df = pd.DataFrame({
        'Portfolio Returns': portfolio_returns,
        'Compounded Returns': compounded_returns,
        'Portfolio Weights': portfolio_weights,
        'r_hat': r_hat_values
    }, index=dates[1:])

    # === Save to cache ===
    with open(cache_path, "wb") as f:
        pkl.dump(result_df, f)
        print(f"✅ Saved backtest to: {cache_path}")

    return result_df



def resultgenerator_bma_dynamic_rhat(
    lambda_values,
    gamma_values,
    bma_returns,
    strategies,
    date_tag: str,
    cache_dir="./bma_cache",
    reference_rule="prev_portfolio",
    market_returns: dict = None
) -> dict[str, pd.DataFrame]:
    """
    Runs backtests over a grid of (lambda, gamma, strategy) using BMA-simulated returns.
    Uses internal caching and supports custom reference point definitions.

    Returns:
        results_dict: strategy_key → DataFrame of backtest results
    """
    results_dict = {}
    os.makedirs(cache_dir, exist_ok=True)

    for lambdas in lambda_values:
        print(f"\nλ = {lambdas}")
        for gammas in gamma_values:
            print(f"  γ = {gammas}")
            for strategy in strategies:
                key = f"{strategy}_{lambdas}_{gammas}"
                print(f"    Strategy: {key} — ref: {reference_rule}")

                toc = time.time()

                # Run backtest (automatically cached)
                results = backtest_portfolio_bma(
                    bma_returns=bma_returns,
                    strategy=strategy,
                    lambda_=lambdas,
                    gamma=gammas,
                    reference_rule=reference_rule,
                    market_returns=market_returns,
                    cache_dir=cache_dir
                )

                tic = time.time()
                print(f"    → Completed in {(tic - toc):.2f}s")

                results_dict[key] = results

    return results_dict


def parse_strategy_key(key: str) -> tuple:
    strategy, lam, gamma = key.split("_")
    return strategy, float(lam), float(gamma)


def summarize_backtest_results(results_dict: dict) -> pd.DataFrame:
    """
    Summarizes backtest performance across strategies.
    
    Args:
        results_dict (dict): Output from resultgenerator_bma(), 
                             where keys are strategy_lambda_gamma identifiers 
                             and values are result DataFrames.
    
    Returns:
        pd.DataFrame with summary metrics per strategy/parameter combo.
    """
    summary_rows = []

    for key, df in results_dict.items():
        returns = df['Portfolio Returns']
        compounded = df['Compounded Returns']

        # Performance metrics
        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe = mean_ret / std_ret if std_ret != 0 else np.nan
        final_wealth = compounded.iloc[-1]
        min_wealth = compounded.min()
        max_drawdown = 1 - (min_wealth / compounded.cummax()).min()

        strategy, lam, gamma = parse_strategy_key(key)

        summary_rows.append({
            'Strategy_Key': key,
            'Strategy': strategy,
            'Lambda': lam,
            'Gamma': gamma,
            'Mean Return': mean_ret,
            'Std Dev': std_ret,
            'Sharpe Ratio': sharpe,
            'Final Wealth': final_wealth,
            'Max Drawdown': max_drawdown
        })

    summary_df = pd.DataFrame(summary_rows).set_index('Strategy_Key')
    return summary_df.sort_values(by='Sharpe Ratio', ascending=False)


def compare_methods(results_by_method: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Compares multiple backtest methods across strategies.
    
    Args:
        results_by_method: dictionary where 
            key = method name (e.g., 'BMA', 'Historical Mean', 'Shrinkage')
            value = corresponding results_dict (strategy_key -> DataFrame)

    Returns:
        pd.DataFrame: combined summary with Method labels
    """
    all_summaries = []

    for method_name, results_dict in results_by_method.items():
        summary = summarize_backtest_results(results_dict)
        summary["Method"] = method_name
        all_summaries.append(summary.reset_index())

    # Combine all summaries
    combined_summary = pd.concat(all_summaries)

    # Optional: sort nicely
    combined_summary = combined_summary.sort_values(by=["Strategy", "Lambda", "Gamma", "Method"]).set_index("Strategy_Key")

    return combined_summary





def evaluate_reference_robustness(
    lambda_values,
    gamma_values,
    bma_returns,
    strategies,
    date_tag,
    parent_dir,
    reference_rules=("prev_portfolio", "fixed_zero", "market_return", "spy")
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Evaluates backtests and CE across multiple reference point rules.

    Returns:
        dict: {reference_rule: results_dict}
              where each results_dict is {strategy_key: result_df}
    """
    all_results_by_reference = {}

    for ref_rule in reference_rules:
        print(f"\n🧭 Evaluating reference: {ref_rule}")

        # Load required reference series if applicable
        if ref_rule in {"market_return", "spy"}:
            market_returns = load_reference_returns(parent_dir, ref_type=ref_rule.split("_")[0])
        else:
            market_returns = None

        results_dict = resultgenerator_bma(
            lambda_values=lambda_values,
            gamma_values=gamma_values,
            bma_returns=bma_returns,
            strategies=strategies,
            date_tag=date_tag,
            cache_dir="./bma_cache",
            reference_rule=ref_rule,
            market_returns=market_returns
        )

        all_results_by_reference[ref_rule] = results_dict

    return all_results_by_reference









def perform_mann_whitney(specific_portfolios):
    """
    Perform the Mann-Whitney U test to check if the returns of specific portfolio strategies are statistically different.

    Parameters:
    specific_portfolios (list of tuples): List of tuples containing pairs of portfolio DataFrames and their names.

    Returns:
    pd.DataFrame: DataFrame containing the results of the Mann-Whitney U tests.
    """
    test_results = []

    for i, ((a_name, a_df), (c_name, c_df)) in enumerate(specific_portfolios, start=1):
        # Perform Mann-Whitney U test on the portfolio returns
        stat, p_value = mannwhitneyu(
            a_df['Portfolio Returns'].dropna(),
            c_df['Portfolio Returns'].dropna(),
            alternative='two-sided'
        )

        # Collect the test results
        test_results.append({
            'Test Number': i,
            'Aggressive Strategy': a_name,
            'Conservative Strategy': c_name,
            'U Statistic': stat,
            'p-value': p_value,
            'Significant': p_value < 0.05  # Significance threshold of 0.05
        })

    # Convert results to a DataFrame
    return pd.DataFrame(test_results)

def test_volatility(specific_portfolios):
    """
    Perform Levene's test to compare the variance (volatility) of portfolio returns.

    Parameters:
    specific_portfolios (list of tuples): List of tuples containing pairs of portfolio DataFrames and their names.

    Returns:
    pd.DataFrame: DataFrame containing the results of the Levene's tests.
    """
    volatility_results = []

    for i, ((a_name, a_df), (c_name, c_df)) in enumerate(specific_portfolios, start=1):
        # Perform Levene's test on the portfolio returns
        stat, p_value = levene(
            a_df['Portfolio Returns'].dropna(),
            c_df['Portfolio Returns'].dropna()
        )

        # Collect the test resultsfrom scipy.stats import mannwhitneyu, levene, f_oneway

        volatility_results.append({
            'Test Number': i,
            'Aggressive Strategy': a_name,
            'Conservative Strategy': c_name,
            'Levene Statistic': stat,
            'p-value': p_value,
            'Equal Variance': p_value >= 0.05  # True if variances are not significantly different
        })

    # Convert results to a DataFrame
    return pd.DataFrame(volatility_results)



def perform_t_test(specific_portfolios):
    """
    Perform the t-test to check if the mean of log-transformed returns are statistically different.

    Parameters:
    specific_portfolios (list of tuples): List of tuples containing pairs of portfolio DataFrames and their names.

    Returns:
    pd.DataFrame: DataFrame containing the results of the t-tests.
    """
    t_test_results = []

    for i, ((a_name, a_df), (c_name, c_df)) in enumerate(specific_portfolios, start=1):
        # Perform t-test on log-transformed portfolio returns
        a_returns = np.log(1 + a_df['Portfolio Returns'].dropna())
        c_returns = np.log(1 + c_df['Portfolio Returns'].dropna())

        stat, p_value = ttest_ind(conservative_results[4]["conservative_result_lambda1.5_gamma0.5"],["aggressive_result_lambda1.5_gamma0.12"], c_returns, equal_var=False)  # Welch's t-test

        t_test_results.append({
            'Test Number': i,
            'Aggressive Strategy': a_name,
            'Conservative Strategy': c_name,
            'T Statistic': stat,
            'p-value': p_value,
            'Significant': p_value < 0.05
        })

    return pd.DataFrame(t_test_results)

def perform_f_test(specific_portfolios):
    """
    Perform the F-test (ANOVA) to test for equality of variances between groups.

    Parameters:
    specific_portfolios (list of tuples): List of tuples containing pairs of portfolio DataFrames and their names.

    Returns:
    pd.DataFrame: DataFrame containing the results of the F-test.
    """
    f_test_results = []

    for i, ((a_name, a_df), (c_name, c_df)) in enumerate(specific_portfolios, start=1):
        # Perform ANOVA F-test on the log-transformed returns
        a_returns = np.log(1 + a_df['Portfolio Returns'].dropna())
        c_returns = np.log(1 + c_df['Portfolio Returns'].dropna())

        stat, p_value = f_oneway(a_returns, c_returns)

        f_test_results.append({
            'Test Number': i,
            'Aggressive Strategy': a_name,
            'Conservative Strategy': c_name,
            'F Statistic': stat,
            'p-value': p_value,
            'Equal Variance': p_value >= 0.05
        })

    return pd.DataFrame(f_test_results)


def calculate_portfolio_stats(aggressive_results, conservative_results):
    """
    Calculate mean returns and return volatility for all portfolios.
    
    Parameters:
    aggressive_results (list of dicts): List of dictionaries containing aggressive portfolio results.
    conservative_results (list of dicts): List of dictionaries containing conservative portfolio results.
    
    Returns:
    pd.DataFrame: DataFrame containing portfolio names, mean returns, and return volatilities.
    """
    portfolios_stats = []

    # Process aggressive portfolios
    for result in aggressive_results:
        for name, data in result.items():
            
            mean_return = data['Portfolio Returns'].mean()
            return_volatility = data['Portfolio Returns'].std()
            portfolios_stats.append({
                'Portfolio Name': name,
                'Type': 'Aggressive',
                'Mean Return': mean_return,
                'Return Volatility': return_volatility
            })
            print(mean_return, return_volatility)

    # Process conservative portfolios
    for result in conservative_results:
        for name, data in result.items():
            mean_return = data['Portfolio Returns'].mean()
            return_volatility = data['Portfolio Returns'].std()
            portfolios_stats.append({
                'Portfolio Name': name,
                'Type': 'Conservative',
                'Mean Return': mean_return,
                'Return Volatility': return_volatility
            })
            print(mean_return, return_volatility)

    # Convert results to a DataFrame
    return pd.DataFrame(portfolios_stats)


# Function to calculate yearly alpha and beta with Plotly
def calculate_yearly_alpha_beta_plotly(df):
    # Ensuring 'Date' is in datetime format and setting it as index if not already done
    df['Date'] = pd.to_datetime(df.index)
    df.set_index('Date', inplace=True)
    
    # Calculating yearly alpha and beta
    yearly_alpha_beta = []
    for year, group in df.groupby(df.index.year):
        X = sm.add_constant(group['SPY'])
        y = group['Portfolio Returns']
        model = sm.OLS(y, X).fit()
        alpha, beta = model.params['const'], model.params['SPY']
        yearly_alpha_beta.append({'Year': year, 'Alpha': alpha, 'Beta': beta})
    
    # Converting to DataFrame
    yearly_df = pd.DataFrame(yearly_alpha_beta)
    
    # Calculating overall alpha and beta for reference
    X_overall = sm.add_constant(df['SPY'])
    y_overall = df['Portfolio Returns']
    model_overall = sm.OLS(y_overall, X_overall).fit()
    overall_alpha, overall_beta = model_overall.params['const'], model_overall.params['SPY']
    
    # Plotly graph for Yearly Alpha
    fig_alpha = go.Figure()
    fig_alpha.add_trace(go.Scatter(
        x=yearly_df['Year'], y=yearly_df['Alpha'], mode='lines+markers', name='Yearly Alpha'
    ))
    fig_alpha.add_hline(
        y=overall_alpha, line_dash="dash", line_color="red",
        annotation_text=f"Overall Alpha ({overall_alpha:.4f})", annotation_position="bottom left"
    )
    fig_alpha.update_layout(
        title='Yearly Alpha',
        xaxis_title='Year',
        yaxis_title='Alpha'
    )
    
    # Plotly graph for Yearly Beta
    fig_beta = go.Figure()
    fig_beta.add_trace(go.Scatter(
        x=yearly_df['Year'], y=yearly_df['Beta'], mode='lines+markers', name='Yearly Beta'
    ))
    fig_beta.add_hline(
        y=overall_beta, line_dash="dash", line_color="red",
        annotation_text=f"Overall Beta ({overall_beta:.4f})", annotation_position="bottom left"
    )
    fig_beta.update_layout(
        title='Yearly Beta',
        xaxis_title='Year',
        yaxis_title='Beta'
    )
    
    # Display data tables for Alpha and Beta
    alpha_table = yearly_df[['Year', 'Alpha']]
    beta_table = yearly_df[['Year', 'Beta']]
    
    return fig_alpha, fig_beta, alpha_table, beta_table, overall_alpha, overall_beta

# Function to calculate and store averages
def calculate_strategy_average(category):
    if aggregated_results[category]:
        # Numeric columns only
        numeric_columns = [
            col for col in aggregated_results[category][0].columns
            if pd.api.types.is_numeric_dtype(aggregated_results[category][0][col])
        ]
        
        # Combine numeric columns for the category
        average_df = pd.concat([df[numeric_columns] for df in aggregated_results[category]]).groupby("Date").mean()
        
        # Calculate average portfolio weights
        weights = pd.concat(
            [pd.DataFrame(df['Portfolio Weights'].tolist(), index=df.index) for df in aggregated_results[category]]
        )
        average_weights = weights.groupby(weights.index).mean()
        
        # Calculate alpha and beta for the average portfolio
        fig_alpha, fig_beta, alpha_table, beta_table, overall_alpha, overall_beta = calculate_yearly_alpha_beta_plotly(average_df)
        
        # Store in results_dict
        results_dict[f"{category}_average"] = {
            "fig_alpha": fig_alpha,
            "fig_beta": fig_beta,
            "alpha_table": alpha_table,
            "beta_table": beta_table,
            "overall_alpha": overall_alpha,
            "overall_beta": overall_beta,
            "average_weights": average_weights
        }


# Function to calculate yearly alpha, beta, and variance
def calculate_yearly_alpha_beta_variance(df):
    # Ensuring 'Date' is in datetime format and setting it as index if not already done
    df['Date'] = pd.to_datetime(df.index)
    df.set_index('Date', inplace=True)
    
    # Calculating yearly alpha, beta, and variance
    yearly_alpha_beta_var = []
    for year, group in df.groupby(df.index.year):
        X = sm.add_constant(group['SPY'])
        y = group['Portfolio Returns']
        model = sm.OLS(y, X).fit()
        alpha, beta = model.params['const'], model.params['SPY']
        variance = y.var(ddof=1)  # sample variance
        
        yearly_alpha_beta_var.append({'Year': year, 'Alpha': alpha, 'Beta': beta, 'Variance': variance})
    
    # Converting to DataFrame
    yearly_df = pd.DataFrame(yearly_alpha_beta_var)
    
    # Calculating overall alpha, beta, and variance
    X_overall = sm.add_constant(df['SPY'])
    y_overall = df['Portfolio Returns']
    model_overall = sm.OLS(y_overall, X_overall).fit()
    overall_alpha, overall_beta = model_overall.params['const'], model_overall.params['SPY']
    overall_variance = y_overall.var(ddof=1)
    
    return yearly_df, overall_alpha, overall_beta, overall_variance

# Function to create Plotly figures for alpha and beta
def create_alpha_beta_figs(yearly_df, overall_alpha, overall_beta):
    # Plotly graph for Yearly Alpha
    fig_alpha = go.Figure()
    fig_alpha.add_trace(go.Scatter(
        x=yearly_df['Year'], y=yearly_df['Alpha'], mode='lines+markers', name='Yearly Alpha'
    ))
    fig_alpha.add_hline(
        y=overall_alpha, line_dash="dash", line_color="red",
        annotation_text=f"Overall Alpha ({overall_alpha:.4f})", annotation_position="bottom left"
    )
    fig_alpha.update_layout(
        title='Yearly Alpha',
        xaxis_title='Year',
        yaxis_title='Alpha'
    )
    
    # Plotly graph for Yearly Beta
    fig_beta = go.Figure()
    fig_beta.add_trace(go.Scatter(
        x=yearly_df['Year'], y=yearly_df['Beta'], mode='lines+markers', name='Yearly Beta'
    ))
    fig_beta.add_hline(
        y=overall_beta, line_dash="dash", line_color="red",
        annotation_text=f"Overall Beta ({overall_beta:.4f})", annotation_position="bottom left"
    )
    fig_beta.update_layout(
        title='Yearly Beta',
        xaxis_title='Year',
        yaxis_title='Beta'
    )
    
    return fig_alpha, fig_beta


# Function to calculate and store averages for overall, aggressive, and conservative
def calculate_strategy_average(category):
    aggregated_results = {"overall": [], "aggressive": [], "conservative": []}
    if aggregated_results[category]:
        # Numeric columns only (for averaging)
        numeric_columns = [
            col for col in aggregated_results[category][0].columns
            if pd.api.types.is_numeric_dtype(aggregated_results[category][0][col])
        ]
        
        # Combine numeric columns for the category
        average_df = pd.concat([df[numeric_columns] for df in aggregated_results[category]]).groupby("Date").mean()
        
        # Calculate average portfolio weights (if 'Portfolio Weights' was provided as a column)
        if 'Portfolio Weights' in aggregated_results[category][0].columns:
            weights = pd.concat(
                [pd.DataFrame(df['Portfolio Weights'].tolist(), index=df.index) for df in aggregated_results[category]]
            )
            average_weights = weights.groupby(weights.index).mean()
        else:
            average_weights = pd.DataFrame()
        
        # Calculate alpha, beta, and variance for the average portfolio
        yearly_df, overall_alpha, overall_beta, overall_variance = calculate_yearly_alpha_beta_variance(average_df)
        fig_alpha, fig_beta = create_alpha_beta_figs(yearly_df, overall_alpha, overall_beta)
        
        # Extract tables
        alpha_table = yearly_df[['Year', 'Alpha']]
        beta_table = yearly_df[['Year', 'Beta']]
        variance_table = yearly_df[['Year', 'Variance']]
        
        # Store in results_dict
        results_dict[f"{category}_average"] = {
            "fig_alpha": fig_alpha,
            "fig_beta": fig_beta,
            "alpha_table": alpha_table,
            "beta_table": beta_table,
            "variance_table": variance_table,
            "overall_alpha": overall_alpha,
            "overall_beta": overall_beta,
            "overall_variance": overall_variance,
            "average_weights": average_weights
        }


# Function to plot all portfolios and averages
def plot_all_portfolios(results, category="Portfolio Returns"):
    fig_all = go.Figure()

    # Containers for averages
    aggressive_returns = []
    conservative_returns = []

    # Loop through the results for each portfolio
    for strategy_list in results:
        for result in strategy_list:
            for name, df in result.items():
                # Convert returns to percentage format
                #df[f"{category} (%)"] = df[category] * 100
                df['Compounded Returns (%)'] = (df['Compounded Returns'] - 1) * 100

                # Add to figure
                fig_all.add_trace(go.Scatter(
                    x=df.index,
                    y=df[f"{category} (%)"],
                    mode='lines',
                    name=name
                ))

                # Collect data for averages
                if "aggressive" in name:
                    aggressive_returns.append(df[category])
                elif "conservative" in name:
                    conservative_returns.append(df[category])

    # Calculate averages for aggressive and conservative portfolios
    if aggressive_returns:
        aggressive_average = (pd.concat(aggressive_returns, axis=1).mean(axis=1)-1) * 100
        fig_all.add_trace(go.Scatter(
            x=aggressive_average.index,
            y=aggressive_average,
            mode='lines',
            name='Aggressive Average',
            line=dict(dash='dot', width=3, color='red')
        ))

    if conservative_returns:
        conservative_average = (pd.concat(conservative_returns, axis=1).mean(axis=1)-1) * 100
        fig_all.add_trace(go.Scatter(
            x=conservative_average.index,
            y=conservative_average,
            mode='lines',
            name='Conservative Average',
            line=dict(dash='dash', width=3, color='green')
        ))

    # Update layout
    fig_all.update_layout(
        title=f'{category} for All Portfolios',
        xaxis_title='Date',
        yaxis_title=f'{category} (%)',
        template='plotly_white',
        legend_title='Portfolios',
        xaxis=dict(tickangle=-45),
        yaxis=dict(showgrid=True)
    )

    return fig_all


def calculate_portfolio_statistics_and_averages(results, category="returns"):

    portfolio_statistics = []

    aggressive_returns = []

    conservative_returns = []



    # Loop through the results for each portfolio

    for strategy_list in results:

        for result in strategy_list:

            for name, df in result.items():

                # Calculate mean return and return volatility
                
                mean_return = df[category].mean()

                return_volatility = df[category].std()



                # Append to the statistics list

                portfolio_statistics.append({

                    "Portfolio": name,

                    "Mean Return": mean_return,

                    "Return Volatility": return_volatility

                })



                # Collect data for averages

                if "aggressive" in name.lower():

                    aggressive_returns.append(df[category])

                elif "conservative" in name.lower():

                    conservative_returns.append(df[category])



    # Calculate average statistics for aggressive and conservative portfolios

    if aggressive_returns:
        
        avg_aggressive_mean = pd.concat(aggressive_returns, axis=1).mean(axis=1).mean()

        avg_aggressive_volatility = pd.concat(aggressive_returns, axis=1).mean(axis=1).std()

        portfolio_statistics.append({

            "Portfolio": "Aggressive Average",

            "Mean Return": avg_aggressive_mean,

            "Return Volatility": avg_aggressive_volatility

        })



    if conservative_returns:

        avg_conservative_mean = pd.concat(conservative_returns, axis=1).mean(axis=1).mean()

        avg_conservative_volatility = pd.concat(conservative_returns, axis=1).mean(axis=1).std()

        portfolio_statistics.append({

            "Portfolio": "Conservative Average",

            "Mean Return": avg_conservative_mean,

            "Return Volatility": avg_conservative_volatility

        })



    # Convert to a DataFrame for exporting

    stats_df = pd.DataFrame(portfolio_statistics)



    # Sort by highest to lowest mean return

    stats_df.sort_values(by="Mean Return", ascending=False, inplace=True)

    return stats_df



# Function to plot only the averages for aggressive and conservative portfolios
def plot_averages_only(results, category="Portfolio Returns"):
    fig_averages = go.Figure()

    # Containers for averages
    aggressive_returns = []
    conservative_returns = []

    # Loop through the results to collect data for averages
    for strategy_list in results:
        for result in strategy_list:
            for name, df in result.items():
                # Collect data for aggressive and conservative portfolios only
                if "aggressive" in name:
                    aggressive_returns.append(df[category])
                elif "conservative" in name:
                    conservative_returns.append(df[category])

    # Calculate averages for aggressive and conservative portfolios
    if aggressive_returns:
        aggressive_average = (pd.concat(aggressive_returns, axis=1).mean(axis=1) - 1) * 100
        fig_averages.add_trace(go.Scatter(
            x=aggressive_average.index,
            y=aggressive_average,
            mode='lines',
            name='Aggressive Average',
            line=dict(dash='dot', width=3, color='red')
        ))

    if conservative_returns:
        conservative_average = (pd.concat(conservative_returns, axis=1).mean(axis=1) - 1) * 100
        fig_averages.add_trace(go.Scatter(
            x=conservative_average.index,
            y=conservative_average,
            mode='lines',
            name='Conservative Average',
            line=dict(dash='dash', width=3, color='green')
        ))

    # Update layout
    fig_averages.update_layout(
        title=f'{category} Averages for Aggressive and Conservative Portfolios',
        xaxis_title='Date',
        yaxis_title=f'{category} (%)',
        template='plotly_white',
        legend_title='Portfolios',
        xaxis=dict(tickangle=-45),
        yaxis=dict(showgrid=True)
    )

    return fig_averages


# Function to create individual strategy asset weighting charts and separate plots for aggressive/conservative averages
def plot_weights_and_separate_averages(results, returns_columns):
    # Dictionary to store individual strategy plots
    strategy_weight_figures = {}

    # Containers for averages
    aggressive_weights = []
    conservative_weights = []

    # Loop through each portfolio to create individual strategy plots
    for strategy_list in results:
        for result in strategy_list:
            for name, df in result.items():
                # Convert the Portfolio Weights column to a DataFrame
                weights_df = pd.DataFrame(df['Portfolio Weights'].tolist(),
                                          index=df.index,
                                          columns=returns_columns)

                # Create a new Plotly figure for the strategy weights
                fig = go.Figure()

                # Add traces for each asset in the portfolio
                for asset in weights_df.columns:
                    fig.add_trace(go.Scatter(
                        x=weights_df.index,
                        y=weights_df[asset] * 100,  # Convert to percentage
                        mode='none',  # No lines, just filled areas
                        stackgroup='one',  # Enables stacking of areas
                        name=f'{asset} Weight (%)',
                        hoverinfo='x+y'  # Display date and percentage on hover
                    ))

                # Update layout for clean visualization
                fig.update_layout(
                    title=f'Asset Weights for Strategy: {name}',
                    xaxis_title='Date',
                    yaxis_title='Weight (%)',
                    legend_title='Assets',
                    template='plotly_white',
                    xaxis=dict(tickangle=-45),
                    yaxis=dict(range=[0, 100])  # Ensures the y-axis ranges from 0 to 100%
                )

                # Save the figure to the dictionary
                strategy_weight_figures[name] = fig

                # Collect data for averages
                if "aggressive" in name:
                    aggressive_weights.append(weights_df)
                elif "conservative" in name:
                    conservative_weights.append(weights_df)

    # Create separate plots for aggressive and conservative average weights
    fig_aggressive_weights = go.Figure()
    fig_conservative_weights = go.Figure()

    if aggressive_weights:
        aggressive_average = pd.concat(aggressive_weights).groupby(level=0).mean()
        for asset in aggressive_average.columns:
            fig_aggressive_weights.add_trace(go.Scatter(
                x=aggressive_average.index,
                y=aggressive_average[asset] * 100,  # Convert to percentage
                mode='none',
                stackgroup='one',
                name=f'Aggressive Avg - {asset} (%)'
            ))

    if conservative_weights:
        conservative_average = pd.concat(conservative_weights).groupby(level=0).mean()
        for asset in conservative_average.columns:
            fig_conservative_weights.add_trace(go.Scatter(
                x=conservative_average.index,
                y=conservative_average[asset] * 100,  # Convert to percentage
                mode='none',
                stackgroup='one',
                name=f'Conservative Avg - {asset} (%)'
            ))

    # Update layouts for the average weights plots
    fig_aggressive_weights.update_layout(
        title='Average Asset Weights (Aggressive Strategies)',
        xaxis_title='Date',
        yaxis_title='Weight (%)',
        legend_title='Assets',
        template='plotly_white',
        xaxis=dict(tickangle=-45),
        yaxis=dict(range=[0, 100])  # Ensures the y-axis ranges from 0 to 100%
    )

    fig_conservative_weights.update_layout(
        title='Average Asset Weights (Conservative Strategies)',
        xaxis_title='Date',
        yaxis_title='Weight (%)',
        legend_title='Assets',
        template='plotly_white',
        xaxis=dict(tickangle=-45),
        yaxis=dict(range=[0, 100])  # Ensures the y-axis ranges from 0 to 100%
    )

    return strategy_weight_figures, fig_aggressive_weights, fig_conservative_weights

