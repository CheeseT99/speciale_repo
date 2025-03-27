import os
import numpy as np
import pandas as pd
import itertools
import time
import h5py
from scipy.special import multigammaln
from numpy.linalg import slogdet, inv
from collections import defaultdict
import statistics

toc = time.time()

### Step by step guide (from __init__.py)
# 1. Store Inputs & Basic Checks
# 2. Identify Estimation vs Test range
# 3. Extract Factor Names and Convert to NumPy
# 4. Compute the Market Sharpe Ratio
# 5. Process zz (predictors)
# 6. Optional Mean/Std Demeaning
# 7. Process rr (Test-Asset Returns)
# 8. Compute OmegaOrigEstimation

# --- Helper functions to help with numerical stability in ML computation--- 
    
def safe_logdet(matrix):
    if matrix.ndim == 0 or np.isscalar(matrix):
        return np.log(np.abs(matrix))
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise ValueError(f"Non-positive definite matrix detected! det={sign}")
    return logdet

def compute_marginal_likelihood_unrestricted(S0, Sf0, Sr, Sf, T, T0, N, K, M):
    Tstar = T0 + T

    # SKAL FINDES I PAPERET
    log_ML_unrestricted = (
        - T * (N + K) / 2 * np.log(np.pi)
        + (K * (M + 1) + N * (1 + M + K + K * M)) / 2 * np.log(T0 / Tstar)
        + multigammaln((Tstar - (K + 1) * M - 1) / 2, N)
        - multigammaln((T0 - (K + 1) * M - 1) / 2, N)
        + multigammaln((Tstar + N - M - 1) / 2, K)
        - multigammaln((T0 + N - M - 1) / 2, K)
        + (T0 - (K + 1) * M - 1) / 2 * safe_logdet(S0 / T0)
        - (Tstar - (K + 1) * M - 1) / 2 * safe_logdet(Sr / Tstar)
        + (T0 + N - M - 1) / 2 * safe_logdet(Sf0 / T0)
        - (Tstar + N - M - 1) / 2 * safe_logdet(Sf / Tstar)
    )
    return log_ML_unrestricted

def compute_marginal_likelihood_restricted(S0, Sf0, Sf, T, T0, N, K, M, F, Omega, R):
    Tstar = T0 + T

    # --- Restricted marginal likelihood (zero intercepts/no mispricing) ---
    # phi0R: coefficients for restricted model (no intercept/predictors, only factors)
    WR = np.hstack([F, Omega])  # Restricted regression matrix excluding intercept and predictors
    WRtWR = WR.T @ WR
    WRtWR_inv = np.linalg.pinv(WRtWR)
    phiTildaR = WRtWR_inv @ WR.T @ R

    # Residual covariance for restricted model
    R_minus_WR_phiTildaR = R - WR @ phiTildaR
    SrR = (S0 + (R_minus_WR_phiTildaR.T @ R_minus_WR_phiTildaR) / T)

    # SKAL FINDES I PAPERET
    log_ML_restricted = (
        - T * (N + K) / 2 * np.log(np.pi)
        + (K * (M + 1) + N * (K + K * M)) / 2 * np.log(T0 / Tstar)
        + multigammaln((Tstar - K * M) / 2, N)
        - multigammaln((T0 - K * M) / 2, N)
        + multigammaln((Tstar + N - M - 1) / 2, K)
        - multigammaln((T0 + N - M - 1) / 2, K)
        + (T0 - K * M) / 2 * safe_logdet(S0 / T0)
        - (Tstar - K * M) / 2 * safe_logdet(SrR / Tstar)
        + (T0 + N - M - 1) / 2 * safe_logdet(Sf0 / T0)
        - (Tstar + N - M - 1) / 2 * safe_logdet(Sf / Tstar)
    )
    return log_ML_restricted


def omega_matrices(KMax, MMax, ff, zz):
    # Iterate over all possible combinations of factor models and predictors
        # Open an HDF5 file to store Omega matrices
        with h5py.File('omega_matrices.h5', 'w') as h5f:

            # Iterate over all combinations
            for k in range(1, KMax + 1): # factors: 1 to KMax inclusive
                for m in range(1, MMax + 1):  # predictors: 1 to MMax inclusive
                    for factors in itertools.combinations(factor_names, k):
                        for predictors in itertools.combinations(predictor_names, m):
                            model_id = "__".join([
                                        "_".join(sorted(factors)),
                                        "_".join(sorted(predictors))
                                    ])

                            F = ff[list(factors)].values
                            Z = zz[list(predictors)].values
                            T = ff.shape[0]

                            Omega = np.zeros((T, len(factors) * len(predictors)))
                            for t in range(T):
                                Omega[t, :] = np.kron(F[t, :], Z[t, :])

                            # Save Omega in the HDF5 file
                            h5f.create_dataset(model_id, data=Omega)

def calc_regression_matrices(model_id, Omega, ff, zz, rr):
    ### Calculate Model-specific Quantities

    # Extract Factors (F) and Predictors (Z) from model name
    factors_str, predictors_str = model_id.split('__')
    factors = factors_str.split('_')
    predictors = predictors_str.split('_')

    # Get F, Z, and returns (R) data
    F = ff[factors].values       # Factors (T x k)
    Z = zz[predictors].values    # Predictors (T x m)
    R = rr.values.reshape(-1, 1) # Asset returns (T x 1), ensure it's 2D column vector

    T = len(R)

    # 1. Regression matrices (X, W):
    # X matrix (predictor matrix with intercept): (T x m+1)
    X = np.hstack([np.ones((T, 1)), Z])  # X (predictor matrix with intercept): (T, m+1)
    W = np.hstack([X, F, Omega])         # W (Full model regression matrix): (T, 1+m+k+km)

    # 2. Mean vectors:
    RMean = np.mean(R, axis=0)  # (1,)
    FMean = np.mean(F, axis=0)  # (k,)
    ZMean = np.mean(Z, axis=0)  # (m,)

    # 3. Covariance matrices:
    # Covariance matrix of factor returns (Vf)
    Vf = np.cov(F, rowvar=False, bias=True)  # shape: (k x k)

    # Compute residual covariance matrix SigmaRR:
    # First, compute beta0 using OLS (regression of R on F):
    beta0 = np.linalg.pinv(F.T @ F) @ F.T @ R  # (k x 1)

    # Projection matrix Qw:
    Qw = np.eye(T) - W @ np.linalg.pinv(W.T @ W) @ W.T  # (T x T)

    # SigmaRR (Covariance matrix of residuals):
    SigmaRR = (R.T @ Qw @ R) / T
    return R, F, X, W, RMean, FMean, ZMean, Vf, beta0, SigmaRR


def estimate_regression_coefficients(F, X, R):
    """
    --- Summary ---
    \hat{\beta}_{0} = (F^T F)^{-1} F^T R
    \hat{A}_{f0} = (X^T X)^{-1} X^T F
    --- Explaination ---
    Beta is an estimate of how returns (R) are explained by factor returns (F).
    Af0 is an estimate of how factors (F) depend on predictors (X, including intercept).
    --- Dimension ---
    Beta: (k x 1) where k is the number of factors
    Af0: ((m+1) x k) where m is the number of predictors
    """
    # ---- 1. Compute beta0: returns (R) regressed on factors (F) ----
    # beta0 dimension: (k x 1)
    beta0 = np.linalg.pinv(F.T @ F) @ F.T @ R

    # ---- 2. Compute Af0: factors (F) regressed on predictors (X) ----
    # Af0 dimension: ((m+1) x k), intercept included in X
    Af0 = np.linalg.pinv(X.T @ X) @ X.T @ F

    return beta0, Af0


def matrices_for_ML(F, R, X, W, beta0, Omega, T):
    # Construct matrices required for marginal likelihood computation
    """
    S0 (Initial residual covariance matrix)
    Sf0 (Initial covariance of factor returns)
    Sr (Residual covariance considering the full model)
    Sf (Factor covariance conditional on predictors)
    """

    # --- 1. Compute S0 ---
    FtF = F.T @ F
    RtR = R.T @ R
    S0 = (RtR - beta0.T @ FtF @ beta0) / T

    # --- 2. Compute Sf0 (Initial covariance of factors) ---
    Vf = np.cov(F, rowvar=False, bias=True)  # (k x k)
    Sf0 = Vf.copy()  # Directly assign Vf as Sf0 (initial factor covariance)

    # --- 3. Compute Sr (Residual covariance for the full model) ---
    # phi0 combines zeros for intercept and predictors, beta0 for factors, zeros for interactions
    num_predictors = X.shape[1] - 1  # subtract intercept
    num_factors = F.shape[1]
    num_interactions = Omega.shape[1]

    # phi0: shape ((1+m+k+km) x 1)
    phi0 = np.vstack([
        np.zeros((1 + num_predictors, 1)),  # intercept + predictors
        beta0,                              # factor coefficients
        np.zeros((num_interactions, 1))     # interactions
    ])

    # phiTilda: adjusted estimate for full model
    WtW_inv = np.linalg.pinv(W.T @ W)
    phiTilda = WtW_inv @ (W.T @ R)

    # Residuals considering the full model:
    ### SKAL FINDES I PAPERET
    R_minus_WphiTilda = R - W @ phiTilda
    Sr = (S0 + (R_minus_WphiTilda.T @ R_minus_WphiTilda) / T
        + (phiTilda - phi0).T @ (W.T @ W / T) @ (phiTilda - phi0))

    # --- 4. Compute Sf (Factor covariance conditional on predictors) ---
    ### SKAL FINDES I PAPERET
    XtX_inv = np.linalg.pinv(X.T @ X)
    AfTilda = XtX_inv @ (X.T @ F)

    F_minus_XAfTilda = F - X @ AfTilda
    Sf = (F_minus_XAfTilda.T @ F_minus_XAfTilda) / T
    return S0, Sf0, Sr, Sf

def compute_log_marginal_likelihoods(log_marginal_likelihoods, factor_loadings, predictor_loadings, Omega_dict, ff, zz, rr, total_iterations):

    with h5py.File('log_marginal_likelihoods.h5', 'w') as file:
        for i, (model_id, Omega) in enumerate(Omega_dict.items(), 1):
            if i % max(total_iterations // 20, 1) == 0 or i == total_iterations:
                percent_done = (i / total_iterations) * 100
                print(f"{percent_done:.0f}% done: currently processing model {model_id} ({i}/{total_iterations})")
            
            R, F, X, W, RMean, FMean, ZMean, Vf, beta0, SigmaRR = calc_regression_matrices(model_id, Omega, ff, zz, rr)
            # Comments about parameters:
            """ 
            --- beta0: Baseline factor loadings (coefficients from regression of returns solely on factors), used as prior estimates in marginal likelihood calculation ---
            --- SigmaRR: Residual covariance matrix from the regression of returns on the complete model (intercept, predictors, factors, interactions) ---            --- W: Design matrix for the full regression model, including intercept, predictors, factors, and their interactions ---
            --- X: Predictor matrix including intercept (constant term) ---
            --- RMean, FMean, ZMean: Mean vectors of asset returns, factor returns, and macroeconomic predictors respectively ---
            --- Omega: Interaction matrix representing factor-predictor interactions for the selected model ---
            """
            beta0, Af0 = estimate_regression_coefficients(F, X, R)
            S0, Sf0, Sr, Sf = matrices_for_ML(F, R, X, W, beta0, Omega, len(R))
            
            # --- Marginal likelihood computation ---
            N = R.shape[1]
            K = F.shape[1]
            M = X.shape[1] - 1  # Exclude intercept
            T0 = indexEndOfEstimation if 'indexEndOfEstimation' in locals() else len(R)
            Tstar = T0 + len(R)
            
            # Compute explicitly unrestricted and restricted:
            log_ML_unrestricted = compute_marginal_likelihood_unrestricted(S0, Sf0, Sr, Sf, T, T0, N, K, M)
            log_ML_restricted = compute_marginal_likelihood_restricted(S0, Sf0, Sf, T, T0, N, K, M, F, Omega, R)
            
            # Write to dictionary and HDF5 file
            log_marginal_likelihoods[model_id] = {
                'unrestricted': log_ML_unrestricted,
                'restricted': log_ML_restricted
            }

            # Store factor and predictor loadings
            factor_loadings[model_id] = {factor: beta0[idx] for idx, factor in enumerate(FMean)}
            predictor_loadings[model_id] = {predictor: Af0[idx] for idx, predictor in enumerate(ZMean)}
            
            model_group = file.create_group(str(model_id))
            model_group.create_dataset('unrestricted', data=log_ML_unrestricted)
            model_group.create_dataset('restricted', data=log_ML_restricted)
            model_group.create_dataset('factor_loadings', data=np.array(list(factor_loadings[model_id].values())))
            model_group.create_dataset('predictor_loadings', data=np.array(list(predictor_loadings[model_id].values())))
            model_group.create_dataset('FMean', data=FMean)
            model_group.create_dataset('ZMean', data=ZMean)
    return log_marginal_likelihoods, factor_loadings, predictor_loadings, FMean, ZMean


def posterior_predictive(
    X_new: np.ndarray,
    m_n: np.ndarray,
    V_n: np.ndarray,
    a_n: float,
    b_n: float
    ) -> tuple[float, float]:
    """
    Compute posterior predictive mean and variance for a new input X_new,
    given posterior parameters (m_n, V_n, a_n, b_n) from a normal-inverse-gamma model.

    Arguments:
      X_new  : shape (K,) new regressor row (including a constant if needed).
      m_n    : shape (K,) posterior mean of beta.
      V_n    : shape (K, K) posterior covariance of beta.
      a_n,b_n: scalar posterior parameters for sigma^2 (inverse-gamma).

    Returns:
      pred_mean, pred_var : posterior predictive mean and variance
    """
    # Posterior predictive mean:
    pred_mean = X_new @ m_n  # X_new is row vector, m_n is K-vector

    # Posterior mean of sigma^2 = b_n / (a_n - 1)
    sigma2_hat = b_n / (a_n - 1.0)

    # Posterior predictive variance includes data noise + parameter uncertainty
    pred_var = sigma2_hat * (1.0 + X_new @ V_n @ X_new)

    return pred_mean, pred_var



if __name__ == "__main__":
    # Construct data path
    parent_dir = os.path.dirname(os.getcwd())
    csv_path = os.path.join(parent_dir, 'speciale_repo', 'data', 'data_for_BMA_short.csv')
    
    # Read data
    df = pd.read_csv(csv_path)
    
    # Clean df columns
    df = df.drop(columns=['month','month:1','month:2'])

    # Define rr
    rr = df['ret_excess']

    # Define ff (just a few factors)
    ff = df[['date','mktcap', 'smb', 'hml']]

    # Define zz (13 predictors)
    zz = df[['date','dp','dy','ep','de','svar','bm','ntis','tbl','lty','ltr','tms','dfy','infl']]


    # INPUT PARAMETERS
    Tau = 1.25 # Upper bound Sharpe ratio
    indexEndOfEstimation = int(ff.shape[0] / 2) # int() takes floor value
    # Mangler significantPredictors ?

    # Calculate squared market Sharpe ratio
    market_mean = np.mean(rr)
    market_std = np.std(rr)

    SR2Mkt = (market_mean / market_std) ** 2 # squared market Sharpe ratio
    print("SR2Mkt:\n", SR2Mkt)
    print("Tau:\n", Tau)
    print("Upper bound Sharpe ratio is: ", Tau*np.sqrt(SR2Mkt))

    ### IN THIS FRAMEWORK WE HAVE:
    """
    Models with factors only: 2^3 = 8
    Models with predictors only: 2^13 = 8192
    Thus, 
    Total models: 2^(3+13) = 2^16 = 65,536

    Models with 0 factors: 2^0 * 2^13 = 8192
    Models with 0 predictors: 2^3 * 2^0 = 8
    The intersection (models with 0 factors and 0 predictors) counted twice, so add 1
    Total Omega matrices: 65,536 − 8192 − 8 + 1 = 57,337
    
    Omega dimension for a single matrix is (if all factors and predictors included): 791 x 39"""
    
    # Define T
    T = ff.shape[0]

    # Define T0 (sample size)?
    T0 = indexEndOfEstimation

    # Model enumeration
    factor_names = ff.columns[1:]
    predictor_names = zz.columns[1:]

    # Define KMax (max number of factors)
    KMax = len(factor_names)
    # Define MMax (max number of predictors)
    MMax = len(predictor_names)

    # Prepare Omega dictionary
    Omega_dict = {}

    # Omega matrices
    if not os.path.exists('omega_matrices.h5'):
        print("Omega matrices file does not exist. Creating...")
        omega_matrices(KMax, MMax, ff, zz)
    else:
        print("Omega matrices file exists. Loading...")

    with h5py.File('omega_matrices.h5', 'r') as h5f:
        for model_id in h5f.keys():
            Omega_dict[model_id] = h5f[model_id][:]
    print(f"Loaded {len(Omega_dict)} Omega matrices.")

    # Check Omega matrix dimensions
    expected_total = (2**KMax - 1)*(2**MMax - 1)
    print("Expected total Omegas:", expected_total)

    actual_total = len(Omega_dict)
    # print("Actual Omegas calculated:", actual_total)

    # Check explicitly missing keys:
    if actual_total != expected_total:
        print("Missing Omega keys detected!")

        missing_models = []
        for k in range(1, KMax + 1):
            for m in range(1, MMax + 1):
                for factors in itertools.combinations(factor_names, k):
                    for predictors in itertools.combinations(predictor_names, m):
                        model_id = "__".join([
                            "_".join(sorted(factors)),
                            "_".join(sorted(predictors))
                        ])
                        if model_id not in Omega_dict:
                            print(f"Missing model: {model_id}")

    # Prepare dictionaries to store log marginal likelihoods, factor loadings, and predictor loadings
    log_marginal_likelihoods = {}
    factor_loadings = {}
    predictor_loadings = {}

    # For simplicity
    total_iterations = len(Omega_dict)

    if not os.path.exists('log_marginal_likelihoods.h5'):
        print("Log marginal likelihoods file does not exist. Creating...")
        log_marginal_likelihoods, factor_loadings, predictor_loadings, FMean, ZMean = compute_log_marginal_likelihoods(log_marginal_likelihoods, factor_loadings, predictor_loadings, Omega_dict, ff, zz, rr, total_iterations)  
    else:
        print("Log marginal likelihoods file exists. Loading...")
    with h5py.File('log_marginal_likelihoods.h5', 'r') as file:
        for model_id in file.keys():
            log_marginal_likelihoods[model_id] = {
                'unrestricted': file[model_id]['unrestricted'][()],
                'restricted': file[model_id]['restricted'][()]
            }
            factor_loadings[model_id] = file[model_id]['factor_loadings'][()]
            predictor_loadings[model_id] = file[model_id]['predictor_loadings'][()]
            FMean = file[model_id]['FMean'][()]
            ZMean = file[model_id]['ZMean'][()]
        print(f"Log marginal likelihoods are loaded.")


    # --- Posterior Model Probabilities and BMA ---
    # Step 1: Compare and rank models:
    sorted_models = sorted(log_marginal_likelihoods.items(), 
                           key=lambda x: x[1]['unrestricted'], 
                           reverse=True
                           )
    
    # Extract model IDs and unrestricted log marginal likelihoods explicitly
    model_ids = [model for model, _ in sorted_models]
    log_ml_values = np.array([ml_values['unrestricted'] for _, ml_values in sorted_models])

    # Step 2: Convert log marginal likelihoods explicitly to posterior model probabilities
    max_log_ml = np.max(log_ml_values)
    relative_ml = np.exp(log_ml_values - max_log_ml)
    posterior_probs = relative_ml / np.sum(relative_ml)
    posterior_model_probs = dict(zip(model_ids, posterior_probs))

    # Print top 10 models as df:
    df_top_models = pd.DataFrame(list(posterior_model_probs.items())[:10], columns=['Model', 'Posterior Probability'])
    print("\nTop 10 Models by Posterior Probability (Unrestricted):")
    print(df_top_models.to_string(index=False, formatters={'Posterior Probability': '{:.4f}'.format}))
    
    print("Sum of probability of top 10 models:\n", sum(df_top_models['Posterior Probability']))


    # Step 3: Bayesian Model Averaging (BMA) - compute inclusion probabilities explicitly
    # Define default dictionaries for factor and predictor probabilities
    factor_probs = defaultdict(float)
    predictor_probs = defaultdict(float)
        
    # Calculate inclusion probabilities explicitly for each factor and predictor
    # Aka: accummulate the posterior probabilities for each factor and predictor
    for model_id, prob in posterior_model_probs.items():
        factor_str, predictor_str = model_id.split('__')
        factors = factor_str.split('_')
        predictors = predictor_str.split('_')

        for factor in factors:
            factor_probs[factor] += prob

        for predictor in predictors:
            predictor_probs[predictor] += prob

    # Display BMA inclusion probabilities
    print("\nFactor Inclusion Probabilities (BMA):")
    for factor, prob in sorted(factor_probs.items(), key=lambda x: -x[1]):
        print(f"{factor}: {prob:.4f}")
    print("\nPredictor Inclusion Probabilities (BMA):")
    for predictor, prob in sorted(predictor_probs.items(), key=lambda x: -x[1]):
        print(f"{predictor}: {prob:.4f}")

    # --- COMPUTE FACTOR EQUATION (f_{t+1}) AND RETURN EQUATION (r_{t+1}) ---
    
    # Define dictionary to store model data
    model_data = {}

    # Define iotaT
    iotaT = np.ones((T, 1)) # T x 1 vector of ones

    # Initiate for loop to compute factor and return equations for each model
    for model_id, prob in posterior_model_probs.items():
        factor_str, predictor_str = model_id.split('__')
        factors = factor_str.split('_')
        predictors = predictor_str.split('_')

        # Assign model-specific data to dictionary
        model_data[model_id] = {
            'factors': factors,
            'predictors': predictors,
            'prob': prob,
            'factor_loadings': factor_loadings[model_id],
            'predictor_loadings': predictor_loadings[model_id]
        }

        # Step 2: Construct the design matrix for the factor equation and return equation
        S0 =




    breakpoint = 1
    exit() # HERFRA:
    # # Compute posterior-weighted averages of factor and predictor loadings
    # avg_factor_loadings, avg_predictor_loadings = compute_posterior_weighted_loadings(posterior_model_probs, factor_loadings, predictor_loadings)
    # avg_factor_loadings = {}
    # for (key, value) in factor_loadings.items():
    #     avg_factor_loadings[key] = [statistics.mean(posterior_model_probs.values()) * value for value in factor_loadings.values()]

    # -------------------------------------------------------------------
    # 1) Determine maximum lengths across ALL models for factors/predictors
    # -------------------------------------------------------------------
    max_len_factors = 0
    max_len_predictors = 0
    for model_id in posterior_model_probs.keys():
        # Only consider if the model actually exists in the dictionary
        if model_id in factor_loadings:
            max_len_factors = max(max_len_factors, len(factor_loadings[model_id]))
        if model_id in predictor_loadings:
            max_len_predictors = max(max_len_predictors, len(predictor_loadings[model_id]))

    # -------------------------------------------------------------------
    # 2) Initialize accumulators:
    #    - sum_arrays store the sum(prob * array[i]) for each index i
    #    - weight_arrays store the sum(prob) for each index i
    # -------------------------------------------------------------------
    factor_sum = np.zeros(max_len_factors, dtype=float)
    factor_weight = np.zeros(max_len_factors, dtype=float)

    predictor_sum = np.zeros(max_len_predictors, dtype=float)
    predictor_weight = np.zeros(max_len_predictors, dtype=float)

    # -------------------------------------------------------------------
    # 3) Accumulate posterior-weighted values 
    # -------------------------------------------------------------------
    for model_id, prob in posterior_model_probs.items():
        # Factor array 
        if model_id in factor_loadings:
            arr = factor_loadings[model_id]
            for i, val in enumerate(arr):
                factor_sum[i] += prob * val
                factor_weight[i] += prob

        # Predictor array
        if model_id in predictor_loadings:
            arr = predictor_loadings[model_id]
            for i, val in enumerate(arr):
                predictor_sum[i] += prob * val
                predictor_weight[i] += prob

    # -------------------------------------------------------------------
    # 4) Compute final average by dividing sum by weight for each index
    # -------------------------------------------------------------------
    avg_factor_array = np.zeros(max_len_factors, dtype=float)
    for i in range(max_len_factors):
        if factor_weight[i] > 0:
            avg_factor_array[i] = factor_sum[i] / factor_weight[i]
        else:
            avg_factor_array[i] = 0.0  # or remain 0 if no model had that index

    avg_predictor_array = np.zeros(max_len_predictors, dtype=float)
    for i in range(max_len_predictors):
        if predictor_weight[i] > 0:
            avg_predictor_array[i] = predictor_sum[i] / predictor_weight[i]
        else:
            avg_predictor_array[i] = 0.0


    print("\nFactor Weighted Average (BMA):", avg_factor_loadings)
    print("\nPredictor Weighted Average (BMA):", avg_predictor_loadings)

    # for factor, average in sorted(avg_factor_loadings.items(), key=lambda x: -x[1]):
    #     print(f"{factor}: {average:.4f}")

    # print("\nPredictor Weighted Average (BMA):")
    # for predictor, average in sorted(avg_predictor_loadings.items(), key=lambda x: -x[1]):
    #     print(f"{predictor}: {average:.4f}")

    # Step 4: Diagnostic metrics explicitly for T0 (assuming T0 fixed at T across all models)
    # If T0 varies explicitly by model, replace the following line with actual T0 values per model
    T0_values = [T0 for _ in model_ids]  # Adjust explicitly if T0 varies by model

    print("\nDiagnostic Metrics for T0 Across All Models:")
    print(f"Average T0: {np.mean(T0_values):.2f}")
    print(f"Minimum T0: {np.min(T0_values):.2f}")
    print(f"Maximum T0: {np.max(T0_values):.2f}")



    tic = time.time()
    print(f"Elapsed time: {(tic-toc):.2f}")
    breakpoint=1