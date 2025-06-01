import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, invgamma
import pandas as pd

# ------------------------------
# Simulate toy data for Model 1 and Model 2
# ------------------------------

np.random.seed(42)
n = 30
z = np.random.normal(0, 1, n)  # Macro predictor
f = np.random.normal(0, 1, n)  # Factor
alpha_true = 1.0
beta_true = 0.8
alpha1_true = 0.5
beta1_true = -0.3
sigma2_true = 1.0

# Model 1 (constant alpha and beta)
r1 = alpha_true + beta_true * f + np.random.normal(0, np.sqrt(sigma2_true), n)

# Model 2 (time-varying alpha and beta)
r2 = (alpha_true + alpha1_true * z) + (beta_true + beta1_true * z) * f + np.random.normal(0, np.sqrt(sigma2_true), n)

# ------------------------------
# Bayesian Setup for Model 1: r_t = alpha + beta * f_t + eps_t
# ------------------------------

X1 = np.column_stack((np.ones(n), f))
y1 = r1

# Priors: Normal-Inverse-Gamma
mu0 = np.zeros(2)
Lambda0 = np.eye(2) * 0.1  # prior precision (tau^{-2})
a0 = 2.0  # shape
b0 = 1.0  # scale

# Posterior parameters (conjugate updating)
XtX = X1.T @ X1
Xty = X1.T @ y1
Lambda_n = XtX + Lambda0
mu_n = np.linalg.solve(Lambda_n, Xty + Lambda0 @ mu0)
a_n = a0 + n / 2
b_n = b0 + 0.5 * (y1.T @ y1 + mu0.T @ Lambda0 @ mu0 - mu_n.T @ Lambda_n @ mu_n)

# Predictive distribution components
posterior_variance = invgamma.var(a_n, scale=b_n)

# Prepare results
df_results_model1 = pd.DataFrame({
    "Posterior alpha mean": [mu_n[0]],
    "Posterior beta mean": [mu_n[1]],
    "Posterior sigma^2 mean": [b_n / (a_n - 1)],
    "Posterior sigma^2 variance": [posterior_variance]
})

# ------------------------------
# Bayesian Setup for Model 2: r_t = (alpha + alpha1 * z_t) + (beta + beta1 * z_t) * f_t + eps_t
# ------------------------------

# Construct regressors for Model 2
X2 = np.column_stack((np.ones(n), z, f, z * f))  # Columns: 1, z, f, z*f
y2 = r2

# Priors: Normal-Inverse-Gamma (4 parameters now)
mu0_2 = np.zeros(4)
Lambda0_2 = np.eye(4) * 0.1  # prior precision
a0_2 = 2.0
b0_2 = 1.0

# Posterior updates
XtX_2 = X2.T @ X2
Xty_2 = X2.T @ y2
Lambda_n2 = XtX_2 + Lambda0_2
mu_n2 = np.linalg.solve(Lambda_n2, Xty_2 + Lambda0_2 @ mu0_2)
a_n2 = a0_2 + n / 2
b_n2 = b0_2 + 0.5 * (y2.T @ y2 + mu0_2.T @ Lambda0_2 @ mu0_2 - mu_n2.T @ Lambda_n2 @ mu_n2)

# Predictive distribution components
posterior_variance_2 = invgamma.var(a_n2, scale=b_n2)

# Prepare results
df_results_model2 = pd.DataFrame({
    "Posterior alpha mean": [mu_n2[0]],
    "Posterior alpha1 mean": [mu_n2[1]],
    "Posterior beta mean": [mu_n2[2]],
    "Posterior beta1 mean": [mu_n2[3]],
    "Posterior sigma^2 mean": [b_n2 / (a_n2 - 1)],
    "Posterior sigma^2 variance": [posterior_variance_2]
})

print("Model 1 Results:")
print(df_results_model1)
print("\nModel 2 Results:")
print(df_results_model2)

# Plot prior, likelihood (approximate), and posterior for both models

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
alpha_range = np.linspace(-2, 3, 300)

# Model 1: Plot posterior of alpha and beta
axes[0, 0].plot(alpha_range, norm.pdf(alpha_range, loc=mu0[0], scale=np.sqrt(1 / Lambda0[0, 0])), label="Prior")
axes[0, 0].plot(alpha_range, norm.pdf(alpha_range, loc=mu_n[0], scale=np.sqrt(1 / Lambda_n[0, 0])), label="Posterior")
axes[0, 0].set_title("Model 1: Alpha Posterior")
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(alpha_range, norm.pdf(alpha_range, loc=mu0[1], scale=np.sqrt(1 / Lambda0[1, 1])), label="Prior")
axes[0, 1].plot(alpha_range, norm.pdf(alpha_range, loc=mu_n[1], scale=np.sqrt(1 / Lambda_n[1, 1])), label="Posterior")
axes[0, 1].set_title("Model 1: Beta Posterior")
axes[0, 1].legend()
axes[0, 1].grid(True)

# Model 2: Plot posterior of alpha1 and beta1
axes[1, 0].plot(alpha_range, norm.pdf(alpha_range, loc=mu0_2[1], scale=np.sqrt(1 / Lambda0_2[1, 1])), label="Prior")
axes[1, 0].plot(alpha_range, norm.pdf(alpha_range, loc=mu_n2[1], scale=np.sqrt(1 / Lambda_n2[1, 1])), label="Posterior")
axes[1, 0].set_title("Model 2: Alpha1 Posterior")
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(alpha_range, norm.pdf(alpha_range, loc=mu0_2[3], scale=np.sqrt(1 / Lambda0_2[3, 3])), label="Prior")
axes[1, 1].plot(alpha_range, norm.pdf(alpha_range, loc=mu_n2[3], scale=np.sqrt(1 / Lambda_n2[3, 3])), label="Posterior")
axes[1, 1].set_title("Model 2: Beta1 Posterior")
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

from scipy.special import gammaln
from numpy.linalg import slogdet

def compute_log_marginal_likelihood(n, k, a0, b0, a_n, b_n, Lambda0, Lambda_n):
    """Compute log marginal likelihood for Normal-Inverse-Gamma prior"""
    sign0, logdet0 = slogdet(Lambda0)
    signn, logdetn = slogdet(Lambda_n)
    
    term1 = 0.5 * (logdet0 - logdetn)
    term2 = a0 * np.log(b0) - a_n * np.log(b_n)
    term3 = gammaln(a_n) - gammaln(a0)
    term4 = -0.5 * n * np.log(2 * np.pi)
    
    log_ml = term1 + term2 + term3 + term4
    return log_ml

# Compute log marginal likelihoods
k1 = X1.shape[1]
k2 = X2.shape[1]

log_ml_1 = compute_log_marginal_likelihood(n, k1, a0, b0, a_n, b_n, Lambda0, Lambda_n)
log_ml_2 = compute_log_marginal_likelihood(n, k2, a0_2, b0_2, a_n2, b_n2, Lambda0_2, Lambda_n2)

# Display results
df_marginal_likelihoods = pd.DataFrame({
    "Model": ["Model 1", "Model 2"],
    "Log Marginal Likelihood": [log_ml_1, log_ml_2]
})

print(df_marginal_likelihoods)

# Assume uniform model priors
P_M1_prior = 0.5
P_M2_prior = 0.5

# Convert log marginal likelihoods to raw (unnormalized) posterior weights
ml1 = np.exp(log_ml_1)
ml2 = np.exp(log_ml_2)

# Unnormalized posterior probabilities
unnormalized_p1 = ml1 * P_M1_prior
unnormalized_p2 = ml2 * P_M2_prior

# Normalize to get posterior model probabilities
total = unnormalized_p1 + unnormalized_p2
posterior_prob_M1 = unnormalized_p1 / total
posterior_prob_M2 = unnormalized_p2 / total

# Display results
df_posterior_model_probs = pd.DataFrame({
    "Model": ["Model 1", "Model 2"],
    "Posterior Model Probability": [posterior_prob_M1, posterior_prob_M2]
})
print("\nPosterior Model Probabilities:")
print(df_posterior_model_probs)

# Merge all relevant results into one comprehensive table

df_combined = pd.DataFrame({
    "Model": ["Model 1", "Model 2"],
    "Posterior alpha mean": [mu_n[0], mu_n2[0]],
    "Posterior alpha1 mean": [np.nan, mu_n2[1]],
    "Posterior beta mean": [mu_n[1], mu_n2[2]],
    "Posterior beta1 mean": [np.nan, mu_n2[3]],
    "Posterior sigma^2 mean": [b_n / (a_n - 1), b_n2 / (a_n2 - 1)],
    "Posterior sigma^2 variance": [invgamma.var(a_n, scale=b_n), invgamma.var(a_n2, scale=b_n2)],
    "Log Marginal Likelihood": [log_ml_1, log_ml_2],
    "Posterior Model Probability": [posterior_prob_M1, posterior_prob_M2]
})

print("\nCombined Results:")
print(df_combined)

# Split into Panel A (posterior parameters) and Panel B (model comparison stats)

panel_a = df_combined[[
    "Model",
    "Posterior alpha mean",
    "Posterior alpha1 mean",
    "Posterior beta mean",
    "Posterior beta1 mean",
    "Posterior sigma^2 mean",
    "Posterior sigma^2 variance"
]]

panel_b = df_combined[[
    "Model",
    "Log Marginal Likelihood",
    "Posterior Model Probability"
]]

# Rename for clarity in table headers
panel_a.columns = pd.MultiIndex.from_product([["Panel A: Posterior Parameter Estimates"], panel_a.columns])
panel_b.columns = pd.MultiIndex.from_product([["Panel B: Model Comparison Statistics"], panel_b.columns])

# Combine horizontally
panel_full = pd.concat([panel_a, panel_b], axis=1)

print("\nFinal Combined Panel:")
print(panel_full)

# Create Panel C: Inputs used in the Bayesian models
panel_c = pd.DataFrame({
    "Model": ["Model 1", "Model 2"],
    "n (observations)": [n, n],
    "k (parameters)": [X1.shape[1], X2.shape[1]],
    "a0 (prior shape)": [a0, a0_2],
    "b0 (prior scale)": [b0, b0_2],
    "a_n (posterior shape)": [a_n, a_n2],
    "b_n (posterior scale)": [b_n, b_n2]
})

# Assign MultiIndex headers
panel_c.columns = pd.MultiIndex.from_product([["Panel C: Model Inputs"], panel_c.columns])

# Reuse panel_a and panel_b from earlier step
panel_a.columns = pd.MultiIndex.from_product([["Panel A: Posterior Parameter Estimates"], panel_a.columns.get_level_values(1)])
panel_b.columns = pd.MultiIndex.from_product([["Panel B: Model Comparison Statistics"], panel_b.columns.get_level_values(1)])

# Concatenate all panels
panel_full_all = pd.concat([panel_a, panel_b, panel_c], axis=1)

# Print Panel A, Panel B, and Panel C as three separate plain dataframes

panel_a_df = panel_a.copy()
panel_b_df = panel_b.copy()
panel_c_df = panel_c.copy()

# Reset column names to flat for printing
panel_a_df.columns = panel_a_df.columns.get_level_values(1)
panel_b_df.columns = panel_b_df.columns.get_level_values(1)
panel_c_df.columns = panel_c_df.columns.get_level_values(1)

print(panel_a_df, "\n", panel_b_df, "\n", panel_c_df)
