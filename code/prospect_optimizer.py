import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, levene, f_oneway

def prospect_value(weights, r_s, r_hat, lambda_, gamma=0.1, strategy="conservative"):
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

    # Get last and second last returns from portfolio performance
    last_return = portfolio_returns[-1]
    
    second_last_return = portfolio_returns[-2]
    

    # Calculate zt based on these returns
    zt = calculate_zt(np.mean(portfolio_returns), last_return)

    # Dynamically adjust lambda based on strategy and portfolio performance
    if strategy == "conservative":
        lambda_dynamic = calculate_conservative_lambda(last_return, second_last_return, zt, lambda_)
    elif strategy == "aggressive":
        lambda_dynamic = calculate_aggressive_lambda(last_return, second_last_return, zt, lambda_)
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

def optimize_portfolio(r_s, r_hat, lambda_, strategy="conservative", gamma=0.1):
    num_assets = len(r_s[0])

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

    # Now use COBYLA
    result = minimize(prospect_value, initial_weights, 
                      args=(r_s, r_hat, lambda_, gamma, strategy),
                      method='COBYLA', constraints=constraints,
                      options={'maxiter': 1000, 'tol': 1e-6})

    return result


# Adjusted backtest function to ensure each rebalancing period is restricted to one month only



def backtest_portfolio_adjusted(returns, lookback_period='5', rebalancing_freq='1M', strategy="conservative", lambda_=1, gamma=0.9):

    """
    Perform backtesting on a portfolio with precise period restriction for specified strategy, lookback period, and rebalancing frequency.
    """
    # Ensure that the index is in datetime format
    returns.index = pd.to_datetime(returns.index)
    returns = returns.loc[:'2023-12-31']
    # Set up result containers
    portfolio_returns = []
    compounded_returns = []
    portfolio_weights = []
    individual_asset_returns = {col: [] for col in returns.columns}  # Store each asset's return per period
    r_hat_values = []  # Store r_hat values for each period
    # Convert lookback_period and rebalancing_freq to Pandas offsets
    lookback_offset = pd.DateOffset(months=int(lookback_period[:-1]) * (12 if lookback_period[-1] == 'Y' else 1))
    rebalancing_offset = pd.DateOffset(months=int(rebalancing_freq[:-1]) * (12 if rebalancing_freq[-1] == 'Y' else 1))
    # Initialize compound return
    cumulative_return = 1
    # Start backtesting over the range of dates
    current_date = returns.index[0] + lookback_offset
    end_date = returns.index[-1]
    while current_date <= end_date:

        # Define the lookback window

        lookback_start = current_date - lookback_offset

        lookback_data = returns.loc[lookback_start:current_date]
        # Find the last available SPY return before `current_date`
        previous_date = returns.index[returns.index < current_date].max()
        #print(returns)
        r_hat = returns.loc[previous_date, 'SPY']#returns.loc[previous_date, '10Y_Bond_Return'] #np.mean(portfolio_returns)returns.loc[previous_date, 'SPY']
        r_hat_values.append(r_hat)  # Store the current r_hat for the period
        # Convert lookback data to NumPy array

        r_s = np.array(lookback_data)
        # Call the optimization function
        result = optimize_portfolio(r_s, r_hat, lambda_, strategy, gamma)
        weights = [round(weight, 4) for weight in result.x]
        # Calculate the portfolio return for the exact next rebalancing period only

        rebalancing_end_date = (current_date + rebalancing_offset) - pd.Timedelta(days=1)

        next_period_data = returns.loc[current_date:rebalancing_end_date]



        # Calculate portfolio return if there is data within the period

        if not next_period_data.empty:

            portfolio_return = np.dot(next_period_data.mean(), weights)  # weighted average of returns

            portfolio_returns.append(portfolio_return)

            cumulative_return *= (1 + portfolio_return)

            compounded_returns.append(cumulative_return)

        else:

            portfolio_returns.append(0)

            compounded_returns.append(cumulative_return)



        # Save portfolio weights and each asset's actual mean return for the period

        portfolio_weights.append(weights)

        for col in returns.columns:

            individual_asset_returns[col].append(next_period_data[col].mean() if not next_period_data.empty else 0)

        

        # Move to next rebalancing period

        current_date += rebalancing_offset



    # Create a DataFrame with results

    result_data = {

        'Portfolio Returns': portfolio_returns,

        'Compounded Returns': compounded_returns,

        'Portfolio Weights': portfolio_weights,

        'r_hat': r_hat_values

    }

    result_data.update(individual_asset_returns)  # Add individual asset returns to result data



    # Align results DataFrame with the rebalancing dates

    rebalancing_dates = returns.loc[returns.index[0] + lookback_offset:].resample(rebalancing_freq).first().index

    result_df = pd.DataFrame(result_data, index=rebalancing_dates[:len(portfolio_returns)])

    

    return result_df





def resultgenerator(lambda_values, gamma_values, returns):
    aggressive_result_list = []
    conservative_result_list = []
    for i, lambdas in enumerate(lambda_values):
        for j, gammas in enumerate(gamma_values):
            # Create a unique name using lambda and gamma values
            result_name_aggressive = f"aggressive_result_lambda{lambdas}_gamma{gammas}"
            result_name_conservative = f"conservative_result_lambda{lambdas}_gamma{gammas}"
            
            # Run the backtest with the specified lambda and gamma values
            aggressive_results = backtest_portfolio_adjusted(
                returns, lookback_period='2Y', 
                rebalancing_freq='1M', 
                strategy="aggressive", 
                lambda_=lambdas, 
                gamma=gammas
            )
            conservative_results = backtest_portfolio_adjusted(
                returns, lookback_period='2Y', 
                rebalancing_freq='1M', 
                strategy="conservative", 
                lambda_=lambdas, 
                gamma=gammas
            )
            
            # Append the results with a unique name
            aggressive_result_list.append({result_name_aggressive: aggressive_results})
            conservative_result_list.append({result_name_conservative: conservative_results})
    
    return aggressive_result_list, conservative_result_list




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

