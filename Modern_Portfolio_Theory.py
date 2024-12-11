#%%
"""
This Python script leverages concepts from Modern Portfolio Theory (MPT) 
to simulate and visualize portfolio performance, generate the Efficient Frontier,
and identify the portfolio with the maximum Sharpe Ratio. 
The code incorporates realistic portfolio characteristics using jump diffusion 
processes for asset price simulation and covariance matrix computation.
"""
import numpy as np
from Jump_Diffusion import jump_diffusion
from Jump_Diffusion import plot_jump_diffusion_simulation
import plotly.graph_objects as go
def find_pareto_frontier(results):

    # Sample data
    x = results[1]   # Volatility
    y = results[0]   # Return

    # Combine and sort data by x (ascending)
    data = np.array(list(zip(x, y)))
    data = data[np.argsort(data[:, 0])]  # Sort by x ascending

    # Find Pareto frontier
    pareto_frontier = []
    max_y = -np.inf  # Initialize to a very small value

    for point in data:
        if point[1] > max_y:  # Check if current point is Pareto-optimal
            pareto_frontier.append(point)
            max_y = point[1]

    pareto_frontier = np.array(pareto_frontier)
    return pareto_frontier

def plot_results(results, max_sharpe_idx, risk_free_rate, show_risk_free=True):
    
    pareto_frontier = find_pareto_frontier(results)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results[1],   # Volatility
        y=results[0],   # Return
        mode='markers',
        marker=dict(
            size=5, 
            color=results[2], 
            colorscale='Viridis', 
            # colorbar=dict(title='Sharpe Ratio'),
        ),
        showlegend=False,
    ))
    if show_risk_free:
        fig.add_trace(go.Scatter(
            x=[0], 
            y=[risk_free_rate], 
            mode='markers', 
            marker=dict(
                size=10, 
                color='red', 
                symbol='x', 
                line=dict(width=2)
                ),
            name='Risk-Free Rate',
            ))
        # connect risk-free rate to the point of max Sharpe Ratio
        fig.add_trace(go.Scatter(
            x=[0, results[1, max_sharpe_idx]], 
            y=[risk_free_rate, results[0, max_sharpe_idx]], 
            mode='lines', 
            line=dict(color='blue', width=1, dash='dash'),
            name='Capital Market Line',
            # showlegend=,
            ))
    
    # Efficient Frontier
    fig.add_trace(go.Scatter(
        x=pareto_frontier[:, 0], 
        y=pareto_frontier[:, 1], 
        mode='lines+markers', 
        line=dict(color='red', width=2),
        marker=dict(size=7, color='red'),
        name='Efficient Frontier',
        ))
    
    # Max Sharpe Ratio Portfolio
    fig.add_trace(go.Scatter(
        x=[results[1, max_sharpe_idx]], 
        y=[results[0, max_sharpe_idx]], 
        mode='markers', 
        marker=dict(
            size=10, 
            color='blue', 
            symbol='x', 
            line=dict(width=2)
            ),
        name='Max Sharpe Ratio',
        ))
    
    fig.update_layout(
        # title='Efficient Frontier', 
        xaxis_title='Volatility', 
        yaxis_title='Return', 
        # legend=dict(
        #     yanchor="top", 
        #     y=1.15, 
        #     xanchor="left", 
        #     x=1,
        #     orientation='v'),
        coloraxis_showscale=False,
        )
    # hide the colorbar
    # fig.update_layout(coloraxis_showscale=False)
    fig.show()

def simulate_portfolio_performance(mean_returns, cov_matrix, num_portfolios=100, risk_free_rate=0.02):
    
    num_assets = len(mean_returns)
    # Generate random portfolios
    results = np.zeros((3, num_portfolios))  # Store return, volatility, Sharpe Ratio
    weights_record = []
    
    # find the analytical solution for the portfolio with max Sharpe Ratio [needs more work]
    # inv_cov_matrix = np.linalg.inv(cov_matrix)
    # ones = np.ones(num_assets)
    # A = np.dot(ones, np.dot(inv_cov_matrix, mean_returns))
    # B = np.dot(mean_returns, np.dot(inv_cov_matrix, mean_returns))
    # C = np.dot(ones, np.dot(inv_cov_matrix, ones))
    # D = B * C - A**2
    # max_sharpe_weights = (1/D) * np.dot(inv_cov_matrix, mean_returns - risk_free_rate * ones)
    # max_sharpe_weights = max_sharpe_weights / np.sum(max_sharpe_weights)
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)  # Normalize to 1
        weights_record.append(weights)
        
        # Calculate portfolio return and risk
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = sharpe_ratio
    
    # Locate portfolio with max Sharpe Ratio
    max_sharpe_idx = np.argmax(results[2])
    max_sharpe_weights = weights_record[max_sharpe_idx]
    
    
    # Plot results
    plot_results(results, max_sharpe_idx, risk_free_rate, show_risk_free=False)
    
    # print("Max Sharpe Ratio Portfolio Weights:")
    # for i, weight in enumerate(max_sharpe_weights):
    #     print(f"Asset {i + 1}: {weight:.2%}")
    # print(f"Expected Return: {results[0, max_sharpe_idx]:.2%}")
    # print(f"Volatility: {results[1, max_sharpe_idx]:.2%}")
    # print(f"Sharpe Ratio: {results[2, max_sharpe_idx]:.2f}")

# Call the function
num_assets = 10
num_portfolios = 10000
risk_free_rate = 0.02

random_returns = False
seed = 21

np.random.seed(seed) if seed else None

S0 = np.random.uniform(100, 200, num_assets)            # Initial price
mu = np.random.uniform(0.05, 0.2, num_assets)           # Annual drift (5%-20%)
sigma = np.random.uniform(0.1, 0.3, num_assets)         # Annual volatility (10%-30%)
lambda_ = np.random.uniform(0.5, 1, num_assets)         # Jump frequency (10%-50%)
jump_mean = np.random.uniform(-0.1, 0.1, num_assets)    # Jump size mean (-10%-10%)
jump_std = np.random.uniform(0.1, 0.2, num_assets)      # Jump size std (10%-30%)
T = 1              # Time horizon in years
dt = 1/252         # Daily time steps



# Simulated returns and covariance for demonstration
if random_returns:
    mean_returns = np.random.uniform(0.05, 0.2, num_assets)  # Expected annual returns
    cov_matrix = np.random.uniform(0.01, 0.05, (num_assets, num_assets))
    cov_matrix = (cov_matrix + cov_matrix.T) / 2  # Symmetric covariance matrix
else:
    prices = [jump_diffusion(S0[i], mu[i], sigma[i], lambda_[i], jump_mean[i], jump_std[i], T, dt, 1)[1] for i in range(num_assets)]
    prices = np.array(prices).reshape(num_assets, -1)
    plot_jump_diffusion_simulation(num_assets, prices)

    # annual returns
    mean_returns = (prices[:, -1] - S0) / S0
    # covariance matrix
    cov_matrix = np.cov(mean_returns)

simulate_portfolio_performance(
    mean_returns,
    cov_matrix,
    num_portfolios, 
    risk_free_rate,
    )
