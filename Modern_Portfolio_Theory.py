#%%
"""
This Python script leverages concepts from Modern Portfolio Theory (MPT) 
to simulate and visualize portfolio performance, generate the Efficient Frontier,
and identify the portfolio with the maximum Sharpe Ratio. 
The code incorporates realistic portfolio characteristics using jump diffusion 
processes for asset price simulation and covariance matrix computation.
"""
import streamlit as st
import numpy as np
from plotly.subplots import make_subplots
from Stochastic_Processes import StochasticProcesses 
# from Stochastic_Processes import plot_paths
import plotly.graph_objects as go
def find_efficient_frontier(x, y, maximize_x=True, maximize_y=True):
    """
    Calculate the Pareto frontier for two objectives.

    Args:
        x (array-like): Values for the x-axis objective.
        y (array-like): Values for the y-axis objective.
        maximize_x (bool): Whether to maximize the x-axis objective.
        maximize_y (bool): Whether to maximize the y-axis objective.

    Returns:
        pareto_frontier (ndarray): Array of Pareto-optimal points.
    """
    # Combine and sort data based on x-axis direction
    data = np.array(list(zip(x, y)))
    sort_order = -1 if maximize_x else 1
    data = data[np.argsort(sort_order * data[:, 0])]  # Sort based on x-axis objective

    # Find Pareto frontier
    pareto_frontier = []
    extreme_y = -np.inf if maximize_y else np.inf  # Initialize based on y-axis direction

    for point in data:
        if (maximize_y and point[1] > extreme_y) or (not maximize_y and point[1] < extreme_y):
            pareto_frontier.append(point)
            extreme_y = point[1]

    return np.array(pareto_frontier)

def plot_results(results, risk_free_rate = 0.02, show_cml=True):
    
    max_sharpe_idx = np.argmax(results[2])
    pareto_frontier = find_efficient_frontier(results[1], results[0], maximize_x=False, maximize_y=True)

    fig = make_subplots(
        rows=1, cols=2, 
        # subplot_titles=('Return vs Volatility', 'Sharpe Ratio vs VaR'),
        )
    
    fig.add_trace(go.Scatter(
        x=results[1],   # Volatility
        y=results[0],   # Return
        mode='markers',
        marker=dict(
            size=5, 
            color=results[2], 
            colorscale='Viridis', 
            colorbar=dict(
                title='Sharpe Ratio',
                # position is set to be at the bottom of the plot
                xanchor='left',
                x=0.05,
                yanchor='bottom',
                y = 0.98,
                len=0.3,
                orientation='h',
                thickness=15,
                ),
        ),
        showlegend=False,
    ), row=1, col=1)
    if show_cml:
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
            ), row=1, col=1)
        # connect risk-free rate to the point of max Sharpe Ratio
        fig.add_trace(go.Scatter(
            x=[0, results[1, max_sharpe_idx]], 
            y=[risk_free_rate, results[0, max_sharpe_idx]], 
            mode='lines', 
            line=dict(color='blue', width=1, dash='dash'),
            name='Capital Market Line',
            # showlegend=,
            ), row=1, col=1)
    
    # Efficient Frontier
    fig.add_trace(go.Scatter(
        x=pareto_frontier[:, 0], 
        y=pareto_frontier[:, 1], 
        mode='lines+markers', 
        line=dict(color='red', width=2),
        marker=dict(size=7, color='red', opacity=0.5),
        name='Efficient Frontier',
        ), row=1, col=1)
    
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
        ), row=1, col=1)
    
    # Plot Sharpe Ratio vs VaR
    pareto_frontier = find_efficient_frontier(results[2], results[3], maximize_x=True, maximize_y=True)

    fig.add_trace(go.Scatter(
        x=results[2],   # Sharpe Ratio
        y=results[3],   # VaR
        mode='markers',
        marker=dict(
            size=5,
            color=results[0],
            colorscale='Viridis',
            colorbar=dict(
                title='Return',
                # position is set to be at the bottom of the plot
                xanchor='left',
                x=0.6,
                yanchor='bottom',
                y = 0.98,
                len=0.3,
                orientation='h',
                thickness=15,
                ),
        ),
        showlegend=False,
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=pareto_frontier[:, 0], 
        y=pareto_frontier[:, 1], 
        mode='lines+markers', 
        line=dict(color='red', width=2),
        marker=dict(size=7, color='red', opacity=0.5),
        name='Efficient Frontier',
        showlegend=False,
        ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=[results[2, max_sharpe_idx]], 
        y=[results[3, max_sharpe_idx]], 
        mode='markers', 
        marker=dict(
            size=10, 
            color='blue', 
            symbol='x', 
            line=dict(width=2)
            ),
        showlegend=False,
        name='Max Sharpe Ratio',
        ), row=1, col=2)
    
    fig.update_layout(
        # title='Portfolio Optimization', 
        xaxis_title='Volatility', 
        yaxis_title='Return', 
        xaxis2_title='Sharpe Ratio',
        yaxis2_title=f'VaR (%) at {100-VaR_percentile}% confidence level',
        legend=dict(
            yanchor="top", 
            y=1.15, 
            xanchor="left", 
            x=0.4,
            orientation='h'),
        
        )
    fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
    # save html file
    # fig.write_html('portfolio_optimization.html', auto_open=True)

    # histograms
    fig_hist = make_subplots(
        rows=2, cols=2,)
    hist_data = [
        (results[0], 'Return', 'blue', (1,1)),
        (results[1], 'Volatility', 'red', (1,2)),
        (results[2], 'Sharpe Ratio', 'green', (2,1)),
        (results[3], 'VaR', 'orange', (2,2)),
    ]

    for _, (data, name, color, pos) in enumerate(hist_data):
        fig_hist.add_trace(go.Histogram(
            x=data,
            name=name,
            # marker_color=color,
        ), row=pos[0], col=pos[1])
        fig_hist.add_vline(
            x=np.mean(data), 
            line_width=2, 
            line_dash="dash", 
            line_color='black',
            annotation_text=f'Mean: {np.mean(data):.2f}',
            annotation_position="top right", 
            row=pos[0], col=pos[1])
    fig_hist.update_layout(
        title='Portfolio Optimization Histograms',
        showlegend=False,
        xaxis_title='Return',
        yaxis_title='Frequency',
        xaxis2_title='Volatility',
        yaxis2_title='Frequency',
        xaxis3_title='Sharpe Ratio',
        yaxis3_title='Frequency',
        xaxis4_title='VaR',
        yaxis4_title='Frequency',
        )
    fig_hist.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
    fig_hist.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
    # fig_hist.write_html('portfolio_optimization_histograms.html', auto_open=True)

    return fig, fig_hist

def simulate_portfolio_performance(
        mean_returns, 
        cov_matrix,
        var, 
        num_portfolios=100, 
        risk_free_rate=0.02):
    
    num_assets = len(mean_returns)
    # Generate random portfolios
    results = np.zeros((4, num_portfolios))  # Store return, volatility, Sharpe Ratio
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
        
        # calculate value at risk
        VaR = np.dot(weights, var)     # in percentage
        
        
        # Calculate portfolio return and risk
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = sharpe_ratio
        results[3, i] = VaR    
    
    # Plot results
    fig, fig_hist = plot_results(results, risk_free_rate, show_cml=False)
    return fig, fig_hist

def plot_paths(paths, times=None, N_show=10):
    if times is None:
        times = np.arange(paths.shape[1])
    
    simulations = paths.shape[0]
    if N_show > simulations:
        N_show = simulations

    fig = make_subplots(
    rows=1, cols=2, 
    shared_yaxes=True, 
    horizontal_spacing=0.02,
    column_widths=[0.8, 0.2]
    )
    for i in range(min(simulations, N_show)):  # Plot only the first 10 paths for clarity
        fig.add_trace(go.Scatter(
        x=times, 
        y=paths[i], 
        mode='lines', 
        name=f"Path {i+1}",
        line=dict(width=0.7),
        showlegend=False,
        ), row=1, col=1)
    
    fig.add_trace(go.Histogram(
        y=paths[:, -1],
        marker=dict(color='gray'),
        showlegend=False,
        orientation='h',
    ), row=1, col=2)

    fig.update_layout(
        title=f'N = {simulations} paths',
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis2_title='Count',
        template='seaborn',
    )

    return fig

# Call the function
# np.random.seed(seed) if seed else None
with st.sidebar:
    st.title('Portfolio Optimization')
    st.write('This app simulates portfolio performance, generates the Efficient Frontier, and identifies the portfolio with the maximum Sharpe Ratio.')
    num_assets = st.slider('Number of Assets', min_value=2, max_value=100, value=5, step=1)
    num_portfolios = st.slider('Number of Portfolios', min_value=100, max_value=10000, value=1000, step=100)
    risk_free_rate = st.slider('Risk-Free Rate (%)', min_value=0.0, max_value=.1, value=0.02, step=0.01)
    risk_free_rate = 0.02

    random_returns = st.checkbox('Use Random Returns', value=False)
    seed = st.slider('Random Seed', min_value=0, max_value=1000, value=0, step=1)


    S0_low = st.slider('Initial Price Lower Bound', min_value=50, max_value=100, value=100, step=1)
    S0_high = st.slider('Initial Price Higher Bound', min_value=100, max_value=200, value=200, step=1)
    S0 = np.random.uniform(S0_low, S0_high, num_assets)            # Initial price
    
    mu_low = st.slider('Annual Drift Lower Bound (%)', min_value=5, max_value=20, value=5, step=1)
    mu_high = st.slider('Annual Drift Higher Bound (%)', min_value=20, max_value=50, value=20, step=1)
    mu = np.random.uniform(mu_low/100, mu_high/100, num_assets)           # Annual drift (5%-20%)
    
    sigma_low = st.slider('Annual Volatility Lower Bound (%)', min_value=10, max_value=30, value=10, step=1)
    sigma_high = st.slider('Annual Volatility Higher Bound (%)', min_value=30, max_value=50, value=30, step=1)
    sigma = np.random.uniform(sigma_low/100, sigma_high/100, num_assets)         # Annual volatility (10%-30%)
    
    lambda_low = st.slider('Jump Frequency Lower Bound (%)', min_value=5, max_value=50, value=5, step=1)
    lambda_high = st.slider('Jump Frequency Higher Bound (%)', min_value=50, max_value=100, value=50, step=1)
    lambda_ = np.random.uniform(lambda_low/100, lambda_high/100, num_assets)         # Jump frequency (10%-50%)
    
    jump_mean_low = st.slider('Jump Size Mean Lower Bound (%)', min_value=-10, max_value=0, value=-10, step=1)
    jump_mean_high = st.slider('Jump Size Mean Higher Bound (%)', min_value=0, max_value=10, value=0, step=1)
    jump_mean = np.random.uniform(jump_mean_low/100, jump_mean_high/100, num_assets)    # Jump size mean (-10%-10%)
    
    jump_std_low = st.slider('Jump Size Standard Deviation Lower Bound (%)', min_value=0, max_value=10, value=10, step=1)
    jump_std_high = st.slider('Jump Size Standard Deviation Higher Bound (%)', min_value=10, max_value=30, value=30, step=1)
    jump_std = np.random.uniform(jump_std_low/100, jump_std_high/100, num_assets)      # Jump size std (10%-30%)

    T = st.slider('Time horizon in years', min_value=1, max_value=5, value=2, step=1)              # Time horizon in years
    VaR_percentile = st.slider('VaR Percentile (%)', min_value=1, max_value=10, value=5, step=1)  # VaR percentile

    n_paths = st.slider('Number of Price Paths to Show', min_value=10, max_value=1000, value=100, step=10)

dt = 1/252         # Daily time steps


# Simulated returns and covariance for demonstration
if random_returns:
    mean_returns = np.random.uniform(0.05, 0.2, num_assets)  # Expected annual returns
    cov_matrix = np.random.uniform(0.01, 0.05, (num_assets, num_assets))
    cov_matrix = (cov_matrix + cov_matrix.T) / 2  # Symmetric covariance matrix
else:
    sp = StochasticProcesses()
    prices = [sp.jump_diffusion(S0[i], mu[i], sigma[i], lambda_[i], jump_mean[i], jump_std[i], T, dt, 1)[1] for i in range(num_assets)]
    prices = np.array(prices).reshape(num_assets, -1)
    fig_paths = plot_paths(prices, N_show=n_paths)
    returns = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1] * 100
    # log_returns = np.log(prices[:, 1:] / prices[:, :-1])
    # plot_jump_diffusion_simulation(num_assets, prices)

    # annual returns
    mean_returns = (prices[:, -1] - S0) / S0
    # covariance matrix
    cov_matrix = np.cov(mean_returns)

    var = np.percentile(returns, VaR_percentile, axis=1)



fig, fig_hist = simulate_portfolio_performance(
    mean_returns,
    cov_matrix,
    var,
    num_portfolios,
    risk_free_rate,
    )

st.title('Portfolio Optimization')
st.plotly_chart(fig)
st.title('Histograms')
st.plotly_chart(fig_hist)
st.title('Price Paths')
st.plotly_chart(fig_paths)