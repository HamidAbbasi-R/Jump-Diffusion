#%%
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def jump_diffusion(S0, mu, sigma, lambda_, jump_mean, jump_std, T, dt, simulations):
    """
    Simulates asset prices using Jump-Diffusion (Merton) model.
    
    Parameters:
    - S0: Initial asset price
    - mu: Drift (mean return)
    - sigma: Volatility (standard deviation of returns)
    - lambda_: Poisson jump intensity (average number of jumps per year)
    - jump_mean: Mean of jump size (log-normal distribution, log scale)
    - jump_std: Standard deviation of jump size (log-normal distribution, log scale)
    - T: Time horizon (in years)
    - dt: Time step (e.g., 1/252 for daily steps)
    - simulations: Number of simulation paths
    
    Returns:
    - Simulated paths as a NumPy array of shape (simulations, time_steps)
    """
    time_steps = int(T / dt)
    times = np.linspace(0, T, time_steps)
    
    # Preallocate array for simulation results
    paths = np.zeros((simulations, time_steps))
    paths[:, 0] = S0  # Set initial price

    for t in range(1, time_steps):
        # Generate random normal increments for Brownian motion
        Z = np.random.normal(0, 1, simulations)
        
        # Generate Poisson-distributed number of jumps
        N = np.random.poisson(lambda_ * dt, simulations)
        
        # Generate jump sizes (log-normal distribution)
        J = np.random.normal(jump_mean, jump_std, simulations)
        jump_factor = np.exp(J)  # Convert to multiplicative jump sizes
        
        # Compute price using GBM with jumps
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z) \
                      * (jump_factor ** N)  # Include jumps if any
    
    return times, paths

def plot_jump_diffusion_simulation(simulations, paths, times=None, N=10):
    if times is None:
        times = np.arange(paths.shape[1])
    if N > simulations:
        N = simulations
    fig = make_subplots(
    rows=1, cols=2, 
    shared_yaxes=True, 
    horizontal_spacing=0.02,
    column_widths=[0.8, 0.2]
    )
    for i in range(min(simulations, N)):  # Plot only the first 10 paths for clarity
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
        marker=dict(color='lightgrey'),
        showlegend=False,
        orientation='h',
    ), row=1, col=2)

    fig.update_layout(
        title='Jump-Diffusion Model Simulation',
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis2_title='Count',
        template='plotly_dark',
    )

    fig.show()

# Parameters
S0 = 100           # Initial price
mu = 0.05          # Annual drift (5%)
sigma = 0.2        # Annual volatility (20%)
lambda_ = 0        # One jump per year on average
jump_mean = 0.02   # Average jump size (-2%)
jump_std = 0.1     # Jump size volatility (10%)
T = 1              # Time horizon in years
dt = 1/252         # Daily time steps
simulations = 100  # Number of simulation paths

# Run simulation
times, paths = jump_diffusion(S0, mu, sigma, lambda_, jump_mean, jump_std, T, dt, simulations)

plot_jump_diffusion_simulation(simulations, paths, times, N=100)