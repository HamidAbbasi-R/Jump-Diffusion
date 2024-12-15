#%%
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np

class StochasticProcesses:
    def __init__(self):
        pass

    def brownian_motion(self, S0, mu, sigma, T, dt, simulations):
        """
        Simulates Brownian Motion.
        
        Parameters:
        - S0: Initial value
        - mu: Drift
        - sigma: Volatility
        - T: Time horizon (in years)
        - dt: Time step
        - simulations: Number of simulation paths
        
        Returns:
        - Simulated paths as a NumPy array
        """
        time_steps = int(T / dt)
        times = np.linspace(0, T, time_steps)
        paths = np.zeros((simulations, time_steps))
        paths[:, 0] = S0

        for t in range(1, time_steps):
            Z = np.random.normal(0, 1, simulations)
            paths[:, t] = paths[:, t-1] + mu * dt + sigma * np.sqrt(dt) * Z

        return times, paths

    def geometric_brownian_motion(self, S0, mu, sigma, T, dt, simulations):
        """
        Simulates Geometric Brownian Motion.
        
        Parameters:
        - S0: Initial value
        - mu: Drift
        - sigma: Volatility
        - T: Time horizon (in years)
        - dt: Time step
        - simulations: Number of simulation paths
        
        Returns:
        - Simulated paths as a NumPy array
        """
        time_steps = int(T / dt)
        times = np.linspace(0, T, time_steps)
        paths = np.zeros((simulations, time_steps))
        paths[:, 0] = S0

        for t in range(1, time_steps):
            Z = np.random.normal(0, 1, simulations)
            paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

        return times, paths

    def jump_diffusion(self, S0, mu, sigma, lambda_, jump_mean, jump_std, T, dt, simulations):
        """
        Simulates Jump-Diffusion (Merton) model.
        
        Parameters:
        - S0: Initial value
        - mu: Drift
        - sigma: Volatility
        - lambda_: Poisson jump intensity
        - jump_mean: Mean jump size (log scale)
        - jump_std: Jump size standard deviation (log scale)
        - T: Time horizon (in years)
        - dt: Time step
        - simulations: Number of simulation paths
        
        Returns:
        - Simulated paths as a NumPy array
        """
        time_steps = int(T / dt)
        times = np.linspace(0, T, time_steps)
        paths = np.zeros((simulations, time_steps))
        paths[:, 0] = S0

        for t in range(1, time_steps):
            Z = np.random.normal(0, 1, simulations)
            N = np.random.poisson(lambda_ * dt, simulations)
            J = np.random.normal(jump_mean, jump_std, simulations)
            jump_factor = np.exp(J)
            paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z) * (jump_factor ** N)

        return times, paths

    def ornstein_uhlenbeck(self, S0, mu_OU, sigma, theta, T, dt, simulations):
        """
        Simulates Ornstein-Uhlenbeck process.
        
        Parameters:
        - S0: Initial value
        - mu_OU: Long-term mean
        - sigma: Volatility
        - theta: Mean reversion speed
        - T: Time horizon (in years)
        - dt: Time step
        - simulations: Number of simulation paths
        
        Returns:
        - Simulated paths as a NumPy array
        """
        time_steps = int(T / dt)
        times = np.linspace(0, T, time_steps)
        paths = np.zeros((simulations, time_steps))
        paths[:, 0] = S0

        for t in range(1, time_steps):
            Z = np.random.normal(0, 1, simulations)
            paths[:, t] = paths[:, t-1] + theta * (mu_OU - paths[:, t-1]) * dt + sigma * np.sqrt(dt) * Z

        return times, paths

    def cox_ingersoll_ross(self, S0, mu, sigma, theta, T, dt, simulations):
        """
        Simulates Cox-Ingersoll-Ross (CIR) process.
        
        Parameters:
        - S0: Initial value
        - mu: Long-term mean
        - sigma: Volatility
        - theta: Mean reversion speed
        - T: Time horizon (in years)
        - dt: Time step
        - simulations: Number of simulation paths
        
        Returns:
        - Simulated paths as a NumPy array
        """
        time_steps = int(T / dt)
        times = np.linspace(0, T, time_steps)
        paths = np.zeros((simulations, time_steps))
        paths[:, 0] = S0

        for t in range(1, time_steps):
            Z = np.random.normal(0, 1, simulations)
            paths[:, t] = paths[:, t-1] + theta * (mu - paths[:, t-1]) * dt + sigma * np.sqrt(paths[:, t-1] * dt) * Z
            paths[:, t] = np.maximum(paths[:, t], 0)  # Ensure non-negativity

        return times, paths

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
        title='Jump-Diffusion Model Simulation',
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis2_title='Count',
        template='seaborn',
    )

    fig.show()

# Parameters
S0 = 100           # Initial price
mu = 0.05          # Annual drift (5%)
mu_OU = 100        # Mean return for OU process
sigma = 0.2        # Annual volatility (20%)
lambda_ = 5        # One jump per year on average
theta = 0.1       # Mean reversion speed
jump_mean = 0.05   # Average jump size (-2%)
jump_std = 0.1     # Jump size volatility (10%)
T = 1              # Time horizon in years
dt = 1/252         # Daily time steps
simulations = 10000  # Number of simulation paths (run in parallel)

sp = StochasticProcesses()
times, paths = sp.ornstein_uhlenbeck(S0, mu_OU, sigma, theta, T, dt, simulations)

plot_paths(paths, times, N_show=100)