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


# Parameters for testing
# S0 = 100           # Initial price
# mu = 0.05          # Annual drift (5%)
# mu_OU = 100        # Mean return for OU process
# sigma = 0.2        # Annual volatility (20%)
# lambda_ = 2        # One jump per year on average
# theta = 0.1       # Mean reversion speed
# jump_mean = 0.05   # Average jump size (-2%)
# jump_std = 0.1     # Jump size volatility (10%)
# T = 1              # Time horizon in years
# dt = 1/252         # Daily time steps
# simulations = 10000  # Number of simulation paths (run in parallel)

# sp = StochasticProcesses()
# times_bm, paths_bm = sp.brownian_motion(S0, mu, sigma, T, dt, simulations)
# times_gbm, paths_gbm = sp.geometric_brownian_motion(S0, mu, sigma, T, dt, simulations)
# times_jd, paths_jd = sp.jump_diffusion(S0, mu, sigma, lambda_, jump_mean, jump_std, T, dt, simulations)
# times_ou, paths_ou = sp.ornstein_uhlenbeck(S0, mu_OU, sigma, theta, T, dt, simulations)
# times_cir, paths_cir = sp.cox_ingersoll_ross(S0, mu, sigma, theta, T, dt, simulations)

# plot_paths(paths_bm, times_bm, N_show=100)
# plot_paths(paths_gbm, times_gbm, N_show=100)
# plot_paths(paths_jd, times_jd, N_show=100)