# Documentation for Financial Modeling Scripts

This document combines the documentation for three key financial modeling scripts:
<!-- create links to different section -->

## Table of Contents
- [Black-Scholes Option Pricing and Greeks Analysis](#1-black-scholes-option-pricing-and-greeks-analysis)
- [Portfolio Simulation and Visualization with Efficient Frontier](#2-portfolio-simulation-and-visualization-with-efficient-frontier)
- [Jump-Diffusion Model Simulation Script](#3-jump-diffusion-model-simulation-script)

---

# 1. Black-Scholes Option Pricing and Greeks Analysis

## Overview

This script implements the Black-Scholes model to calculate European option prices and key sensitivities (Greeks) such as delta, gamma, theta, vega, and rho. The model assumes a log-normal distribution of stock prices and constant volatility, making it ideal for pricing options under simplified market conditions.

## Key Features

- **Option Pricing**: Computes call and put prices for European options.
- **Greeks Calculation**: Evaluates the sensitivities of the option price to various parameters.
- **Visualization**: Plots option price and Greek values for a range of stock prices or times.

## Functions

### `call_option_price` and `put_option_price`

Calculates the price of a European call or put option using the Black-Scholes formula.

**Parameters**:

- `S` *(float)*: Current stock price.
- `K` *(float)*: Strike price of the option.
- `T` *(float)*: Time to maturity (in years).
- `r` *(float)*: Risk-free interest rate (annualized).
- `sigma` *(float)*: Volatility of the underlying stock (annualized).

**Returns**:

- `C`: Price of the call option.
- `P`: Price of the put option.

### `calculate_greeks`

Computes the Greeks for a given option.

**Parameters**:

- Same as `call_option_price` and `pull_option_price`.

**Returns**:

- `delta`: Sensitivity of the option price to changes in the stock price.
- `gamma`: Rate of change of delta with respect to the stock price.
- `theta`: Rate of change of the option price with respect to time.
- `vega`: Sensitivity of the option price to changes in volatility.
- `rho`: Sensitivity of the option price to changes in the risk-free rate.

### `plotting`

Plots the option prices and Greeks over a range of stock prices.

---

# 2. Portfolio Simulation and Visualization with Efficient Frontier

## Overview

This script performs portfolio optimization to maximize returns for a given level of risk or minimize risk for a target return. It also visualizes the efficient frontier, displaying the trade-off between risk and return. It uses Monte Carlo simulation to generate random portfolios and identify the optimal portfolios along the efficient frontier.

This script can use random returns and covariance matrices for assets, or calculate the returns and covariance matrices from jump-diffusion simulations with a range of predefined parameters. When using jump-diffusion simulations, MPT uses the historical returns and covariance matrices of the assets (not the *expected* returns and covariance matrices).

Ideally this script should use historical data and should be linked with another framework to make a prediction of future returns and covariance matrices. 

## Key Features

- **Portfolio Simulation**: Generates random portfolios based on asset weights. The aim is to make a visual representation of the efficient frontier along with all the possible portfolios.
- **Efficient Frontier**: Identifies optimal portfolios along the frontier.
- **Visualization**: Plots portfolio compositions and the efficient frontier.

## Functions

### `simulate_portfolio_performance`

Generates random portfolios and calculates their returns and risks.

**Parameters**:

- `mean_returns` *(array)*: Historical returns of assets. Could be expected returns.
- `cov_matrix` *(array)*: Covariance matrix of asset returns.
- `num_portfolios` *(int)*: Number of portfolios to simulate. Default is 100.
- `risk_free_rate` *(float)*: Annualized risk-free rate for Sharpe ratio calculation. Default is 0.02.

### `plot_results`

Plots the simulated portfolios on a scatter plot.

**Parameters**:
- `results`: *(array)*: Portfolio performance metrics. Output of `simulate_portfolio_performance`.
- `risk_free_rate` *(float)*: Annualized risk-free rate for Sharpe ratio calculation. Default is 0.02.
- `show_cml` *(bool)*: Whether to show the Capital Market Line. Default is `True`.

### `find_efficient_frontier`

Finds the optimal portfolios from a set of simulated portfolios.

**Parameters**:

- `results` *(array)*: Portfolio performance metrics. Output of `simulate_portfolio_performance`.

---

# 3. Jump-Diffusion Model Simulation Script

## Overview

This script simulates asset price movements using the **Jump-Diffusion Model** (also known as the Merton model). The model combines the features of Geometric Brownian Motion (GBM) with jumps, capturing both continuous price changes and sudden, discrete price movements due to external shocks.

## Key Features

- **Jump-Diffusion Model**: Combines Geometric Brownian Motion with a Poisson-driven jump component.
- **Simulation**: Customizable parameters for drift, volatility, jump intensity, jump size, and number of simulation paths.
- **Visualization**: Simulates multiple asset price paths and displays a histogram of terminal prices.

## Functions

### `jump_diffusion`

Simulates asset prices using the Jump-Diffusion model.

**Parameters**:

- `S0` *(float)*: Initial asset price.
- `mu` *(float)*: Drift or average return (annualized).
- `sigma` *(float)*: Volatility or standard deviation of returns (annualized).
- `lambda_` *(float)*: Intensity of jumps (average number of jumps per year).
- `jump_mean` *(float)*: Mean of log jump sizes (in log-normal terms).
- `jump_std` *(float)*: Standard deviation of log jump sizes.
- `T` *(float)*: Time horizon for the simulation (in years).
- `dt` *(float)*: Time step (e.g., 1/252 for daily steps).
- `simulations` *(int)*: Number of simulation paths to generate.

**Returns**:

- `times`: A NumPy array of time points.
- `paths`: A NumPy array of simulated paths with shape `(simulations, time_steps)`.

### `plot_jump_diffusion_simulation`

Plots the simulated paths and a histogram of the final asset prices.

**Parameters**:

- `simulations` *(int)*: Number of simulated paths.
- `paths` *(array)*: Simulated asset price paths from the `jump_diffusion` function.
- `times` *(array, optional)*: Time points corresponding to the simulation. Defaults to index values if not provided.
- `N` *(int, optional)*: Number of paths to visualize in the plot. Defaults to 10.

---

# Usage Example

## Black-Scholes Option Pricing

```python
S = 100
K = 105
T = 1
r = 0.05
sigma = 0.2

call_price, put_price = black_scholes_price(S, K, T, r, sigma)
greeks = calculate_greeks(S, K, T, r, sigma)
plot_option_metrics('price', S, K, T, r, sigma, S_range=(50, 150))
```

## Portfolio Simulation and Visualization

```python
returns = np.random.randn(100, 5)  # Simulated returns for 5 assets
cov_matrix = np.cov(returns.T)
risk_free_rate = 0.02

simulated_data = simulate_portfolios(returns, cov_matrix, risk_free_rate, 1000)
efficient_frontier = calculate_efficient_frontier(returns, cov_matrix, np.linspace(0.05, 0.15, 50))
plot_efficient_frontier(simulated_data, efficient_frontier)
```

## Jump-Diffusion Model Simulation

```python
S0 = 100
mu = 0.05
sigma = 0.2
lambda_ = 1
jump_mean = 0.02
jump_std = 0.1
T = 1
dt = 1/252
simulations = 1000

times, paths = jump_diffusion(S0, mu, sigma, lambda_, jump_mean, jump_std, T, dt, simulations)
plot_jump_diffusion_simulation(simulations, paths, times, N=20)
```

---
