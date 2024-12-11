#%% 
"""
This script implements the Black-Scholes model to calculate the prices of European call and put options, 
computes the option Greeks (Delta, Gamma, Theta, Vega, and Rho), 
and visualizes the results using interactive plots.
"""

import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Black-Scholes call option price function
def call_option_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    C = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return C

def put_option_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    P = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return P

def calculate_greeks(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    return delta, gamma, theta, vega, rho

def plotting():
    fig = make_subplots(
    rows=2, cols=3,
    )

    option_prices = [call_prices, put_prices]
    option_names = ['Call Option Price', 'Put Option Price']
    fill_patterns = ['\\', '/']

    for prices, name, pattern in zip(option_prices, option_names, fill_patterns):
        fig.add_trace(go.Scatter(
            x=stock_prices, 
            y=prices, 
            mode='lines', 
            name=name,
            fill='tozeroy',
            fillpattern={
                'shape': pattern,  # options are ['', '/', '\\', 'x', '-', '|', '+', '.']
                'size': 5,
            }
        ), row=1, col=1)

    greeks = [delta, gamma, theta, vega, rho]
    greek_names = ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho']
    positions = [(1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]

    for greek, name, pos in zip(greeks, greek_names, positions):
        fig.add_trace(go.Scatter(
            x=stock_prices, 
            y=greek, 
            mode='lines', 
            name=name,
            fill='tozeroy',
            fillpattern={
                'shape': '.',       # options are ['', '/', '\\', 'x', '-', '|', '+', '.']
                'size': 5,
            }
        ), row=pos[0], col=pos[1])

    # Add vertical line for strike price
    for i in range(1, 3):
        for j in range(1, 4):
            fig.add_vline(
                x=K, 
                line_width=1, 
                line_dash="dash", 
                line_color="black", 
                annotation_text="Strike Price", 
                annotation_position="top left",
                row=i, col=j)

    fig.update_layout(
        title='Black-Scholes Option Price',
        xaxis_title='Stock Price',
        xaxis2_title='Stock Price',
        xaxis3_title='Stock Price',
        xaxis4_title='Stock Price',
        xaxis5_title='Stock Price',
        xaxis6_title='Stock Price',

        yaxis_title='Option Price',
        yaxis2_title='Delta',
        yaxis3_title='Gamma',
        yaxis4_title='Theta',
        yaxis5_title='Vega',
        yaxis6_title='Rho',

        height=800,
        width=1200,
        # legend position
        legend_orientation="h",
        legend=dict(x=0, y=1.05),
        # secondary y axis
        # yaxis2=dict(title='Delta', overlaying='y', side='right'),
        # legend orientation
        )
    fig.write_html("Black_Scholes_Model.html", auto_open=True)

# Parameters
K = 50       # Strike price
T = 1        # Time to expiration (1 year)
r = 0.05      # Risk-free rate (5%)
sigma = 0.1   # Volatility (10% and 20%)

# Stock prices range
stock_prices = np.linspace(K*0.7, K*1.3, 200)  # Stock prices from $30 to $70

call_prices = call_option_price(stock_prices, K, T, r, sigma) 
put_prices = put_option_price(stock_prices, K, T, r, sigma) 

# Greeks
delta, gamma, theta, vega, rho = calculate_greeks(stock_prices, K, T, r, sigma)

plotting()
