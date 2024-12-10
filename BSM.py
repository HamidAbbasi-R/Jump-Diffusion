#%%
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
    rows=2, cols=1, 
    shared_xaxes=True, 
    )

    fig.add_trace(go.Scatter(
        x=stock_prices, 
        y=call_prices, 
        mode='lines', 
        name = 'Call Option Price',
        # line=dict(color='blue', width=1),
        ), row=1, col=1)


    fig.add_trace(go.Scatter(
        x=stock_prices, 
        y=put_prices, 
        mode='lines', 
        name = 'Put Option Price',
        # line=dict(color='red', width=1),
        ), row=1, col=1)

    fig.add_vline(
        x=K, 
        line_width=1, 
        line_dash="dash", 
        line_color="black", 
        annotation_text="Strike Price", 
        annotation_position="top right",
        row=1, col=1)

    fig.add_trace(go.Scatter(
        x=stock_prices, 
        y=delta, 
        mode='lines', 
        name = 'Delta',
        # secondary y axis
        # line=dict(color='green', width=1),
        ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=stock_prices, 
        y=gamma, 
        mode='lines', 
        name = 'Gamma',
        # secondary y axis
        # yaxis="y2",
        # line=dict(color='orange', width=1),
        ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=stock_prices, 
        y=theta, 
        mode='lines', 
        name = 'Theta',
        # secondary y axis
        # yaxis="y2",
        # line=dict(color='purple', width=1),
        ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=stock_prices, 
        y=vega, 
        mode='lines', 
        name = 'Vega',
        # secondary y axis
        # yaxis="y2",
        # line=dict(color='brown', width=1),
        ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=stock_prices, 
        y=rho, 
        mode='lines', 
        name = 'Rho',
        # secondary y axis
        # yaxis="y2",
        # line=dict(color='pink', width=1),
        ), row=2, col=1)

    fig.add_vline(
        x=K, 
        line_width=1, 
        line_dash="dash", 
        line_color="black", 
        annotation_text="Strike Price", 
        annotation_position="top right",
        row=2, col=1)

    fig.update_layout(
        title='Black-Scholes Option Price',
        xaxis2_title='Stock Price',
        yaxis_title='Option Price',
        yaxis2_title='Greeks',
        # legend position
        # legend=dict(x=0, y=1.15),
        # secondary y axis
        # yaxis2=dict(title='Delta', overlaying='y', side='right'),
        # legend orientation
        # legend_orientation="h",
        )
    fig.show()

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
