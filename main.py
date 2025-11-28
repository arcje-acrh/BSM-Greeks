import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

# ============================================================
# USER INPUTS
# ============================================================
TICKER = "HDFCBANK.NS"   # change to your chosen NIFTY 200 stock
RISK_FREE = 0.07        # annual risk-free rate
DAYS_IN_YEAR = 252
START_MONTHS = 3

# ============================================================
# BLACK–SCHOLES, GREEKS, IV
# ============================================================
def bs_price(S, K, T, r, sigma, opt_type="C"):
    if T <= 0 or sigma <= 0:
        return max(0.0, (S-K) if opt_type == "C" else (K-S))
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt_type == "C":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def greeks(S, K, T, r, sigma, opt_type="C"):
    if T <= 0 or sigma <= 0:
        return 0, 0, 0, 0, 0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    delta = norm.cdf(d1) if opt_type == "C" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    vega  = S*norm.pdf(d1)*np.sqrt(T)
    theta = -(S*norm.pdf(d1)*sigma/(2*np.sqrt(T))) \
            - (r*K*np.exp(-r*T)*norm.cdf(d2 if opt_type == "C" else -d2))
    rho   = K*T*np.exp(-r*T)*(norm.cdf(d2) if opt_type == "C" else -norm.cdf(-d2))
    return delta, gamma, vega, theta, rho

def implied_vol(mkt_price, S, K, T, r, opt_type="C"):
    if T <= 0 or mkt_price <= 0:
        return np.nan
    def f(sig):
        return bs_price(S, K, T, r, sig, opt_type) - mkt_price
    try:
        return brentq(f, 1e-4, 5.0, maxiter=100)
    except ValueError:
        return np.nan

# ============================================================
# PART A – DOWNLOAD PRICES, RETURNS, STATS
# ============================================================
data = yf.download(
    TICKER,
    period=f"{START_MONTHS}mo",
    interval="1d",
    auto_adjust=False  # explicit: we want 'Adj Close' column
)

# Handle possible MultiIndex columns
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

prices = data["Adj Close"].rename("price").to_frame().dropna()

prices["log_ret"] = np.log(prices["price"]).diff()

daily_std = prices["log_ret"].std()
ann_vol = daily_std * np.sqrt(DAYS_IN_YEAR)
skew = prices["log_ret"].skew()
kurt = prices["log_ret"].kurt()

summary = pd.DataFrame({
    "Metric": ["Daily mean log return", "Daily std", "Annualized vol", "Skewness", "Kurtosis"],
    "Value": [prices["log_ret"].mean(), daily_std, ann_vol, skew, kurt]
})
print("Part A – Summary statistics")
print(summary)

prices["price"].plot(title=f"{TICKER} Price – last {START_MONTHS} months")
plt.show()
prices["log_ret"].plot(title=f"{TICKER} Daily log returns")
plt.show()

# ============================================================
# PART B – OPTION PRICES (BSM)
# ============================================================
S0 = prices["price"].iloc[-1]
atm = S0

strikes = [
    atm * 0.95,
    atm * 0.98,
    atm,
    atm * 1.02,
    atm * 1.05,
]
maturities_days = [30, 60, 90]
maturities_T = [d / DAYS_IN_YEAR for d in maturities_days]

hist_sigma = ann_vol

rows = []
for K in strikes:
    for T in maturities_T:
        for opt_type in ["C", "P"]:
            price_bs = bs_price(S0, K, T, RISK_FREE, hist_sigma, opt_type)
            rows.append({
                "Strike": round(K, 2),
                "Maturity_days": int(T*DAYS_IN_YEAR),
                "Type": "Call" if opt_type == "C" else "Put",
                "BSM_price": price_bs
            })

option_table = pd.DataFrame(rows)
print("\nPart B – Option pricing table")
print(option_table)

# ============================================================
# PART C – GREEKS (HIST VOL) + SIMPLE IV SURFACE SETUP
# ============================================================
greek_rows = []
for K in strikes:
    for T in maturities_T:
        for opt_type in ["C", "P"]:
            d, g, v, t, rho = greeks(S0, K, T, RISK_FREE, hist_sigma, opt_type)
            greek_rows.append({
                "Strike": round(K, 2),
                "Maturity_days": int(T*DAYS_IN_YEAR),
                "Type": "Call" if opt_type == "C" else "Put",
                "Sigma_used": hist_sigma,
                "Delta": d,
                "Gamma": g,
                "Vega": v,
                "Theta": t,
                "Rho": rho
            })

greeks_hist = pd.DataFrame(greek_rows)
print("\nPart C – Greeks with historical volatility")
print(greeks_hist.head())

# ============================================================
# PART D – PORTFOLIO, HEDGE, PNL
# ============================================================
# Example portfolio using first 6 options
opt_subset = option_table.iloc[:6].copy()
opt_subset["Qty"] = [1, -1, 1, -1, 1, -1]

def greek_for_row(row, sigma):
    K = row["Strike"]
    T = row["Maturity_days"] / DAYS_IN_YEAR
    opt_type = "C" if row["Type"] == "Call" else "P"
    return greeks(S0, K, T, RISK_FREE, sigma, opt_type)

opt_subset[["Delta", "Gamma", "Vega", "Theta", "Rho"]] = opt_subset.apply(
    lambda r: pd.Series(greek_for_row(r, hist_sigma)), axis=1
)

for g in ["Delta", "Gamma", "Vega"]:
    opt_subset[f"Port_{g}"] = opt_subset[g] * opt_subset["Qty"]

portals = opt_subset[[f"Port_{x}" for x in ["Delta", "Gamma", "Vega"]]].sum()
print("\nPart D – Portfolio Greeks (unhedged)")
print(portals)

shares_hedge = -portals["Port_Delta"]  # Delta hedge with stock
gamma_hedge = 0.0                      # placeholder if you want to add options for Gamma hedge

hedged = portals.copy()
hedged["Port_Delta"] += shares_hedge
hedged["Port_Gamma"] += gamma_hedge
print("\nPart D – Portfolio Greeks after hedge")
print(hedged)

def portfolio_value(S, sigma=hist_sigma):
    total = 0.0
    for _, r in opt_subset.iterrows():
        K = r["Strike"]
        T = r["Maturity_days"] / DAYS_IN_YEAR
        opt_type = "C" if r["Type"] == "Call" else "P"
        price_opt = bs_price(S, K, T, RISK_FREE, sigma, opt_type)
        total += price_opt * r["Qty"]
    total += shares_hedge * S
    return total

base_val = portfolio_value(S0)
shocks = np.array([-0.02, -0.01, 0.01, 0.02])
pnl_rows = []
for s in shocks:
    S_new = S0 * (1 + s)
    v_new = portfolio_value(S_new)
    pnl_rows.append({"Shock_pct": s*100, "PnL": v_new - base_val})

pnl_table = pd.DataFrame(pnl_rows)
print("\nPart D – Hedged portfolio PnL under shocks")
print(pnl_table)

# ============================================================
# PART E – VAR (PARAMETRIC + HISTORICAL)
# ============================================================
prices["portfolio_val"] = prices["price"].apply(lambda S: portfolio_value(S))
prices["port_ret"] = prices["portfolio_val"].pct_change()
port_rets = prices["port_ret"].dropna()

mu = port_rets.mean()
sigma_p = port_rets.std()
var_95_param = -(mu + sigma_p * norm.ppf(0.05))
var_99_param = -(mu + sigma_p * norm.ppf(0.01))

hist_sample = port_rets.iloc[-60:]
var_95_hist = -np.percentile(hist_sample, 5)
var_99_hist = -np.percentile(hist_sample, 1)

var_table = pd.DataFrame({
    "Method": ["Parametric", "Parametric", "Historical", "Historical"],
    "CL": ["95%", "99%", "95%", "99%"],
    "VaR_1d": [var_95_param, var_99_param, var_95_hist, var_99_hist]
})
print("\nPart E – 1‑day VaR table")
print(var_table)