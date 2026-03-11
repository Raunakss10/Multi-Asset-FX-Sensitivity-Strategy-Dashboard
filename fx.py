# =========================================================
# Multi-Asset FX Sensitivity & Strategy Dashboard
# =========================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 0) CONFIG
# ---------------------------------------------------------

plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["font.size"] = 11

pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", "{:,.4f}".format)

SEED = 42
np.random.seed(SEED)

os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------
# 1) BUILD SYNTHETIC MULTI-ASSET MACRO DATA
# ---------------------------------------------------------

dates = pd.bdate_range("2021-01-01", "2025-12-31")
n = len(dates)

factors = [
    "us2y_change",
    "us10y_change",
    "de10y_change",
    "jp10y_change",
    "vix_change",
    "move_change",
    "spx_return",
    "sx5e_return",
    "nikkei_return",
    "wti_return",
    "gold_return",
    "copper_return",
]

factor_scales = {
    "us2y_change": 0.035,
    "us10y_change": 0.025,
    "de10y_change": 0.022,
    "jp10y_change": 0.010,
    "vix_change": 0.050,
    "move_change": 0.045,
    "spx_return": 0.010,
    "sx5e_return": 0.011,
    "nikkei_return": 0.012,
    "wti_return": 0.018,
    "gold_return": 0.009,
    "copper_return": 0.014,
}

macro = pd.DataFrame(index=dates)

for col in factors:
    eps = np.random.normal(0, factor_scales[col], n)
    arr = np.zeros(n)
    phi = 0.10 if "return" in col else 0.15

    for i in range(1, n):
        arr[i] = phi * arr[i - 1] + eps[i]

    macro[col] = arr

# ---------------------------------------------------------
# 2) SIMULATE FX RETURNS WITH TIME-VARYING BETAS
# ---------------------------------------------------------

fx_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

base_betas = {
    "EURUSD": {
        "us2y_change": -0.30,
        "us10y_change": -0.20,
        "de10y_change": 0.25,
        "vix_change": -0.10,
        "spx_return": 0.08,
        "sx5e_return": 0.10,
        "wti_return": 0.02,
        "gold_return": 0.03,
        "copper_return": 0.02,
    },
    "GBPUSD": {
        "us2y_change": -0.25,
        "us10y_change": -0.12,
        "de10y_change": 0.10,
        "vix_change": -0.08,
        "spx_return": 0.07,
        "sx5e_return": 0.08,
        "wti_return": 0.03,
        "gold_return": 0.01,
        "copper_return": 0.02,
    },
    "USDJPY": {
        "us2y_change": 0.35,
        "us10y_change": 0.18,
        "jp10y_change": -0.20,
        "vix_change": 0.15,
        "nikkei_return": 0.05,
        "gold_return": -0.02,
    },
    "AUDUSD": {
        "us2y_change": -0.18,
        "vix_change": -0.12,
        "spx_return": 0.08,
        "wti_return": 0.06,
        "gold_return": 0.10,
        "copper_return": 0.20,
    },
    "USDCAD": {
        "us2y_change": 0.10,
        "us10y_change": 0.08,
        "vix_change": 0.05,
        "wti_return": -0.22,
        "gold_return": -0.03,
        "copper_return": -0.04,
        "spx_return": -0.02,
    },
}


def smooth_regime(length, amplitude=0.10, periods=3):
    x = np.linspace(0, periods * 2 * np.pi, length)
    return amplitude * np.sin(x)


fx = pd.DataFrame(index=dates)

for pair in fx_pairs:
    signal = np.zeros(n)

    for factor, beta in base_betas[pair].items():
        dynamic_beta = beta + smooth_regime(
            n,
            amplitude=abs(beta) * 0.25 + 0.02,
            periods=2.4,
        )
        signal += dynamic_beta * macro[factor].values

    noise = np.random.normal(0, 0.0045, n)
    fx[pair] = signal + noise

# ---------------------------------------------------------
# 3) MODELING FRAME
# ---------------------------------------------------------

data = fx.join(macro, how="inner")

# ---------------------------------------------------------
# 4) ROLLING OLS FUNCTION
# ---------------------------------------------------------

def rolling_ols_betas(y, X, window=63):
    """
    Estimate rolling OLS coefficients with numpy lstsq.
    Returns a DataFrame containing intercept + factor betas.
    """
    cols = ["intercept"] + list(X.columns)
    out = pd.DataFrame(index=y.index, columns=cols, dtype=float)

    for end in range(window, len(y) + 1):
        idx = y.index[end - window:end]
        yw = y.loc[idx].values
        Xw = X.loc[idx].values
        Xw = np.column_stack([np.ones(len(Xw)), Xw])

        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        out.iloc[end - 1] = beta

    return out


# ---------------------------------------------------------
# 5) ESTIMATE ROLLING BETAS
# ---------------------------------------------------------

X = macro.copy()
window = 63

rolling_betas = {}
for pair in fx_pairs:
    rolling_betas[pair] = rolling_ols_betas(data[pair], X, window=window)

# ---------------------------------------------------------
# 6) SAVE ROLLING BETA CHARTS
# ---------------------------------------------------------

headline_factors = ["us2y_change", "vix_change", "wti_return", "copper_return"]

for pair in fx_pairs:
    ax = rolling_betas[pair][headline_factors].plot(
        title=f"{pair} | Rolling Macro Betas (63d OLS)",
        figsize=(10, 5),
    )
    plt.axhline(0, linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("Beta")
    plt.tight_layout()
    plt.savefig(f"figures/{pair.lower()}_betas.png", dpi=300, bbox_inches="tight")
    plt.show()

# ---------------------------------------------------------
# 7) LATEST CROSS-SECTIONAL EXPOSURES
# ---------------------------------------------------------

latest_exposure = pd.DataFrame({
    pair: rolling_betas[pair].iloc[-1]
    for pair in fx_pairs
}).T

print("\nLatest headline exposures:")
print(latest_exposure[headline_factors].sort_index().round(4))

latest_exposure[headline_factors].plot(
    kind="bar",
    title="Latest FX Factor Exposures",
    figsize=(10, 5),
)
plt.axhline(0, linewidth=1)
plt.xlabel("FX Pair")
plt.ylabel("Latest beta")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("figures/latest_exposures.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------
# 8) MACRO SCORE CONSTRUCTION
# ---------------------------------------------------------

lookback_signal = 10
recent_factor_move = X.tail(lookback_signal).mean()

macro_scores = {}
for pair in fx_pairs:
    latest_beta_vector = rolling_betas[pair].iloc[-1][X.columns]
    macro_scores[pair] = float((latest_beta_vector * recent_factor_move).sum())

macro_score_df = (
    pd.Series(macro_scores, name="macro_score")
    .sort_values(ascending=False)
    .to_frame()
)

print("\nMacro scores:")
print(macro_score_df.round(4))

macro_score_df.plot(
    kind="bar",
    title="Macro Score by FX Pair",
    figsize=(10, 5),
    legend=False,
)
plt.axhline(0, linewidth=1)
plt.xlabel("FX Pair")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("figures/macro_scores.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------
# 9) STRATEGY SIGNALS
# ---------------------------------------------------------

def signal_from_score(score, threshold=0.0008):
    if pd.isna(score):
        return np.nan
    if score > threshold:
        return 1
    if score < -threshold:
        return -1
    return 0


strategy_scores = pd.DataFrame(index=data.index, columns=fx_pairs, dtype=float)

for dt in strategy_scores.index:
    pos = data.index.get_loc(dt)
    if pos < max(window, lookback_signal):
        continue

    recent_move = X.iloc[pos - lookback_signal:pos].mean()

    for pair in fx_pairs:
        beta_row = rolling_betas[pair].loc[dt, X.columns]
        if beta_row.isna().any():
            continue
        strategy_scores.loc[dt, pair] = float((beta_row * recent_move).sum())

strategy_signals = strategy_scores.map(signal_from_score)
next_day_returns = fx.shift(-1)

strategy_returns = strategy_signals * next_day_returns
strategy_returns["equal_weight"] = strategy_returns[fx_pairs].mean(axis=1)

# ---------------------------------------------------------
# 10) PERFORMANCE METRICS
# ---------------------------------------------------------

def annualized_return(r):
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    cumulative = (1 + r).prod()
    years = len(r) / 252
    return cumulative ** (1 / years) - 1


def annualized_vol(r):
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return r.std() * np.sqrt(252)


def sharpe_ratio(r):
    vol = annualized_vol(r)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return annualized_return(r) / vol


def max_drawdown(r):
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    wealth = (1 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1
    return dd.min()


perf = []
for col in strategy_returns.columns:
    series = strategy_returns[col].dropna()
    perf.append({
        "series": col,
        "ann_return": annualized_return(series),
        "ann_vol": annualized_vol(series),
        "sharpe": sharpe_ratio(series),
        "max_drawdown": max_drawdown(series),
        "hit_rate": (series > 0).mean() if len(series) > 0 else np.nan,
    })

perf_df = (
    pd.DataFrame(perf)
    .set_index("series")
    .sort_values("sharpe", ascending=False)
)

print("\nPerformance summary:")
print(perf_df.round(4))

# ---------------------------------------------------------
# 11) EQUITY CURVE
# ---------------------------------------------------------

equity_curve = (1 + strategy_returns["equal_weight"].fillna(0)).cumprod()

equity_curve.plot(
    title="Equal-Weighted Macro Signal Strategy | Cumulative Return",
    figsize=(10, 5),
)
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.tight_layout()
plt.savefig("figures/equity_curve.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------
# 12) CHARTS OF THE WEEK
# ---------------------------------------------------------

top_pair = macro_score_df.index[0]
bottom_pair = macro_score_df.index[-1]

rolling_betas[top_pair][headline_factors].tail(126).plot(
    title=f"Chart of the Week #1 | {top_pair} has the strongest positive macro score",
    figsize=(10, 5),
)
plt.axhline(0, linewidth=1)
plt.xlabel("Date")
plt.ylabel("Beta")
plt.tight_layout()
plt.savefig("figures/chart_of_week_1.png", dpi=300, bbox_inches="tight")
plt.show()

rolling_betas[bottom_pair][headline_factors].tail(126).plot(
    title=f"Chart of the Week #2 | {bottom_pair} has the weakest macro score",
    figsize=(10, 5),
)
plt.axhline(0, linewidth=1)
plt.xlabel("Date")
plt.ylabel("Beta")
plt.tight_layout()
plt.savefig("figures/chart_of_week_2.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------
# 13) SAVE OUTPUT TABLES
# ---------------------------------------------------------

latest_exposure.round(6).to_csv("figures/latest_exposure_table.csv")
macro_score_df.round(6).to_csv("figures/macro_score_table.csv")
perf_df.round(6).to_csv("figures/performance_summary.csv")

# ---------------------------------------------------------
# 14) FINAL SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)
print("Multi-Asset FX Sensitivity & Strategy Dashboard")
print("- Modeled G10 FX returns across 10+ macro factors using rolling OLS.")
print("- Estimated time-varying cross-asset factor sensitivities by FX pair.")
print("- Built a macro-aware directional score using rolling exposures and recent factor moves.")
print("- Evaluated a simple equal-weighted signal strategy using return, vol, Sharpe, hit rate, and drawdown.")
print("- Saved recruiter-ready charts into the figures/ folder.")
print("=" * 70)