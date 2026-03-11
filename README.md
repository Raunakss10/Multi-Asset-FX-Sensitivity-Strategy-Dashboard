## 📈 Project Overview
# Multi-Asset FX Sensitivity & Strategy Dashboard

This project builds a cross-asset macro framework to analyze how major G10 FX pairs respond to interest rates, volatility, commodities, and risk sentiment.

Using rolling OLS regressions, the project estimates **time-varying factor sensitivities** for major FX pairs and converts those exposures into a **macro-aware directional signal framework**.

---

## Overview

The workflow follows four stages:

1. **Macro factor construction**  
   Build a dataset spanning rates, volatility, commodities, and equity risk factors.

2. **Rolling factor estimation**  
   Estimate rolling OLS betas for each FX pair against the macro factor set.

3. **Signal construction**  
   Translate estimated exposures and recent macro moves into a cross-sectional macro score.

4. **Strategy evaluation**  
   Backtest a simple macro-aware strategy and evaluate returns, volatility, Sharpe ratio, hit rate, and drawdown.

---

## FX Pairs Studied

- EURUSD
- GBPUSD
- USDJPY
- AUDUSD
- USDCAD

## Macro Factors Used

- US 2Y rate change
- US 10Y rate change
- Germany 10Y rate change
- Japan 10Y rate change
- VIX change
- MOVE change
- S&P 500 return
- Euro Stoxx return
- Nikkei return
- WTI return
- Gold return
- Copper return

---

## Methodology

For each currency pair, rolling OLS regressions are estimated over a 63-day window:

\[
r_t^{FX} = \alpha + \beta_1 X_{1,t} + \beta_2 X_{2,t} + \cdots + \beta_k X_{k,t} + \varepsilon_t
\]

This produces time-varying macro sensitivities rather than static coefficients.

A macro score is then built by combining:
- latest rolling factor exposures
- recent macro factor direction

Those scores are converted into simple directional trading signals.

---

## Example Outputs

### Rolling FX Betas
![EURUSD Betas](figures/eurusd_betas.png)

### Latest Cross-Sectional Exposures
![Latest Exposures](figures/latest_exposures.png)

### Macro Score by FX Pair
![Macro Scores](figures/macro_scores.png)

### Strategy Equity Curve
![Equity Curve](figures/equity_curve.png)

---

## Key Takeaways

- Major FX pairs exhibit differentiated exposure to rates, volatility, and commodity factors.
- Rolling regressions are useful for capturing time-varying market regimes.
- Cross-asset exposure analysis can be translated into a simple signal-generation framework.
- The project demonstrates a full pipeline from factor modeling to backtesting.

---

## Repository Contents

- `fx_macro_dashboard.py` — main research and plotting script
- `notebook/FX_Macro_Research.ipynb` — notebook version
- `FX_Macro_Project_Report.tex` — report source
- `FX_Macro_Project_Report.pdf` — exported research report
- `figures/` — saved charts used in the report and README

---

## Future Improvements

- Replace synthetic data with real Bloomberg / Yahoo Finance / FRED data
- Add transaction costs and volatility targeting
- Include rolling t-stats and exposure heatmaps
- Extend to PCA factors or regime clustering
- Package as a Streamlit dashboard

---

## Author

Raunak  
UC San Diego — Data Science
This project aims to:

- Use historical financial and macroeconomic data to forecast **WTI Crude Oil (CL=F)** returns.
- Engineer time-series features and train a **Random Forest model** for prediction.
- Generate trading signals (long/short) based on predictions.
- Evaluate the profitability and performance of the resulting strategy using metrics like **Sharpe Ratio** and **cumulative return**.

---

## 📊 Dataset & Features

**Data Source**: [Yahoo Finance](https://finance.yahoo.com)

### Instruments Used:
- WTI Crude Oil Futures (CL=F)
- US Dollar Index (DX-Y.NYB)
- 10-Year Treasury Yield (^TNX)

### Engineered Features:
- Daily returns for each instrument
- Lagged returns for past 1, 2, and 3 days
- Target variable: next-day return of WTI crude oil

---

## 🧠 Machine Learning

- **Model**: Random Forest Regressor
- **Libraries**: `scikit-learn`, `pandas`, `matplotlib`, `yfinance`
- **Train/Test Split**:
  - Train: Jan 2018 – Dec 2021
  - Test: Jan 2022 – present

### 📌 Results:
- Test MSE: `0.000883`
- Test R²: `-0.6233`
- **Top Features**: Current and lagged WTI returns, Treasury yield changes

---

## 💰 Trading Strategy

A simple rule-based strategy was tested:

- **Go long** if predicted return > 0  
- **Go short** if predicted return < 0  
- **Returns** are calculated using actual next-day returns and model signal

### 📈 Performance:
- **Cumulative Return**: `+99.53%` (test period)
- **Sharpe Ratio**: `0.76`

---

## 🔧 Folder Structure

