# Multi-Asset FX Sensitivity & Strategy Dashboard

This project builds a cross-asset macro framework to analyze how major **G10 FX currency pairs** respond to changes in interest rates, volatility, commodities, and global equity markets.

Using **63-day rolling OLS regressions**, the model estimates **time-varying macro factor sensitivities** for each FX pair. These estimated exposures are then combined with recent macro factor movements to construct a **macro score**, which is converted into simple directional trading signals.

The project demonstrates a full quantitative research workflow:

- macro factor construction  
- rolling regression modeling  
- cross-sectional exposure analysis  
- macro signal generation  
- simple strategy backtest  

Example outputs produced by the project include:

- rolling FX factor exposure charts  
- cross-sectional exposure comparisons across currency pairs  
- macro score rankings  
- cumulative strategy performance  

The repository includes the main analysis script and a research-style report summarizing the methodology and results.

## FX Pairs Analyzed

- EURUSD  
- GBPUSD  
- USDJPY  
- AUDUSD  
- USDCAD  

## Repository Contents

- `fx.py` — main research and analysis script  
- `FX.pdf` — full research report  
- `README.md` — project overview  

## Future Improvements

- Replace synthetic data with real market data (FRED / Yahoo Finance / Bloomberg)
- Add transaction costs and volatility targeting
- Extend factor set or apply PCA for factor reduction
- Deploy the framework as an interactive dashboard