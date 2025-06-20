# 🧠 Trend-Following Asset Allocation Strategy (ML-Driven, Multi-Asset)

This project was developed as part of a graduate-level course in collaboration with three teammates. It combines technical indicators, machine learning models, and a macroeconomic overlay to construct a dynamic, rule-based asset allocation strategy.

📂 All output plots are available in the `Outputs/` folder.  
▶️ To run the full backtest, simply execute `run_strategy.py`.

> 📝 If you're interested in the full report detailing performance metrics and model validation, feel free to reach out!

---

## 🧭 Strategy Overview

Our approach combines:
- **Cross-sectional trend-following** within asset classes
- **Top-down macroeconomic regime filtering** across asset classes

The strategy targets medium-to-high risk investors seeking diversified global exposure and who believe in exploiting inefficiencies under the weak-form Efficient Market Hypothesis.

---

## ⚙️ Signal Engineering

We compute eight technical indicators per asset:
- 22-day and 65-day returns
- Exponential Moving Average (EMA)
- 65-day and 260-day MACD
- 22-day Bollinger Bands
- 65-day breakout signals

These indicators form our feature set `X = {X1, ..., X8}`. Monthly asset returns are used as targets (`Y`).

---

## 🤖 Machine Learning Models

We train two separate models:

- **Ridge Regression** (for expected return estimation)  
  Hyperparameter `α` is optimized using 5-fold cross-validation to maximize cross-validated R².

- **Random Forest Classifier** (for up/down probability)  
  Hyperparameters (n_estimators, max_depth, min_samples_split, etc.) are tuned via grid search to maximize accuracy.

Predictions are converted to **z-scores** within each asset class, then aggregated (Ridge + RF) to form a composite signal per asset.

---

## 📊 Portfolio Construction

1. **Within each asset class:**
   - Assets are ranked by z-scores
   - Weights are set as the inverse of rankings (top-ranked assets receive more capital)

2. **Across asset classes (macroeconomic overlay):**
   - Macroeconomic features: Real GDP YoY, CPI YoY, Unemployment Rate, Yield Curve
   - Regimes are identified via K-means clustering (4 clusters on quarterly data since 1954)
   - Asset class weights are adjusted according to regime-specific rules  
     (e.g., overweight equities in strong growth environments)

3. **Final portfolio weights** = asset-specific weights × regime-based class weights

---

## 📆 Backtesting Logic

At the end of each month:
1. Compute features and predictions
2. Generate asset rankings and assign intra-class weights
3. Identify the current macro regime and apply class-level adjustments
4. Construct the portfolio for the next month

---

## 🚀 Highlights

- Fully automated, interpretable, and rule-based
- Combines supervised ML + unsupervised macro clustering
- Tested across 50 assets from 4 major asset classes


