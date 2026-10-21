# GARCH(1,1) Volatility Modeling and Forecasting Engine

An end-to-end Quantitative Finance project implementing a **GARCH(1,1)** (Generalized Autoregressive Conditional Heteroskedasticity) model from scratch in Python using **Maximum Likelihood Estimation (MLE)**. The model captures time-varying volatility and volatility clustering in financial returns, validated against industry-standard libraries and projected into multi-step-ahead forecasts.

---

## 📌 Project Overview

Financial return series frequently exhibit **volatility clustering**—periods of high volatility tend to cluster together, violating the homoscedasticity assumption of standard linear models. The GARCH(1,1) model addresses this by explicitly modeling conditional variance as a function of past squared residuals (ARCH term) and past variances (GARCH term).

This project implements:
1. **Data Pipeline:** Automated financial time-series extraction for market indices (e.g., NIFTY 50 `^NSEI`) via `yfinance`.
2. **Custom MLE Engine:** Custom log-likelihood function formulation and optimization using `scipy.optimize` with parameter bounds and stationarity constraints.
3. **Forecasting Module:** Multi-step-ahead conditional variance projection over a 30-day horizon converging toward long-run average variance.
4. **Validation:** Cross-benchmarking custom MLE optimization parameters against the benchmark `arch` library.

---

## 🧮 Mathematical Formulation

The GARCH(1,1) model is specified as:

$$r_t = \mu + \epsilon_t$$

$$\epsilon_t = \sigma_t z_t \quad \text{where} \quad z_t \sim \mathcal{N}(0, 1)$$

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

### Model Constraints
To ensure positive variance and model stationarity:
- **Positivity:** $\omega > 0, \quad \alpha \ge 0, \quad \beta \ge 0$
- **Stationarity:** $\alpha + \beta < 1$

The **unconditional (long-run) variance** is given by:

$$\sigma_{unconditional}^2 = \frac{\omega}{1 - (\alpha + \beta)}$$

---

## 📊 Features & Visualizations

- **Return & Volatility Profiling:** Calculation of log/percentage returns and daily rolling volatility.
- **Log-Likelihood Optimization:** Minimization of negative log-likelihood via Nelder-Mead / L-BFGS-B algorithms.
- **Forecast Term Structure:** Analytical multi-step volatility term structure forecasting: $\hat{\sigma}_{t+k}^2 = \sigma_{unconditional}^2 + (\alpha + \beta)^{k-1} (\sigma_{t+1}^2 - \sigma_{unconditional}^2)$
- **Comparative Analysis:** Detailed plot comparing custom model predictions with the `arch` package output.

---

## 🛠️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-aayush5670/garch-volatility-engine.git
   cd garch-volatility-engine
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```

3. **Install Dependencies:**
   ```bash
   pip install numpy pandas matplotlib scipy yfinance arch
   ```

---

## 🚀 Quickstart Usage

```python
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# 1. Fetch Market Data
df = yf.download('^NSEI', start='2023-01-01', end='2024-01-01')
returns = 100 * df['Close'].pct_change().dropna()

# 2. Define Log-Likelihood Function
def garch_mle(params, returns):
    omega, alpha, beta = params
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * (returns[t-1]**2) + beta * sigma2[t-1]
        
    log_likelihood = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + (returns**2) / sigma2)
    return -log_likelihood  # Minimize negative log-likelihood

# 3. Optimize Parameters
initial_guess = [0.1, 0.1, 0.8]
bounds = [(1e-6, None), (0, 1), (0, 1)]
result = minimize(garch_mle, initial_guess, args=(returns,), method='Nelder-Mead', bounds=bounds)

print("Optimized Parameters (omega, alpha, beta):", result.x)
```

---

## 📈 Tech Stack

- **Language:** Python 3.x
- **Data Manipulation:** `pandas`, `numpy`
- **Optimization & Modeling:** `scipy.optimize`, `arch`
- **Data Source:** `yfinance`
- **Visualization:** `matplotlib`
