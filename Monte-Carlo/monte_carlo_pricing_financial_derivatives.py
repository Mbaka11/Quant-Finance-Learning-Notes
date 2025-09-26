import math
import numpy as np
from typing import Tuple, Optional

def bs_call_price(S0: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Black–Scholes European call (continuous dividend yield q).
    """
    if T <= 0:
        return max(0.0, S0 - K)
    if sigma <= 0:
        # With zero vol, price collapses to discounted intrinsic under forward drift r-q
        ST = S0 * math.exp((r - q) * T)
        payoff = max(0.0, ST - K)
        return payoff * math.exp(-r * T)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    # Standard normal CDF
    from math import erf, sqrt
    N = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))

    return S0 * math.exp(-q * T) * N(d1) - K * math.exp(-r * T) * N(d2)


def mc_euro_call_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    n_paths: int = 100_000,
    n_steps: int = 252,
    antithetic: bool = True,
    seed: Optional[int] = 42
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Monte Carlo price of a European call under GBM with continuous dividend yield.
    Uses exact GBM steps and optional antithetic variates.
    
    Returns:
        price: MC estimate
        se:    standard error of estimate
        ci:    95% confidence interval (low, high)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol   = sigma * math.sqrt(dt)

    # --- simulate terminal prices vectorized ---
    if antithetic:
        m = n_paths // 2
        Z = rng.standard_normal((m, n_steps))
        # pair antithetic variates (+Z, -Z)
        Z2 = -Z

        # log returns accumulation
        log_increments_1 = drift + vol * Z
        log_increments_2 = drift + vol * Z2

        ST1 = S0 * np.exp(log_increments_1.sum(axis=1))
        ST2 = S0 * np.exp(log_increments_2.sum(axis=1))
        ST = np.concatenate([ST1, ST2], axis=0)
        if ST.shape[0] < n_paths:  # if n_paths is odd, add one extra path
            extra = S0 * np.exp((drift + vol * rng.standard_normal(n_steps)).sum())
            ST = np.concatenate([ST, np.array([extra])], axis=0)
    else:
        Z = rng.standard_normal((n_paths, n_steps))
        log_increments = drift + vol * Z
        ST = S0 * np.exp(log_increments.sum(axis=1))

    # --- payoff, discounting, statistics ---
    payoffs = np.maximum(ST - K, 0.0)
    discounted = np.exp(-r * T) * payoffs

    price = discounted.mean()
    # standard error of the mean
    se = discounted.std(ddof=1) / math.sqrt(discounted.shape[0])
    ci = (price - 1.96 * se, price + 1.96 * se)

    return price, se, ci


if __name__ == "__main__":
    S0   = 281.68     # from your screenshot
    K    = 375.0      # strike
    T    = 1.0        # ~1 year to Sep 26, 2026 -> adjust precisely (days/365)
    r    = 0.05       # risk-free (continuous) approx
    q    = 0.015      # dividend yield (ballpark for SPY; adjust)
    sigma= 0.22       # implied vol guess; use option chain IV if available

    price_mc, se, (lo, hi) = mc_euro_call_price(S0, K, T, r, sigma, q, n_paths=100_000, n_steps=252, antithetic=True, seed=7)
    price_bs = bs_call_price(S0, K, T, r, sigma, q)

    print("MC price:", price_mc, "SE:", se, "95% CI:", (lo, hi))
    print("BS price:", price_bs)

