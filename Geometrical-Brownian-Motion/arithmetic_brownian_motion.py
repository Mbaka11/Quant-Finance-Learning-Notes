"""
Geometric Brownian Motion (GBM) simulator — annotated

Context & cross-refs:
- Step 2 notebook: method-of-moments for μ, σ from log-returns.
- Step 3 notebook: Itô result for log-prices and the **exact GBM step**.
- Step 4 notebook: calibration on real data + diagnostics & coverage.

Why two update rules?
- exact: S_{t+Δt} = S_t * exp((μ - 0.5 σ²) Δt + σ sqrt(Δt) Z), Z ~ N(0,1)
         -> **always positive**, matches GBM one-step distribution exactly
- euler: S_{t+Δt} = S_t + μ S_t Δt + σ S_t sqrt(Δt) Z
         -> generic, but may go **negative** with coarse Δt / large σ

Units:
- μ is per year, σ is per sqrt(year), Δt is in years (e.g., 1/252 for daily)
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt

class StochasticProcess:
    """
    Minimal GBM process with two step methods:
      - 'exact'  (recommended for GBM)
      - 'euler'  (generic SDE discretization; can go negative)

    Parameters
    ----------
    drift : float
        μ — annualized drift of the **level** S_t.
    volatility : float
        σ — annualized volatility (per sqrt(year)).
    delta_t : float
        Δt in **years** (daily ≈ 1/252).
    initial_asset_price : float
        S_0 > 0.
    method : {'exact','euler'}
        Update rule for each time_step().
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        drift: float,
        volatility: float,
        delta_t: float,
        initial_asset_price: float,
        method: str = "exact",
        seed: int | None = 42,
    ):
        self.drift = float(drift)
        self.volatility = float(volatility)
        self.delta_t = float(delta_t)
        self.current_asset_price = float(initial_asset_price)
        self.asset_prices = [self.current_asset_price]
        self.method = method.lower()
        self.rng = np.random.default_rng(seed)

        if self.current_asset_price <= 0:
            raise ValueError("initial_asset_price must be > 0")
        if self.method not in {"exact", "euler"}:
            raise ValueError("method must be 'exact' or 'euler'")

    def time_step(self):
        """
        Advance one time step of length Δt using the chosen method.

        References:
        - Exact step (Step 3): ensures S_{t+Δt} > 0 and matches one-step log-return
          mean/variance: E[Δ ln S] = (μ − ½σ²)Δt, Var[Δ ln S] = σ²Δt.
        - Euler–Maruyama: generic discretization; may produce negative prices.
        """
        Z = self.rng.standard_normal()                # Z ~ N(0,1)
        dt = self.delta_t
        S = self.current_asset_price
        mu = self.drift
        sigma = self.volatility

        if self.method == "exact":
            # GBM exact multiplicative update (recommended)
            drift_term = (mu - 0.5 * sigma**2) * dt
            diff_term  = sigma * math.sqrt(dt) * Z
            S_next = S * math.exp(drift_term + diff_term)
        else:
            # Euler step (warning: can go negative if shock is large)
            dW = math.sqrt(dt) * Z
            dS = mu * S * dt + sigma * S * dW
            S_next = S + dS

        self.asset_prices.append(S_next)
        self.current_asset_price = S_next

    def simulate(self, years: float) -> np.ndarray:
        """
        Simulate one full path for a given horizon in years.

        Notes:
        - n_steps = round(years / Δt). If years is not a multiple of Δt,
          we ignore the tiny remainder for simplicity.
        """
        n_steps = int(round(years / self.delta_t))
        for _ in range(n_steps):
            self.time_step()
        return np.asarray(self.asset_prices, dtype=float)

# --------------------------
# Example usage (single run)
# --------------------------
if __name__ == "__main__":
    # Params (annualized): feel free to change
    mu = 0.20                 # 20% annual drift
    sigma = 0.30              # 30% annual vol
    dt = 1/252                # daily step (trading year)
    S0 = 300.0
    years = 1.0

    # Simulate many paths to visualize the "fan" (Step 3 intuition)
    n_paths = 100
    paths = []
    for i in range(n_paths):
        proc = StochasticProcess(
            drift=mu, volatility=sigma, delta_t=dt, initial_asset_price=S0,
            method="exact", seed=1234 + i  # different seed per path
        )
        path = proc.simulate(years=years)
        paths.append(path)

    paths = np.vstack(paths)  # shape: (n_paths, n_steps+1)

    # Time axis in trading days
    t = np.arange(paths.shape[1])

    # Plot a handful of paths to avoid over-plotting
    plt.figure(figsize=(9, 4))
    for i in range(min(50, n_paths)):
        plt.plot(t, paths[i], alpha=0.25)
    plt.title("Geometric Brownian Motion — sample paths (exact step)")
    plt.xlabel("days")
    plt.ylabel("price")
    plt.show()

    # Optional: show 10%/50%/90% quantiles over time (mini fan chart)
    qlo, qmed, qhi = np.quantile(paths, [0.10, 0.50, 0.90], axis=0)
    plt.figure(figsize=(9, 4))
    plt.plot(t, qmed, label="median", linewidth=1.8)
    plt.plot(t, qlo,  label="10% quantile", linewidth=1)
    plt.plot(t, qhi,  label="90% quantile", linewidth=1)
    plt.title("GBM quantile fan (theoretical via simulation)")
    plt.xlabel("days")
    plt.ylabel("price")
    plt.legend()
    plt.show()
