"""Cross-sectional factor library, tuned for A-share tech microstructure.

Every factor here is a *cross-sectional* signal: on a given trade date it scores
each name in the universe relative to its peers. Higher = more attractive to
LONG. Signs are chosen so the raw factor already points the profitable way, then
``composite`` z-scores, sign-weights and sums them.

Design rationale (why these and not textbook value/momentum)
------------------------------------------------------------
A-share is ~2/3 retail turnover, T+1, with 10–20% daily limits. Empirically:

* **Short-term reversal dominates.** Retail chases 1–5d moves; they mean-revert
  hard. This is the single strongest, most robust anomaly on the exchange, so it
  gets the largest weight.
* **"Lottery" demand is punished.** High idiosyncratic vol + high MAX-return
  names are over-bought by gamblers and underperform → we short them.
* **Turnover spikes = exhaustion.** A blow-off in turnover marks local tops.
* Classic 6–12m **price momentum is weak/unstable** here, so residual momentum
  is available but off by default (weight 0 in config).

Factors that need fundamentals/flows (northbound, EPS revision) are computed from
the Choice snapshot when live and quietly skipped (NaN) offline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --- primitive transforms ---------------------------------------------------
def winsorize(s: pd.Series, p: float = 0.025) -> pd.Series:
    if s.dropna().empty:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def zscore(s: pd.Series) -> pd.Series:
    m, sd = s.mean(), s.std(ddof=0)
    if not sd or sd != sd:
        return pd.Series(0.0, index=s.index)
    return (s - m) / sd


def neutralize(factor: pd.Series, by: pd.DataFrame) -> pd.Series:
    """Regress the factor on group dummies + numeric controls, return residual.

    ``by`` columns may be categorical (e.g. 'theme') or numeric (e.g. 'logmktcap').
    This removes theme/size tilts so the alpha is a *within-peer* bet.
    """
    y = factor.copy()
    idx = y.dropna().index.intersection(by.dropna(how="all").index)
    if len(idx) < 5:
        return y
    y = y.loc[idx]
    X_parts = [pd.Series(1.0, index=idx, name="const")]
    for col in by.columns:
        v = by.loc[idx, col]
        if v.dtype == object or str(v.dtype).startswith("category"):
            X_parts.append(pd.get_dummies(v, prefix=col, drop_first=True).astype(float))
        else:
            X_parts.append(((v - v.mean()) / (v.std(ddof=0) or 1.0)).rename(col))
    X = pd.concat(X_parts, axis=1).fillna(0.0)
    beta, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
    resid = y.values - X.values @ beta
    out = factor.copy() * np.nan
    out.loc[idx] = resid
    return out


# --- individual factors -----------------------------------------------------
def st_reversal(hist: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Short-term reversal: NEGATIVE of the last-`lookback`-day return.

    Oversold (fell) -> high score -> we lean long into the bounce.
    `hist` is MultiIndex(code, date) with a 'close' column.
    """
    def _f(g):
        c = g["close"].dropna()
        if len(c) <= lookback:
            return np.nan
        return -(c.iloc[-1] / c.iloc[-lookback - 1] - 1.0)
    return hist.groupby(level=0).apply(_f).rename("st_reversal")


def ivol_lottery(hist: pd.DataFrame, window: int = 20) -> pd.Series:
    """Idiosyncratic-vol / lottery factor: NEGATIVE of recent daily-return vol.

    High-vol lottery names are over-owned by retail -> short them -> negative sign
    means high vol gets a low (short) score.
    """
    def _f(g):
        c = g["close"].dropna()
        if len(c) <= window:
            return np.nan
        r = c.pct_change().dropna().iloc[-window:]
        return -float(r.std(ddof=0))
    return hist.groupby(level=0).apply(_f).rename("ivol_lottery")


def turnover_heat(hist: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.Series:
    """Turnover exhaustion: NEGATIVE of (recent turnover / baseline turnover).

    A spike (fast >> slow) marks over-heating -> fade it.
    """
    if "turnover" not in hist.columns:
        return pd.Series(dtype=float, name="turnover_heat")

    def _f(g):
        t = g["turnover"].dropna()
        if len(t) <= slow:
            return np.nan
        base = t.iloc[-slow:].mean()
        if not base:
            return np.nan
        return -float(t.iloc[-fast:].mean() / base)
    return hist.groupby(level=0).apply(_f).rename("turnover_heat")


def residual_mom(hist: pd.DataFrame, window: int = 60, skip: int = 5) -> pd.Series:
    """Medium-term momentum with a short-term-reversal skip window."""
    def _f(g):
        c = g["close"].dropna()
        if len(c) <= window + skip:
            return np.nan
        return float(c.iloc[-skip - 1] / c.iloc[-window - skip - 1] - 1.0)
    return hist.groupby(level=0).apply(_f).rename("residual_mom")


def northbound_flow(snap: pd.DataFrame) -> pd.Series:
    """Institutional/northbound accumulation proxy (higher holding value = better)."""
    if "north_value" not in snap.columns:
        return pd.Series(dtype=float, name="northbound_flow")
    return snap["north_value"].astype(float).rename("northbound_flow")


def eps_revision(snap: pd.DataFrame) -> pd.Series:
    """Analyst EPS/earnings-growth upgrade momentum (consensus growth estimate)."""
    for col in ("eps_growth_est", "analyst_rating"):
        if col in snap.columns:
            return snap[col].astype(float).rename("eps_revision")
    return pd.Series(dtype=float, name="eps_revision")


# --- composite --------------------------------------------------------------
FACTOR_FNS = {
    "st_reversal": lambda h, s: st_reversal(h),
    "ivol_lottery": lambda h, s: ivol_lottery(h),
    "turnover_heat": lambda h, s: turnover_heat(h),
    "residual_mom": lambda h, s: residual_mom(h),
    "northbound_flow": lambda h, s: northbound_flow(s),
    "eps_revision": lambda h, s: eps_revision(s),
}


def compute_factors(hist: pd.DataFrame, snap: pd.DataFrame,
                    which: list[str]) -> pd.DataFrame:
    cols = {}
    for name in which:
        fn = FACTOR_FNS.get(name)
        if fn is None:
            continue
        try:
            cols[name] = fn(hist, snap)
        except Exception:
            continue
    return pd.DataFrame(cols)
