"""Portfolio construction: alpha -> tradeable target weights.

Constraints baked in for A-share reality:
  * long/short OR long-only (no borrow -> long_only)
  * gross scaled by the regime multiplier (quant/regime.py)
  * single-name cap and per-theme gross cap (keep it diversified across chains)
  * T+1: names bought today cannot be sold today (handled by the backtester/live
    layer that owns yesterday's book; construction just emits targets)
  * turnover-aware: we shrink toward the prior book to avoid churning away edge
    in costs (stamp duty 0.05% sell-side + commission + impact).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _cap_theme(weights: pd.Series, theme: pd.Series, theme_cap: float) -> pd.Series:
    w = weights.copy()
    gross = w.abs().sum()
    if gross == 0:
        return w
    for th, grp in theme.groupby(theme):
        idx = grp.index.intersection(w.index)
        tg = w.loc[idx].abs().sum()
        if tg > theme_cap * gross and tg > 0:
            w.loc[idx] *= (theme_cap * gross) / tg
    return w


def construct(alpha: pd.Series, meta: pd.DataFrame, *, book: str = "long_short",
              gross_target: float = 1.0, exposure_mult: float = 1.0,
              max_names: int = 40, max_weight: float = 0.05,
              theme_cap: float = 0.30, prior: pd.Series | None = None,
              turnover_penalty: float = 0.0015) -> pd.Series:
    """Return target weights (sum of abs ~ gross_target*exposure_mult)."""
    a = alpha.dropna().sort_values(ascending=False)
    if a.empty:
        return pd.Series(dtype=float)

    n = min(max_names, len(a))
    if book == "long_only":
        picks = a.head(n)
        raw = picks.clip(lower=0)
        if raw.sum() == 0:
            raw = pd.Series(1.0, index=picks.index)
        w = raw / raw.sum()
    else:
        half = max(1, n // 2)
        longs, shorts = a.head(half), a.tail(half)
        w = pd.concat([
            longs / longs.abs().sum() * 0.5 if longs.abs().sum() else longs,
            -shorts.rank(ascending=False).pipe(lambda r: r / r.sum()) * 0.5
            if len(shorts) else shorts,
        ])
        # simpler symmetric scheme: rank-weight both legs
        lw = np.linspace(1.0, 0.3, len(longs))
        sw = np.linspace(1.0, 0.3, len(shorts))
        w = pd.concat([
            pd.Series(lw / lw.sum() * 0.5, index=longs.index),
            pd.Series(-sw / sw.sum() * 0.5, index=shorts.index),
        ])

    gross = gross_target * exposure_mult
    w = w / w.abs().sum() * gross
    w = w.clip(-max_weight, max_weight)

    if "theme" in meta.columns:
        w = _cap_theme(w, meta["theme"].reindex(w.index), theme_cap)

    # turnover shrink toward prior book (cost-aware): move only if edge > cost
    if prior is not None and not prior.empty:
        aligned_prior = prior.reindex(w.index).fillna(0.0)
        delta = w - aligned_prior
        # dampen small rebalances whose expected edge < transaction cost
        keep = delta.abs() > turnover_penalty * 4
        w = aligned_prior + delta.where(keep, 0.0)

    # renormalise to gross after all caps
    if w.abs().sum() > 0:
        w = w / w.abs().sum() * gross
    return w.round(5)
