"""Combine factors + sentiment into a single cross-sectional alpha score.

Pipeline, per rebalance date:
  1. compute raw factors (quant/factors.py) and sentiment (quant/sentiment/*)
  2. winsorize each factor's tails
  3. cross-sectionally neutralise vs theme + log-mktcap (remove tilts)
  4. z-score each factor
  5. weight (signed) and sum  ->  composite alpha, higher = long
The composite is what portfolio.py turns into weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import factors as F


def _prep(raw: pd.DataFrame, by: pd.DataFrame, winsor: float,
          neutralize_cols: list[str]) -> pd.DataFrame:
    out = {}
    for col in raw.columns:
        s = F.winsorize(raw[col].astype(float), winsor)
        if neutralize_cols and not by.empty:
            s = F.neutralize(s, by[neutralize_cols])
        out[col] = F.zscore(s)
    return pd.DataFrame(out)


def build_alpha(hist: pd.DataFrame, snap: pd.DataFrame, sentiment: pd.DataFrame,
                meta: pd.DataFrame, weights: dict, winsor: float = 0.025,
                neutralize: list[str] | None = None) -> pd.DataFrame:
    """Return DataFrame with per-factor z-scores + 'alpha' composite.

    meta: index=code, columns include 'theme' and 'mktcap' (for neutralisation).
    sentiment: index=code, has 'sentiment_delta', 'divergence', 'mainforce_flow'.
    """
    which = [k for k, w in weights.items() if abs(w) > 0
             and k not in ("sentiment_delta",)]
    raw = F.compute_factors(hist, snap, which)

    # bring sentiment_delta in as a factor column
    if "sentiment_delta" in weights and not sentiment.empty:
        raw = raw.join(sentiment[["sentiment_delta"]], how="outer")

    # build neutralisation frame
    by = pd.DataFrame(index=raw.index)
    if meta is not None and not meta.empty:
        if "theme" in meta.columns:
            by["theme"] = meta["theme"].reindex(raw.index).astype("category")
        if "mktcap" in meta.columns:
            by["logmktcap"] = np.log(meta["mktcap"].reindex(raw.index).astype(float)
                                     .replace(0, np.nan))
    neutralize = neutralize or []
    neutralize = [c for c in neutralize if c in by.columns]

    z = _prep(raw.dropna(axis=1, how="all"), by, winsor, neutralize)

    alpha = pd.Series(0.0, index=z.index)
    for col in z.columns:
        alpha = alpha.add(weights.get(col, 0.0) * z[col].fillna(0.0), fill_value=0.0)
    z["alpha"] = alpha

    # attach divergence as an informational risk flag (not in composite)
    if not sentiment.empty and "divergence" in sentiment.columns:
        z["divergence"] = sentiment["divergence"].reindex(z.index)
    return z.sort_values("alpha", ascending=False)
