"""Market-regime detector -> gross-exposure multiplier.

The book is built on tech names that whip around violently. Being right on the
cross-section but wrong on *how much to be invested* is how good stock-pickers
blow up in A-share. So we gate gross exposure by a simple, robust 2x2 regime on
the CSI 300:

              trend UP           trend DOWN
  calm vol    1.0  (full risk)   0.7
  storm vol   0.6  (de-risk)     0.3  (capital preservation)

* **vol** = 20d realised vol percentile over a trailing window; >80th pct = storm.
* **trend** = sign of the 60d close slope.

Rationale: reversal/sentiment alpha keeps working in calm markets; in a
high-vol *down* regime, correlations spike, limit-downs cluster, and single-name
signals get swamped by beta — so we cut to a third of normal gross and wait.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Regime:
    label: str          # calm_up | calm_down | storm_up | storm_down
    vol: float
    vol_pct: float
    trend: float
    exposure: float     # gross multiplier in [0,1]


_EXPOSURE = {"calm_up": 1.0, "calm_down": 0.7, "storm_up": 0.6, "storm_down": 0.3}


def detect_regime(index_close: pd.Series, vol_window: int = 20,
                  vol_high_pct: float = 0.80, trend_window: int = 60,
                  pct_lookback: int = 250) -> Regime:
    """index_close: date-indexed close of the market index (e.g. CSI 300)."""
    c = index_close.dropna()
    if len(c) < max(vol_window, trend_window) + 5:
        return Regime("calm_up", 0.0, 0.5, 0.0, 1.0)

    ret = c.pct_change()
    rv = ret.rolling(vol_window).std() * np.sqrt(252)
    cur_vol = float(rv.iloc[-1])
    hist = rv.dropna().iloc[-pct_lookback:]
    vol_pct = float((hist <= cur_vol).mean()) if len(hist) else 0.5

    window = c.iloc[-trend_window:]
    x = np.arange(len(window))
    slope = float(np.polyfit(x, window.values, 1)[0])
    trend_up = slope >= 0

    storm = vol_pct >= vol_high_pct
    label = f"{'storm' if storm else 'calm'}_{'up' if trend_up else 'down'}"
    return Regime(label, cur_vol, round(vol_pct, 3), round(slope, 4),
                  _EXPOSURE[label])
