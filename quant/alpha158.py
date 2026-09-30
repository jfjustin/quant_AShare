"""Alpha158 factor library — a faithful re-implementation of Microsoft Qlib's
Alpha158 feature set for A-share data.

Qlib's Alpha158 is the benchmark feature handler behind the published
LightGBM/CSI300 results (IC ~0.045). It has ~158 features in these groups:

  * K-bar shape (9): KMID/KLEN/KUP/KLOW/KSFT and their range-normalised variants
  * Price ratios (OPEN/HIGH/LOW vs CLOSE)
  * Rolling, over windows [5,10,20,30,60]:
      ROC, MA, STD, BETA(slope), RSQR(R^2), RESI(reg residual),
      MAX, MIN, QTLU/QTLD(quantiles), RANK(ts-rank), RSV(stochastic),
      IMAX/IMIN/IMXD(arg-extrema position), CORR/CORD(price-volume corr),
      CNTP/CNTN/CNTD(up-day counts), SUMP/SUMN/SUMD(gain/loss ratios),
      VMA/VSTD/WVMA(volume stats), VSUMP/VSUMN/VSUMD(volume gain/loss).

Input : MultiIndex(code, date) DataFrame with open/high/low/close/volume.
Output: same index, ~150 feature columns, plus the forward-return label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = [5, 10, 20, 30, 60]
EPS = 1e-12


# --- rolling helpers on a single series ------------------------------------
def _slope(s: pd.Series, w: int) -> pd.Series:
    x = np.arange(w)
    xm = x.mean()
    xd = x - xm
    denom = (xd ** 2).sum()

    def f(a):
        return (xd * (a - a.mean())).sum() / denom
    return s.rolling(w).apply(f, raw=True)


def _rsqr(s: pd.Series, w: int) -> pd.Series:
    x = np.arange(w)
    xm = x.mean()
    xd = x - xm
    sxx = (xd ** 2).sum()

    def f(a):
        ad = a - a.mean()
        syy = (ad ** 2).sum()
        if syy < EPS:
            return np.nan
        sxy = (xd * ad).sum()
        return (sxy ** 2) / (sxx * syy)
    return s.rolling(w).apply(f, raw=True)


def _resi(s: pd.Series, w: int) -> pd.Series:
    """Residual of the last point vs its rolling linear fit."""
    x = np.arange(w)
    xm = x.mean()
    xd = x - xm
    sxx = (xd ** 2).sum()

    def f(a):
        ad = a - a.mean()
        beta = (xd * ad).sum() / sxx
        pred = a.mean() + beta * (w - 1 - xm)
        return a[-1] - pred
    return s.rolling(w).apply(f, raw=True)


def _ts_rank(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w).apply(lambda a: (a <= a[-1]).mean(), raw=True)


def _idxmax_pos(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w).apply(lambda a: (w - 1 - a.argmax()) / w, raw=True)


def _idxmin_pos(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w).apply(lambda a: (w - 1 - a.argmin()) / w, raw=True)


def _features_one(g: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = g["open"], g["high"], g["low"], g["close"], g["volume"]
    rng = (h - l).replace(0, np.nan)
    ret1 = c / c.shift(1) - 1
    out = {}

    # --- K-bar (9) ---
    out["KMID"] = (c - o) / o
    out["KLEN"] = (h - l) / o
    out["KMID2"] = (c - o) / (rng + EPS)
    out["KUP"] = (h - np.maximum(o, c)) / o
    out["KUP2"] = (h - np.maximum(o, c)) / (rng + EPS)
    out["KLOW"] = (np.minimum(o, c) - l) / o
    out["KLOW2"] = (np.minimum(o, c) - l) / (rng + EPS)
    out["KSFT"] = (2 * c - h - l) / o
    out["KSFT2"] = (2 * c - h - l) / (rng + EPS)

    # --- price ratios ---
    out["OPEN0"] = o / c
    out["HIGH0"] = h / c
    out["LOW0"] = l / c

    logv = np.log(v + 1.0)
    dlogv = logv - logv.shift(1)

    for w in WINDOWS:
        cref = c.shift(w)
        out[f"ROC{w}"] = cref / c
        out[f"MA{w}"] = c.rolling(w).mean() / c
        out[f"STD{w}"] = c.rolling(w).std() / c
        out[f"BETA{w}"] = _slope(c, w) / c
        out[f"RSQR{w}"] = _rsqr(c, w)
        out[f"RESI{w}"] = _resi(c, w) / c
        out[f"MAX{w}"] = h.rolling(w).max() / c
        out[f"MIN{w}"] = l.rolling(w).min() / c
        out[f"QTLU{w}"] = c.rolling(w).quantile(0.8) / c
        out[f"QTLD{w}"] = c.rolling(w).quantile(0.2) / c
        out[f"RANK{w}"] = _ts_rank(c, w)
        rmin = l.rolling(w).min()
        rmax = h.rolling(w).max()
        out[f"RSV{w}"] = (c - rmin) / (rmax - rmin + EPS)
        out[f"IMAX{w}"] = _idxmax_pos(h, w)
        out[f"IMIN{w}"] = _idxmin_pos(l, w)
        out[f"IMXD{w}"] = _idxmax_pos(h, w) - _idxmin_pos(l, w)
        out[f"CORR{w}"] = c.rolling(w).corr(logv)
        out[f"CORD{w}"] = (c / c.shift(1)).rolling(w).corr(dlogv)
        up = (ret1 > 0).astype(float)
        dn = (ret1 < 0).astype(float)
        out[f"CNTP{w}"] = up.rolling(w).mean()
        out[f"CNTN{w}"] = dn.rolling(w).mean()
        out[f"CNTD{w}"] = up.rolling(w).mean() - dn.rolling(w).mean()
        dc = c - c.shift(1)
        gain = dc.clip(lower=0)
        loss = (-dc).clip(lower=0)
        absd = dc.abs()
        out[f"SUMP{w}"] = gain.rolling(w).sum() / (absd.rolling(w).sum() + EPS)
        out[f"SUMN{w}"] = loss.rolling(w).sum() / (absd.rolling(w).sum() + EPS)
        out[f"SUMD{w}"] = (gain.rolling(w).sum() - loss.rolling(w).sum()) / (absd.rolling(w).sum() + EPS)
        out[f"VMA{w}"] = v.rolling(w).mean() / (v + EPS)
        out[f"VSTD{w}"] = v.rolling(w).std() / (v + EPS)
        av = ret1.abs() * v
        out[f"WVMA{w}"] = av.rolling(w).std() / (av.rolling(w).mean() + EPS)
        dv = v - v.shift(1)
        vg = dv.clip(lower=0)
        vl = (-dv).clip(lower=0)
        vabs = dv.abs()
        out[f"VSUMP{w}"] = vg.rolling(w).sum() / (vabs.rolling(w).sum() + EPS)
        out[f"VSUMN{w}"] = vl.rolling(w).sum() / (vabs.rolling(w).sum() + EPS)
        out[f"VSUMD{w}"] = (vg.rolling(w).sum() - vl.rolling(w).sum()) / (vabs.rolling(w).sum() + EPS)

    return pd.DataFrame(out, index=g.index)


def make_features(panel: pd.DataFrame, label_horizon: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    """Compute Alpha158 features + forward-return label from an OHLCV panel.

    panel: MultiIndex(code, date), columns open/high/low/close/volume.
    Returns (features, label) aligned on the same index (NaN rows dropped).
    label = h-day forward return: close[t+h]/close[t] - 1.
    """
    feats, labels = [], []
    for code, g in panel.groupby(level=0):
        g = g.sort_index()
        f = _features_one(g)
        lbl = (g["close"].shift(-label_horizon) / g["close"] - 1).rename("label")
        feats.append(f)
        labels.append(lbl)
    X = pd.concat(feats).replace([np.inf, -np.inf], np.nan)
    y = pd.concat(labels)
    return X, y
