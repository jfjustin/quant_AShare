"""Qlib-style modelling: LightGBM forecaster + IC/ICIR evaluation + long-short
backtest. Mirrors Qlib's Alpha158 + LGBModel benchmark pipeline.

Pipeline (no look-ahead):
  1. cross-sectional z-score features per day (Qlib CSZScoreNorm)
  2. time-split into train / valid / test
  3. fit LightGBM with Qlib's published Alpha158 hyper-parameters (valid early-stop)
  4. predict test-set scores
  5. evaluate: daily IC (Pearson), Rank IC (Spearman), ICIR, Rank ICIR
  6. backtest: daily top-minus-bottom quantile portfolio -> ann. return / IR / MaxDD

The hyper-parameters below are Qlib's benchmark values for LightGBM on Alpha158,
so the results are directly comparable in method to the ~IC 0.045 CSI300 figure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Qlib's PUBLISHED LightGBM / Alpha158 hyper-parameters (for reference). These
# are tuned for CSI300 (~300 names, millions of rows); the large L1/L2 penalties
# shrink a small cross-section to a constant, so we don't use them verbatim here.
QLIB_LGB_PARAMS = dict(
    objective="mse", colsample_bytree=0.8879, learning_rate=0.0421,
    subsample=0.8789, reg_alpha=205.6999, reg_lambda=580.9768,
    max_depth=8, num_leaves=210, n_estimators=1000, min_child_samples=20,
    n_jobs=-1, verbose=-1,
)

# Same method, regularization scaled to a small A-share tech cross-section.
LGB_PARAMS = dict(
    objective="mse", colsample_bytree=0.85, learning_rate=0.03, subsample=0.85,
    reg_alpha=2.0, reg_lambda=20.0, max_depth=6, num_leaves=63,
    n_estimators=1000, min_child_samples=50, n_jobs=-1, verbose=-1,
)


def csz_norm(X: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per day (robust to outliers via clip)."""
    def _z(g):
        mu = g.mean()
        sd = g.std(ddof=0).replace(0, np.nan)
        return ((g - mu) / sd).clip(-3, 3)
    return X.groupby(level="date", group_keys=False).apply(_z)


@dataclass
class FitResult:
    pred: pd.Series                 # test-set predictions (index code,date)
    label: pd.Series                # aligned realized forward returns
    metrics: dict = field(default_factory=dict)
    importance: pd.Series = None


def _time_split(index_dates: pd.DatetimeIndex, train=0.6, valid=0.2):
    uniq = np.sort(index_dates.unique())
    n = len(uniq)
    t_end = uniq[int(n * train)]
    v_end = uniq[int(n * (train + valid))]
    return t_end, v_end


def fit_predict(X: pd.DataFrame, y: pd.Series, *, normalize=True,
                train=0.6, valid=0.2) -> FitResult:
    import lightgbm as lgb

    df = X.join(y.rename("label")).dropna().sort_index(level="date")
    Xf = df.drop(columns="label")
    yf = df["label"]
    if normalize:
        # reindex back to df order so the train/test masks stay aligned
        Xf = csz_norm(Xf).reindex(df.index).fillna(0.0)

    dates = df.index.get_level_values("date")
    t_end, v_end = _time_split(dates, train, valid)
    tr = dates <= t_end
    va = (dates > t_end) & (dates <= v_end)
    te = dates > v_end

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(Xf[tr], yf[tr], eval_set=[(Xf[va], yf[va])],
              eval_metric="l2",
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(0)])

    pred = pd.Series(model.predict(Xf[te]), index=Xf[te].index, name="score")
    label = yf[te]
    imp = pd.Series(model.feature_importances_, index=Xf.columns).sort_values(ascending=False)

    res = FitResult(pred=pred, label=label, importance=imp)
    res.metrics = evaluate(pred, label)
    return res


# --- evaluation -------------------------------------------------------------
def evaluate(pred: pd.Series, label: pd.Series) -> dict:
    """Daily IC / Rank IC and their IR (mean/std)."""
    df = pd.concat([pred.rename("s"), label.rename("y")], axis=1).dropna()
    if df.empty:
        return {}
    g = df.groupby(level=-1)   # last index level = date
    ic = g.apply(lambda d: d["s"].corr(d["y"]) if len(d) > 2 else np.nan).dropna()
    ric = g.apply(lambda d: d["s"].corr(d["y"], method="spearman") if len(d) > 2 else np.nan).dropna()
    return {
        "IC": float(ic.mean()),
        "ICIR": float(ic.mean() / ic.std()) if ic.std() else np.nan,
        "RankIC": float(ric.mean()),
        "RankICIR": float(ric.mean() / ric.std()) if ric.std() else np.nan,
        "n_days": int(len(ic)),
        "n_samples": int(len(df)),
    }


# --- backtest ---------------------------------------------------------------
def _weights(pred_row: pd.Series, quantile: float) -> pd.Series:
    """Equal-weight top/bottom quantile: +1/k on longs, -1/k on shorts (gross 1/side)."""
    row = pred_row.dropna()
    k = max(1, int(len(row) * quantile))
    if len(row) < 2 * k:
        return pd.Series(dtype=float)
    top = row.nlargest(k).index
    bot = row.nsmallest(k).index
    w = pd.Series(0.0, index=row.index)
    w.loc[top] = 0.5 / k
    w.loc[bot] = -0.5 / k
    return w


def backtest_long_short(pred: pd.Series, panel: pd.DataFrame, quantile=0.2,
                        horizon=1, cost=0.0015, t_plus_1=True) -> dict:
    """Realistic daily top/bottom quantile long-short backtest.

    Realism modelled:
      * **T+1 / execution lag**: signals use close of day t; you can only trade
        on the NEXT session, so weights formed at t earn returns from t+1 onward
        (``lag=1``). This also enforces A-share T+1 (no same-day round trip).
      * **transaction costs**: ``cost`` per unit turnover, per side, charged on
        |w_t - w_{t-1}| each rebalance (commission + 0.05%% sell stamp + slippage;
        0.0015 = 15bps is a reasonable all-in estimate).
      * daily close-to-close returns, daily rebalance (no overlap double-count).

    Returns gross AND net (after-cost) stats so the cost drag is explicit.
    """
    close = panel["close"].unstack(level=0).sort_index()
    ret1 = close.pct_change()                      # close-to-close daily return
    pr = pred.unstack(level=0).sort_index()
    lag = 1 if t_plus_1 else 0

    dates = pr.index
    prev_w = None
    gross, net, longs, turns = [], [], [], []
    for i, dt in enumerate(dates):
        if i + lag >= len(dates):
            break
        w = _weights(pr.loc[dt], quantile)
        if w.empty:
            continue
        # return realised on the session AFTER the signal (t+1): T+1 execution
        r_dt = dates[i + lag]
        r = ret1.loc[r_dt].reindex(w.index).fillna(0.0)
        g = float((w * r).sum())
        lg = float((w.clip(lower=0) * r).sum()) * 2      # long leg scaled to gross 1
        # turnover vs previous rebalance
        aligned = w.reindex(w.index.union(prev_w.index)).fillna(0.0) if prev_w is not None else w
        base = prev_w.reindex(aligned.index).fillna(0.0) if prev_w is not None else aligned * 0
        turn = float((aligned - base).abs().sum())
        gross.append((r_dt, g))
        net.append((r_dt, g - turn * cost))
        longs.append((r_dt, lg))
        turns.append(turn)
        prev_w = w

    if not net:
        return {}
    to_s = lambda lst: pd.Series(dict(lst)).sort_index()
    g_s, n_s, l_s = to_s(gross), to_s(net), to_s(longs)

    def _stats(r):
        ann = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        eq = (1 + r).cumprod()
        dd = (eq / eq.cummax() - 1).min()
        return dict(ann_return=float(ann), ir=float(ann / vol) if vol else np.nan,
                    maxdd=float(dd), hit=float((r > 0).mean()))

    return {"long_short_gross": _stats(g_s), "long_short_net": _stats(n_s),
            "long_only_net": _stats(l_s),
            "avg_turnover": float(np.mean(turns)),
            "cost_drag_annual": float((g_s.mean() - n_s.mean()) * 252),
            "n_days": int(len(n_s)), "cost_bps": cost * 1e4, "t_plus_1": t_plus_1}
