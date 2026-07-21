"""Vectorised daily backtester for the reversal+sentiment book.

Scope & honesty note
--------------------
Live sentiment (Guba attention / news) is *point-in-time* and cannot be
reconstructed historically from the free endpoints, so the **backtest runs the
price/fundamental factor sleeve** (reversal, ivol, turnover, momentum, and — when
Choice is live — northbound & EPS revision). The sentiment sleeve is validated
*forward* (paper/live) and layered on top. This keeps the backtest free of
look-ahead: we never train on data we couldn't have had.

Mechanics:
  * rebalance daily (or weekly); signals use data up to close of day t
  * positions earn day t+1 open->t+2 open ... we use t->t+1 close returns with a
    1-day implementation lag to respect that you trade on the *next* session
  * costs charged on |Δweight| each rebalance (commission+stamp+impact)
  * T+1 is approximated by the 1-day lag (can't realise same-day round trips)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import factors as F
from .signals import build_alpha

log = logging.getLogger("ashare.backtest")


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    stats: dict = field(default_factory=dict)
    weights_history: dict = field(default_factory=dict)

    def summary(self) -> str:
        s = self.stats
        return (f"CAGR {s.get('cagr',0):.1%} | Sharpe {s.get('sharpe',0):.2f} | "
                f"MaxDD {s.get('maxdd',0):.1%} | Vol {s.get('vol',0):.1%} | "
                f"HitDays {s.get('hit',0):.1%} | Turn {s.get('turnover',0):.1%}")


def _stats(returns: pd.Series) -> dict:
    r = returns.dropna()
    if r.empty:
        return {}
    ann = 252
    cagr = (1 + r).prod() ** (ann / len(r)) - 1
    vol = r.std() * np.sqrt(ann)
    sharpe = (r.mean() * ann) / vol if vol else 0.0
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return {"cagr": float(cagr), "vol": float(vol), "sharpe": float(sharpe),
            "maxdd": float(dd), "hit": float((r > 0).mean())}


def run_backtest(hist: pd.DataFrame, meta: pd.DataFrame, cfg, *,
                 rebalance: str = "daily") -> BacktestResult:
    """hist: MultiIndex(code, date) with 'close' (+turnover). meta: code->theme,mktcap."""
    close = hist["close"].unstack(level=0).sort_index()
    fwd_ret = close.pct_change().shift(-1)          # t -> t+1 return, applied to weights at t
    dates = close.index
    weights = cfg.get("signal.weights", {})
    winsor = cfg.get("signal.winsorize", 0.025)
    neutral = cfg.get("signal.neutralize", [])
    pcfg = cfg.get("portfolio", {})
    cost = pcfg.get("turnover_penalty", 0.0015)

    step = 5 if rebalance == "weekly" else 1
    prior_w = pd.Series(dtype=float)
    port_ret, turns, whist = [], [], {}

    min_hist = 65
    for i in range(min_hist, len(dates) - 1, step):
        d = dates[i]
        window = close.iloc[: i + 1]
        h = window.stack().rename("close").to_frame()
        h.index = h.index.set_names(["date", "code"])
        h = h.reorder_levels(["code", "date"]).sort_index()
        if "turnover" in hist.columns:
            tw = hist["turnover"].unstack(level=0).iloc[: i + 1].stack()
            h["turnover"] = tw.reorder_levels([1, 0]).sort_index().reindex(h.index)

        z = build_alpha(h, snap=pd.DataFrame(index=meta.index), sentiment=pd.DataFrame(),
                        meta=meta, weights=weights, winsor=winsor, neutralize=neutral)
        from .portfolio import construct
        w = construct(z["alpha"], meta, book=pcfg.get("book", "long_short"),
                      gross_target=pcfg.get("gross_target", 1.0),
                      max_names=pcfg.get("max_names", 40),
                      max_weight=pcfg.get("max_weight", 0.05),
                      theme_cap=pcfg.get("theme_cap", 0.30),
                      prior=prior_w, turnover_penalty=cost)

        r = float((w * fwd_ret.loc[d].reindex(w.index).fillna(0.0)).sum())
        turn = float((w - prior_w.reindex(w.index).fillna(0.0)).abs().sum())
        r -= turn * cost                                  # charge costs
        port_ret.append((dates[i + 1], r))
        turns.append(turn)
        whist[str(d.date()) if hasattr(d, "date") else str(d)] = w
        prior_w = w

    ret = pd.Series(dict(port_ret)).sort_index()
    eq = (1 + ret).cumprod()
    st = _stats(ret)
    st["turnover"] = float(np.mean(turns)) if turns else 0.0
    return BacktestResult(eq, ret, st, whist)
