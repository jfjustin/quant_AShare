"""End-to-end daily strategy orchestrator.

    login Choice -> build universe -> pull history+snapshot -> compute factors
    -> scrape live sentiment -> build alpha -> detect regime -> size book -> emit
    target weights + an order blotter vs the prior book.

This is the single call the daily runner and the live loop share.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from .config import Config
from .choice_client import ChoiceClient
from .data import MarketData
from .universe import build_universe
from .signals import build_alpha
from .regime import detect_regime, Regime
from .portfolio import construct
from .sentiment.sentiment_engine import compute_sentiment

log = logging.getLogger("ashare.strategy")


@dataclass
class TargetBook:
    date: str
    regime: Regime
    weights: pd.Series
    alpha_table: pd.DataFrame
    sentiment: pd.DataFrame


def _ymd(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def run_strategy(cfg: Config, tradedate: str | None = None,
                 prior: pd.Series | None = None,
                 use_sentiment: bool = True) -> TargetBook:
    today = datetime.now()
    tradedate = tradedate or _ymd(today)
    start_hist = _ymd(today - timedelta(days=140))

    client = ChoiceClient(**{
        "username": cfg.get("choice.username", ""),
        "password": cfg.get("choice.password", ""),
        "start_options": cfg.get("choice.start_options", "ForceLogin=1"),
    })
    client.login()
    md = MarketData(choice=client)

    # 1) universe
    uni = build_universe(cfg, md, tradedate.replace("-", ""))
    codes = list(uni.index)
    log.info("universe: %d names (live=%s)", len(codes), client.is_live)

    # 2) history + snapshot (Choice); realtime fallback keeps 'close' populated
    if client.is_live and codes:
        hist = md.history(codes, start_hist, tradedate)
        snap = md.snapshot(codes, tradedate.replace("-", ""))
    else:
        hist, snap = pd.DataFrame(), pd.DataFrame()

    # 3) meta for neutralisation
    meta = uni.copy()
    if "mktcap" not in meta.columns and "mktcap" in snap.columns:
        meta = meta.join(snap[["mktcap"]])

    # 4) sentiment (live scrape) — needs {code: name}
    sent = pd.DataFrame()
    if use_sentiment and codes:
        names = {}
        rt = md.realtime(codes)
        for c in codes:
            names[c] = (rt.at[c, "name"] if (not rt.empty and c in rt.index
                        and "name" in rt.columns) else c)
        scfg = cfg.get("sentiment", {})
        try:
            sent = compute_sentiment(
                names,
                per_stock=scfg.get("news", {}).get("per_stock", 8),
                concurrency=scfg.get("guba", {}).get("concurrency", 12),
                now=today)
        except Exception as e:
            log.warning("sentiment scrape failed: %s", e)

    # 5) alpha
    if hist.empty:
        # offline: rank purely by live sentiment_delta so we still emit a book
        if not sent.empty:
            z = sent.copy()
            z["alpha"] = z.get("sentiment_delta", pd.Series(0.0, index=z.index))
        else:
            z = pd.DataFrame({"alpha": pd.Series(0.0, index=codes)})
    else:
        z = build_alpha(hist, snap, sent, meta,
                        weights=cfg.get("signal.weights", {}),
                        winsor=cfg.get("signal.winsorize", 0.025),
                        neutralize=cfg.get("signal.neutralize", []))

    # 6) regime gate (CSI 300)
    regime = Regime("calm_up", 0.0, 0.5, 0.0, 1.0)
    if client.is_live:
        try:
            idx = md.history([cfg.get("regime.index_code", "000300.SH")],
                             start_hist, tradedate)
            ic = idx["close"].droplevel(0) if isinstance(idx.index, pd.MultiIndex) else idx["close"]
            regime = detect_regime(ic,
                                   cfg.get("regime.vol_window", 20),
                                   cfg.get("regime.vol_high_pct", 0.80),
                                   cfg.get("regime.trend_window", 60))
        except Exception as e:
            log.warning("regime detect failed: %s", e)

    # 7) construct
    pcfg = cfg.get("portfolio", {})
    w = construct(z["alpha"], meta, book=pcfg.get("book", "long_short"),
                  gross_target=pcfg.get("gross_target", 1.0),
                  exposure_mult=regime.exposure,
                  max_names=pcfg.get("max_names", 40),
                  max_weight=pcfg.get("max_weight", 0.05),
                  theme_cap=pcfg.get("theme_cap", 0.30),
                  prior=prior,
                  turnover_penalty=pcfg.get("turnover_penalty", 0.0015))

    client.close()
    return TargetBook(tradedate, regime, w, z, sent)


def blotter(target: pd.Series, prior: pd.Series | None) -> pd.DataFrame:
    """Order list = target - prior, T+1 aware (informational)."""
    prior = prior if prior is not None else pd.Series(dtype=float)
    idx = target.index.union(prior.index)
    t = target.reindex(idx).fillna(0.0)
    p = prior.reindex(idx).fillna(0.0)
    delta = (t - p)
    orders = delta[delta.abs() > 1e-4].sort_values()
    return pd.DataFrame({"target_w": t.reindex(orders.index),
                         "prior_w": p.reindex(orders.index),
                         "delta_w": orders,
                         "side": ["BUY" if d > 0 else "SELL" for d in orders]})
