#!/usr/bin/env python3
"""Backtest the price/fundamental factor sleeve over a date range.

    python scripts/run_backtest.py                      # uses config dates
    python scripts/run_backtest.py --start 2024-01-01 --end 2025-06-30

Requires a live Choice login for real history. Without it, generates a synthetic
mean-reverting price panel so you can validate the *plumbing* end-to-end (the
printed stats are then meaningless — they only prove the engine runs).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.config import Config
from quant.choice_client import ChoiceClient
from quant.data import MarketData
from quant.universe import build_universe, SEED
from quant.backtest import run_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _synthetic_panel(codes, start, end):
    """Mean-reverting synthetic closes to smoke-test the engine offline."""
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(42)
    data = {}
    for c in codes:
        n = len(dates)
        shocks = rng.normal(0, 0.025, n)
        px = [10.0]
        for t in range(1, n):
            mr = -0.15 * (np.log(px[-1] / 10.0))          # pull back to 10
            px.append(px[-1] * (1 + mr * 0.1 + shocks[t]))
        s = pd.DataFrame({"close": px, "turnover": rng.uniform(1, 8, n)}, index=dates)
        s.index.name = "date"
        s["code"] = c
        data[c] = s.set_index("code", append=True)
    df = pd.concat(data.values()).reorder_levels(["code", "date"]).sort_index()
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    cfg = Config.load()
    start = args.start or cfg.get("backtest.start", "2023-01-01")
    end = args.end or cfg.get("backtest.end", "2025-12-31")

    client = ChoiceClient(username=cfg.get("choice.username", ""),
                          password=cfg.get("choice.password", ""),
                          start_options=cfg.get("choice.start_options", "ForceLogin=1"))
    client.login()
    md = MarketData(choice=client)

    if client.is_live:
        uni = build_universe(cfg, md, end.replace("-", ""))
        codes = list(uni.index)
        hist = md.history(codes, start, end)
        meta = uni[["theme"]].copy()
        if "mktcap" in uni.columns:
            meta["mktcap"] = uni["mktcap"]
    else:
        print("!! Choice OFFLINE — running SYNTHETIC smoke test (stats meaningless).")
        codes = sorted({c for lst in SEED.values() for c in lst})
        hist = _synthetic_panel(codes, start, end)
        theme_of = {c: th for th, lst in SEED.items() for c in lst}
        meta = pd.DataFrame({"theme": [theme_of[c] for c in codes],
                             "mktcap": [5e10] * len(codes)}, index=codes)

    res = run_backtest(hist, meta, cfg, rebalance=cfg.get("backtest.rebalance", "daily"))
    print("\n" + "=" * 60)
    print(" BACKTEST", start, "->", end, "| live=", client.is_live)
    print(" " + res.summary())
    print("=" * 60)
    client.close()


if __name__ == "__main__":
    main()
