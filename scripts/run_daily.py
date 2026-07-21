#!/usr/bin/env python3
"""Daily driver: produce today's target book + order blotter.

    python scripts/run_daily.py                 # live-ish (Choice if creds set)
    python scripts/run_daily.py --no-sentiment  # factors only

Persists the target weights to data/book_YYYY-MM-DD.json so the next run can
compute turnover / T+1-aware orders against it.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.config import Config
from quant.strategy import run_strategy, blotter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
DATA = ROOT / "data"


def _load_prior() -> pd.Series | None:
    books = sorted(DATA.glob("book_*.json"))
    if not books:
        return None
    d = json.loads(books[-1].read_text())
    return pd.Series(d).astype(float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--no-sentiment", action="store_true")
    args = ap.parse_args()

    cfg = Config.load()
    prior = _load_prior()
    book = run_strategy(cfg, args.date, prior=prior, use_sentiment=not args.no_sentiment)

    print("\n" + "=" * 72)
    print(f" A-SHARE TECH BOOK — {book.date}")
    print(f" Regime: {book.regime.label}  (vol pct {book.regime.vol_pct}, "
          f"gross x{book.regime.exposure})")
    print("=" * 72)

    w = book.weights[book.weights.abs() > 1e-4].sort_values(ascending=False)
    if w.empty:
        print(" No positions (empty universe or offline w/o sentiment).")
    else:
        show = book.alpha_table.reindex(w.index)
        for code, wt in w.items():
            a = show.at[code, "alpha"] if "alpha" in show.columns and code in show.index else float("nan")
            side = "LONG " if wt > 0 else "SHORT"
            print(f"  {side} {code:11s}  w={wt:+.3f}  alpha={a:+.2f}")
        print(f"\n  gross={w.abs().sum():.2f}  net={w.sum():+.2f}  names={len(w)}")

    orders = blotter(book.weights, prior)
    if not orders.empty:
        print("\n  ORDER BLOTTER (vs prior book):")
        for code, row in orders.iterrows():
            print(f"    {row['side']:4s} {code:11s}  Δw={row['delta_w']:+.3f}")

    DATA.mkdir(exist_ok=True)
    out = DATA / f"book_{book.date}.json"
    out.write_text(json.dumps({k: float(v) for k, v in book.weights.items()}, indent=0))
    print(f"\n  saved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
