#!/usr/bin/env python3
"""Standalone LIVE sentiment scan — works right now with NO Choice login.

    python scripts/run_sentiment.py 600519.SH 300750.SZ 688981.SH
    python scripts/run_sentiment.py --seed        # scan the built-in tech seed

Pulls live news + Guba attention + main-force flow and prints a ranked table.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.data import MarketData
from quant.choice_client import ChoiceClient
from quant.universe import SEED
from quant.sentiment.sentiment_engine import compute_sentiment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("codes", nargs="*")
    ap.add_argument("--seed", action="store_true", help="use built-in tech seed universe")
    args = ap.parse_args()

    codes = list(args.codes)
    if args.seed or not codes:
        codes = sorted({c for lst in SEED.values() for c in lst})

    # get names via free realtime API (no Choice needed)
    md = MarketData(choice=ChoiceClient())
    rt = md.realtime(codes)
    names = {c: (rt.at[c, "name"] if (not rt.empty and c in rt.index) else c) for c in codes}

    print(f"Scanning live sentiment for {len(codes)} names ...")
    df = compute_sentiment(names, per_stock=8, concurrency=12, update_cache=True)
    df["name"] = [names.get(c, c) for c in df.index]
    cols = ["name", "sentiment", "sentiment_delta", "attention",
            "mainforce_flow", "divergence", "news_n"]
    df = df[cols].sort_values("sentiment", ascending=False)

    print("\n" + df.to_string(float_format=lambda x: f"{x:+.3f}"))
    print("\nTop bullish inflection (sentiment_delta):")
    print(df.sort_values("sentiment_delta", ascending=False).head(5)[["name", "sentiment_delta"]].to_string())
    print("\nEuphoria-divergence shorts (hot attention, weak flow/news):")
    print(df.sort_values("divergence", ascending=False).head(5)[["name", "divergence", "attention"]].to_string())


if __name__ == "__main__":
    main()
