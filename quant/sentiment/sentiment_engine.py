"""Fuse news + retail attention + smart-money flow into a per-stock sentiment.

Outputs, per code:
  sentiment       : level in ~[-1, 1]  (news polarity + flow, attention-weighted)
  sentiment_delta : change vs the last cached snapshot  (the ALPHA input)
  attention       : retail eyeball intensity [0,1]
  divergence      : attention high while flow/news negative  (short tell)

Why *delta* is the tradeable signal
-----------------------------------
The *level* of sentiment is largely priced in. What pays is the **inflection**:
a name whose sentiment is turning up from a low base (fresh good news + main-force
starting to accumulate while price is still depressed) tends to mean-revert up;
one rolling over from euphoria tends to fall. So the strategy weights
``sentiment_delta``, computed against a rolling cache on disk.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from .guba_scraper import attention_flow
from .news_scraper import news_sentiment

log = logging.getLogger("ashare.sentiment")
CACHE = Path(__file__).resolve().parents[2] / "data" / "sentiment_cache.json"


def _load_cache() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(snapshot: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(snapshot, ensure_ascii=False, indent=0))


async def _gather(names: dict[str, str], per_stock: int, now: datetime,
                  concurrency: int) -> pd.DataFrame:
    codes = list(names)
    news, attn = await asyncio.gather(
        news_sentiment(names, per_stock, now, concurrency),
        attention_flow(codes, concurrency),
    )
    rows = {}
    for c in codes:
        n = news.get(c, {})
        a = attn.get(c, {})
        news_s = n.get("news_score", 0.0)
        flow = a.get("mainforce_flow", 0.0) / 100.0   # % -> fraction
        attention = a.get("attention", 0.0)
        # Level: blend news polarity and smart-money flow; louder names count more.
        level = 0.6 * news_s + 0.4 * max(-1.0, min(1.0, flow * 4))
        divergence = attention * (-level)             # hot + negative => +divergence
        rows[c] = {
            "sentiment": round(level, 4),
            "attention": round(attention, 4),
            "mainforce_flow": round(flow, 4),
            "divergence": round(divergence, 4),
            "news_n": n.get("news_n", 0),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def compute_sentiment(names: dict[str, str], *, per_stock: int = 8,
                      concurrency: int = 12, now: datetime | None = None,
                      update_cache: bool = True) -> pd.DataFrame:
    """Synchronous entry point. names={code: chinese_name}."""
    now = now or datetime.now()
    df = asyncio.run(_gather(names, per_stock, now, concurrency))

    prev = _load_cache()
    prev_level = {k: v.get("sentiment", 0.0) for k, v in prev.get("levels", {}).items()}
    df["sentiment_delta"] = [
        round(df.at[c, "sentiment"] - prev_level.get(c, 0.0), 4) for c in df.index
    ]

    if update_cache:
        snap = {"ts": now.isoformat(),
                "levels": {c: {"sentiment": float(df.at[c, "sentiment"])}
                           for c in df.index}}
        _save_cache(snap)
    return df
