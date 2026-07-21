#!/usr/bin/env python3
"""Single-stock deep-dive — performance, factors, sentiment, and the book's verdict.

    python scripts/inspect_stock.py 300750.SZ
    python scripts/inspect_stock.py 688981.SH --theme semiconductors
    python scripts/inspect_stock.py 300750           # suffix auto-detected

Runs entirely on FREE East Money endpoints (no Choice login needed): daily
history, real-time quote, live news + attention + main-force flow. Factor scores
are shown *relative to the stock's theme peers* so they mean something.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.parse as up
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.config import Config
from quant.choice_client import ChoiceClient
from quant.data import MarketData
from quant.universe import SEED, SECTOR_MAP
from quant.signals import build_alpha
from quant.sentiment.sentiment_engine import compute_sentiment
from quant.sentiment.lexicon import score_text

_H = {"User-Agent": "Mozilla/5.0", "Referer": "https://so.eastmoney.com/"}


def _normalize(code: str) -> str:
    """Accept 300750 / 300750.SZ / sz300750 -> 300750.SZ."""
    code = code.upper().replace("SZ", "").replace("SH", "").strip(". ")
    num = "".join(ch for ch in code if ch.isdigit())
    if "." in code:
        return code
    # 6xx -> SH, else SZ (688 is SH STAR, 300/000/002 SZ)
    mkt = "SH" if num.startswith(("6", "9")) or num.startswith("688") else "SZ"
    return f"{num}.{mkt}"


def _theme_of(code: str) -> str | None:
    for th, lst in SEED.items():
        if code in lst:
            return th
    return None


def _headlines(name: str, n: int = 6) -> list[tuple[str, str, float]]:
    body = {"keyword": name, "type": ["cmsArticleWebOld"], "client": "web",
            "param": {"cmsArticleWebOld": {"sort": "time", "pageIndex": 1, "pageSize": n}}}
    url = f"https://search-api-web.eastmoney.com/search/jsonp?cb=x&param={up.quote(json.dumps(body, ensure_ascii=False))}"
    try:
        t = requests.get(url, headers=_H, timeout=12).text
        d = json.loads(t[t.find("(") + 1:t.rfind(")")])
        arts = d.get("result", {}).get("cmsArticleWebOld", []) or []
    except Exception:
        return []
    out = []
    for a in arts[:n]:
        title = (a.get("title") or "").replace("<em>", "").replace("</em>", "")
        out.append((title, a.get("date", "")[:16], score_text(title).polarity))
    return out


def _perf(hist_one: pd.DataFrame) -> dict:
    c = hist_one["close"].dropna()
    def r(k):
        return (c.iloc[-1] / c.iloc[-k - 1] - 1) if len(c) > k else float("nan")
    ret = c.pct_change().dropna()
    return {"last": c.iloc[-1], "ret_1d": r(1), "ret_5d": r(5), "ret_20d": r(20),
            "ret_60d": r(60), "vol_20d_ann": ret.iloc[-20:].std() * np.sqrt(252),
            "max_60d": c.iloc[-60:].max() if len(c) >= 60 else c.max(),
            "min_60d": c.iloc[-60:].min() if len(c) >= 60 else c.min()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("code")
    ap.add_argument("--theme", default=None, help="override theme peer group")
    ap.add_argument("--bars", type=int, default=70)
    args = ap.parse_args()

    code = _normalize(args.code)
    theme = args.theme or _theme_of(code) or "semiconductors"
    peers = sorted(set(SEED.get(theme, [])) | {code})

    cfg = Config.load()
    md = MarketData(choice=ChoiceClient())   # offline; free endpoints only

    # names + live quote
    rt = md.realtime(peers)
    name = rt.at[code, "name"] if (not rt.empty and code in rt.index) else code
    last = rt.at[code, "last"] if (not rt.empty and code in rt.index) else float("nan")
    pctchg = rt.at[code, "pctchg"] if (not rt.empty and code in rt.index) else float("nan")

    # free daily history for the whole peer set
    hist = md.klines(peers, bars=args.bars)
    if hist.empty or code not in hist.index.get_level_values(0):
        print(f"No history for {code}."); return
    perf = _perf(hist.xs(code, level=0))

    # sentiment for the peer set (so this name's delta/attention is comparable)
    names = {c: (rt.at[c, "name"] if (not rt.empty and c in rt.index) else c) for c in peers}
    try:
        sent = compute_sentiment(names, per_stock=8, concurrency=12,
                                 now=datetime.now(), update_cache=False)
    except Exception:
        sent = pd.DataFrame()

    # cross-sectional factor + alpha table over the peer group
    meta = pd.DataFrame({"theme": theme}, index=peers)
    z = build_alpha(hist, snap=pd.DataFrame(index=peers), sentiment=sent, meta=meta,
                    weights=cfg.get("signal.weights", {}),
                    winsor=cfg.get("signal.winsorize", 0.025),
                    neutralize=[])   # single theme -> no theme neutralization
    row = z.loc[code] if code in z.index else pd.Series(dtype=float)
    rank = int((z["alpha"] > z.at[code, "alpha"]).sum()) + 1 if code in z.index else 0
    pct = 1 - (rank - 1) / max(1, len(z))

    # ---- render ----
    print("\n" + "=" * 66)
    print(f"  {name}  ({code})   theme: {theme}")
    print(f"  last {last}  ({pctchg:+.2f}% today)")
    print("=" * 66)

    print("\n  PERFORMANCE")
    print(f"    1d {perf['ret_1d']:+.2%}   5d {perf['ret_5d']:+.2%}   "
          f"20d {perf['ret_20d']:+.2%}   60d {perf['ret_60d']:+.2%}")
    print(f"    20d realized vol (ann): {perf['vol_20d_ann']:.1%}")
    print(f"    60d range: {perf['min_60d']:.2f} – {perf['max_60d']:.2f}   "
          f"(last is {(perf['last']-perf['min_60d'])/(perf['max_60d']-perf['min_60d']+1e-9):.0%} of range)")

    print("\n  FACTOR SCORES  (z vs", len(peers), theme, "peers; + = long-favorable)")
    for col in ["st_reversal", "sentiment_delta", "ivol_lottery", "turnover_heat",
                "residual_mom", "northbound_flow", "eps_revision"]:
        if col in z.columns and code in z.index and pd.notna(z.at[code, col]):
            print(f"    {col:16s} {z.at[code, col]:+.2f}")

    if not sent.empty and code in sent.index:
        s = sent.loc[code]
        print("\n  LIVE SENTIMENT")
        print(f"    level {s.get('sentiment', float('nan')):+.3f}   "
              f"delta {s.get('sentiment_delta', float('nan')):+.3f}   "
              f"attention {s.get('attention', float('nan')):.2f}   "
              f"main-force flow {s.get('mainforce_flow', float('nan')):+.3f}   "
              f"divergence {s.get('divergence', float('nan')):+.3f}")

    print("\n  RECENT HEADLINES (lexicon-scored)")
    for title, dt, pol in _headlines(name):
        tag = "▲" if pol > 0.05 else ("▼" if pol < -0.05 else "·")
        print(f"    {tag} [{pol:+.2f}] {dt}  {title[:46]}")

    print("\n  BOOK VERDICT")
    alpha = float(z.at[code, "alpha"]) if code in z.index else 0.0
    lean = "LONG" if alpha > 0.15 else ("SHORT" if alpha < -0.15 else "NEUTRAL")
    print(f"    composite alpha {alpha:+.2f}  ->  {lean}")
    print(f"    rank {rank}/{len(z)} in {theme}  ({pct:.0%} percentile)")

    # ---- FINAL VERDICT: buy-signal % and probability of profit ----
    def _sig(x):  # logistic
        return 1.0 / (1.0 + np.exp(-x))
    # Buy signal 0-100 (50 = neutral). Blend alpha magnitude and peer percentile.
    buy_signal = 100 * (0.7 * _sig(1.0 * alpha) + 0.3 * pct)
    # P(profit) at the ~5d reversal horizon. Base rate 50%, tilted by signal.
    # Tilt deliberately capped at ±16% — these edges are real but SMALL; a
    # single-name call is noisy. Conviction scales the honest edge.
    edge = 0.16 * np.tanh(alpha / 1.3)
    side = "LONG" if buy_signal >= 50 else "SHORT"
    p_profit = 0.50 + (edge if side == "LONG" else -edge)  # P(dir. trade wins)
    # confidence band widens when the peer sample is small / signal weak
    band = 0.06 + 0.10 * (1 - abs(2 * pct - 1))

    print("\n  ══ FINAL VERDICT " + "═" * 47)
    bar = int(round(buy_signal / 5))
    print(f"    BUY SIGNAL      {buy_signal:5.1f}% "
          f"[{'█'*bar}{'░'*(20-bar)}]  (100=strong buy, 0=strong sell)")
    print(f"    RECOMMENDED     {side}"
          + ("" if lean != "NEUTRAL" else "  (low conviction — near neutral)"))
    print(f"    P(profit ~5d)   {p_profit:6.1%}   ±{band:.0%}   "
          f"(base 50%; edge from reversal+sentiment)")
    print(f"    conviction      {abs(buy_signal-50)/50:.0%}   "
          f"(distance from neutral)")
    print("    " + "─" * 62)
    print("    NOTE: model estimate, ~5-day horizon, NOT a guarantee. Edge is")
    print("    small per-name (that's why the real book holds ~40 names). P(profit)")
    print("    is uncalibrated until a live Choice backtest fits it to realized hits.")
    print()


if __name__ == "__main__":
    main()
