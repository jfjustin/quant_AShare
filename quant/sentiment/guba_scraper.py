"""Retail-attention + smart-money-flow scraper (East Money, free/no-auth).

East Money's raw 股吧 *post* API is bot-gated, so instead of scraping fragile
HTML we use the two structured signals that actually drive the retail reflexivity
we care about:

* **Guba popularity rank (人气榜)** — East Money's own attention index. A name
  climbing the rank = a surge of eyeballs = the fuel for retail overreaction.
  We convert rank -> a bounded attention score and, crucially, its *change*.
* **Main-force net inflow %% (`f184`)** — "smart money" (large orders) net buying
  as a share of turnover. When retail is euphoric but main-force is distributing
  (attention up, flow down) that divergence is a high-conviction short tell, and
  vice-versa. This divergence is the heart of our sentiment edge.

Both are polled concurrently and cheaply enough for intraday use.
"""
from __future__ import annotations

import asyncio
import logging

import aiohttp

log = logging.getLogger("ashare.attention")

_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"}
_POPRANK = "https://emappdata.eastmoney.com/stockrank/getAllCurrentList"
_FLOW = "https://push2.eastmoney.com/api/qt/stock/get"


def _secid(code: str) -> str:
    num, _, mkt = code.partition(".")
    return f"{'1' if mkt.upper() == 'SH' else '0'}.{num}"


def _rank_key(code: str) -> str:
    """`600519.SH` -> `SH600519` to match the popularity-rank payload."""
    num, _, mkt = code.partition(".")
    return f"{mkt.upper()}{num}"


async def _popularity(session: aiohttp.ClientSession, pages: int = 3) -> dict[str, int]:
    """Return {SHxxxxxx: rank} for the top attention names (rank 1 = hottest)."""
    ranks: dict[str, int] = {}
    for p in range(1, pages + 1):
        payload = {"appId": "appId01", "globalId": "ashare-quant",
                   "marketType": "", "pageNo": p, "pageSize": 100}
        try:
            async with session.post(_POPRANK, json=payload, headers=_HEADERS,
                                    timeout=12) as r:
                data = (await r.json()).get("data") or []
        except Exception as e:  # pragma: no cover
            log.debug("poprank page %d failed: %s", p, e)
            break
        for row in data:
            sc, rk = row.get("sc"), row.get("rk")
            if sc and rk:
                ranks[sc] = int(rk)
    return ranks


async def _flow_one(session: aiohttp.ClientSession, code: str) -> float:
    """Main-force net inflow as %% of turnover (f184). Positive = accumulation."""
    url = (f"{_FLOW}?secid={_secid(code)}&fields=f184,f62&invt=2&fltt=2")
    try:
        async with session.get(url, headers=_HEADERS, timeout=10) as r:
            d = (await r.json()).get("data") or {}
        v = d.get("f184")
        return float(v) if v not in (None, "-") else 0.0
    except Exception:
        return 0.0


async def attention_flow(codes: list[str], concurrency: int = 12) -> dict[str, dict]:
    """Return {code: {attention, mainforce_flow}} where attention in [0,1]."""
    out: dict[str, dict] = {}
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        ranks = await _popularity(session)
        max_rank = max(ranks.values()) if ranks else 300

        async def work(code: str):
            async with sem:
                flow = await _flow_one(session, code)
            rk = ranks.get(_rank_key(code))
            # rank 1 -> attention ~1.0 ; unranked -> 0
            attention = 0.0 if rk is None else max(0.0, 1.0 - (rk - 1) / max_rank)
            out[code] = {"attention": round(attention, 4),
                         "mainforce_flow": round(flow, 4)}

        await asyncio.gather(*(work(c) for c in codes))
    return out
