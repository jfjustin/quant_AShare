"""Async news-headline scraper + sentiment scorer (East Money search API).

Pulls the most recent headlines mentioning each name and scores them with the
finance lexicon. Recent, dated, and free — no auth. We weight each headline by
recency (exponential decay) so a fresh negative print outranks week-old praise.
"""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.parse as up
from datetime import datetime

import aiohttp

from .lexicon import score_text

log = logging.getLogger("ashare.news")

_SEARCH = "https://search-api-web.eastmoney.com/search/jsonp"
_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://so.eastmoney.com/"}


def _build_param(keyword: str, page_size: int) -> str:
    body = {
        "keyword": keyword,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "param": {"cmsArticleWebOld": {
            "sort": "time", "pageIndex": 1, "pageSize": page_size}},
    }
    return up.quote(json.dumps(body, ensure_ascii=False))


async def _fetch_one(session: aiohttp.ClientSession, keyword: str,
                     page_size: int) -> list[dict]:
    url = f"{_SEARCH}?cb=x&param={_build_param(keyword, page_size)}"
    try:
        async with session.get(url, headers=_HEADERS, timeout=15) as r:
            text = await r.text()
    except Exception as e:  # pragma: no cover - network
        log.debug("news fetch %s failed: %s", keyword, e)
        return []
    try:
        raw = text[text.find("(") + 1: text.rfind(")")]
        data = json.loads(raw)
        return data.get("result", {}).get("cmsArticleWebOld", []) or []
    except Exception:
        return []


def _recency_weight(date_str: str, now: datetime, half_life_h: float = 24.0) -> float:
    try:
        dt = datetime.strptime(date_str[:19], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return 0.3
    age_h = max(0.0, (now - dt).total_seconds() / 3600.0)
    return 0.5 ** (age_h / half_life_h)


async def news_sentiment(names: dict[str, str], per_stock: int, now: datetime,
                         concurrency: int = 12) -> dict[str, dict]:
    """names: {code: chinese_name}. Returns {code: {score, n, attention}}."""
    sem = asyncio.Semaphore(concurrency)
    out: dict[str, dict] = {}

    async with aiohttp.ClientSession() as session:
        async def work(code: str, name: str):
            async with sem:
                arts = await _fetch_one(session, name or code, per_stock)
            num = den = 0.0
            for a in arts:
                title = (a.get("title") or "").replace("<em>", "").replace("</em>", "")
                doc = score_text(title + " " + (a.get("content") or "")[:120])
                w = _recency_weight(a.get("date", ""), now) * (0.3 + doc.magnitude)
                num += w * doc.polarity
                den += w
            out[code] = {
                "news_score": (num / den) if den else 0.0,
                "news_n": len(arts),
                "news_weight": round(den, 3),
            }

        await asyncio.gather(*(work(c, n) for c, n in names.items()))
    return out
