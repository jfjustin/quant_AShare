"""US cross-market lead-lag overlay.

Thesis
------
Many A-share tech names are *business*-coupled to a US mega-cap (they supply it,
compete with it, or ride the same demand cycle) but their **stocks** don't move
together tick-for-tick — the A-share often reacts a session or two later, or
under-reacts entirely (different investor base, T+1, capital controls, timezone).
That gap is tradeable: when the US peer gaps up hard overnight on a shared
catalyst and the A-share hasn't caught up, the A-share tends to play catch-up.

This module:
  * maps A-share codes -> US business peers (with a rationale),
  * pulls US daily history (Yahoo Finance, free, no key),
  * computes lagged return correlation (US[t-1] vs A[t]) and a "catch-up gap"
    (how much the US peer has moved that the A-share hasn't yet).

`not reflected in stock` == high business linkage but the A-share hasn't repriced
-> positive catch-up gap -> lean long (or short if the US peer sold off).
"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("ashare.us")

_HTTP = requests.Session()
_HTTP.headers.update({"User-Agent": "Mozilla/5.0"})

# A-share code -> US business peers + why they are coupled.
# `link` is a static business-coupling strength in [0,1] (analyst judgement).
A2US_PEERS: dict[str, dict] = {
    "300476.SZ": {"us": ["NVDA"], "link": 0.85,
                  "why": "胜宏科技 Victory Giant — high-layer PCBs for AI/GPU servers (Nvidia supply chain)"},
    "300308.SZ": {"us": ["NVDA", "AVGO"], "link": 0.80,
                  "why": "中际旭创 Innolight — 800G optical transceivers for AI datacenters"},
    "688256.SH": {"us": ["NVDA"], "link": 0.78,
                  "why": "寒武纪 Cambricon — domestic AI training/inference chips (Nvidia analogue)"},
    "688041.SH": {"us": ["NVDA", "AMD"], "link": 0.72,
                  "why": "海光信息 Hygon — x86 CPU + DCU GPGPU accelerators"},
    "688981.SH": {"us": ["TSM", "INTC"], "link": 0.75,
                  "why": "中芯国际 SMIC — leading-edge foundry (TSMC analogue)"},
    "002371.SZ": {"us": ["AMAT", "LRCX"], "link": 0.70,
                  "why": "北方华创 Naura — semiconductor deposition/etch equipment"},
    "688012.SH": {"us": ["LRCX", "AMAT"], "link": 0.70,
                  "why": "中微公司 AMEC — plasma etch tools"},
    "603501.SH": {"us": ["MU", "STM"], "link": 0.60,
                  "why": "韦尔股份 Will Semi — CMOS image sensors / analog"},
    "603986.SH": {"us": ["MU"], "link": 0.62,
                  "why": "兆易创新 GigaDevice — NOR flash / DRAM / MCU (memory cycle)"},
    "300782.SZ": {"us": ["TSM"], "link": 0.55,
                  "why": "卓胜微 Maxscend — RF front-end"},
    "002049.SZ": {"us": ["NVDA", "XLNX"], "link": 0.55,
                  "why": "紫光国微 — FPGA / special-purpose logic"},
    "300750.SZ": {"us": ["TSLA"], "link": 0.55,
                  "why": "宁德时代 CATL — EV battery cells (Tesla demand cycle)"},
    "002594.SZ": {"us": ["TSLA"], "link": 0.60,
                  "why": "比亚迪 BYD — NEV OEM (Tesla competitor/analogue)"},
    "601127.SH": {"us": ["TSLA"], "link": 0.50,
                  "why": "赛力斯 Seres — NEV OEM (AITO/Huawei)"},
    "300474.SZ": {"us": ["NVDA"], "link": 0.70,
                  "why": "景嘉微 JingJiaWei — domestic GPU"},
}

# US peers that anchor the new cross-market theme (for universe membership).
US_THEME_MEMBERS = list(A2US_PEERS.keys())


def _yahoo(ticker: str, rng: str = "6mo") -> pd.Series:
    """Daily close Series (date-indexed) for a US ticker via Yahoo. [] on failure."""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?range={rng}&interval=1d")
    try:
        j = _HTTP.get(url, timeout=15).json()
        res = j["chart"]["result"][0]
        ts = res["timestamp"]
        cl = res["indicators"]["quote"][0]["close"]
        s = pd.Series(cl, index=pd.to_datetime(ts, unit="s").normalize(), name=ticker)
        return s.dropna()
    except Exception as e:  # pragma: no cover - network
        log.debug("yahoo %s failed: %s", ticker, e)
        return pd.Series(dtype=float)


def us_history(tickers: list[str], rng: str = "6mo") -> pd.DataFrame:
    cols = {t: _yahoo(t, rng) for t in tickers}
    cols = {t: s for t, s in cols.items() if not s.empty}
    return pd.DataFrame(cols) if cols else pd.DataFrame()


def leadlag(a_close: pd.Series, us_close: pd.Series, lag: int = 1,
            window: int = 60) -> dict:
    """Correlate US peer with the A-share and measure the pending catch-up.

    a_close / us_close: date-indexed close prices (any calendar overlap ok).
    Returns correlations and catch-up gaps (US move minus A-share move).
    """
    a_ret = a_close.pct_change().rename("a")
    u_ret = us_close.pct_change().rename("u")
    df = pd.concat([a_ret, u_ret], axis=1).dropna()
    if len(df) < 20:
        return {}
    df = df.iloc[-window:]
    corr = float(df["a"].corr(df["u"]))
    # US leads A by `lag` sessions (overnight US -> next-day A):
    lagged = pd.concat([df["a"], df["u"].shift(lag)], axis=1).dropna()
    corr_lag = float(lagged["a"].corr(lagged["u"])) if len(lagged) > 10 else np.nan

    def _ret(s, k):
        s = s.dropna()
        return float(s.iloc[-1] / s.iloc[-k - 1] - 1) if len(s) > k else np.nan

    out = {
        "corr": round(corr, 3),
        "corr_lag1": round(corr_lag, 3) if corr_lag == corr_lag else None,
        "us_ret_1d": _ret(us_close, 1),
        "us_ret_5d": _ret(us_close, 5),
        "a_ret_1d": _ret(a_close, 1),
        "a_ret_5d": _ret(a_close, 5),
    }
    # catch-up gap = how much the US peer moved that the A-share has NOT (5d/20d).
    out["catchup_5d"] = (out["us_ret_5d"] - out["a_ret_5d"]) if (
        out["us_ret_5d"] == out["us_ret_5d"] and out["a_ret_5d"] == out["a_ret_5d"]) else np.nan
    out["catchup_20d"] = (_ret(us_close, 20) - _ret(a_close, 20))
    return out


def cross_market_read(a_code: str, a_close: pd.Series) -> dict | None:
    """Full US cross-market read for one A-share name, or None if no peer mapped."""
    peer = A2US_PEERS.get(a_code)
    if not peer:
        return None
    usd = us_history(peer["us"])
    if usd.empty:
        return {"peer": peer, "error": "US data unavailable"}
    # use the most-correlated peer as the lead
    best_t, best = None, {}
    for t in usd.columns:
        ll = leadlag(a_close, usd[t])
        if ll and (best_t is None or abs(ll.get("corr", 0)) > abs(best.get("corr", 0))):
            best_t, best = t, ll
    return {"peer": peer, "lead_ticker": best_t,
            "lead_close": float(usd[best_t].iloc[-1]) if best_t else None,
            "metrics": best}
