"""Market-data access layer.

Two backends behind one interface:

* **Choice** (``ChoiceClient``) — the authoritative source once logged in:
  adjusted OHLCV history, fundamentals (PE/PB/mktcap), northbound holdings,
  analyst EPS estimates. Used by the daily factor pipeline and backtester.

* **push2** — East Money's *free, no-auth* real-time quote HTTP API
  (``push2.eastmoney.com``). Used for the live intraday loop and as a fallback
  so the stack produces real prices before Choice credentials are wired in.

The Choice indicator codes below are the exact field mnemonics from the Choice
指标手册 (indicator manual) shipped in this SDK.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
import requests

from .choice_client import ChoiceClient

log = logging.getLogger("ashare.data")

# Choice css/csd field mnemonics -> our tidy column names.
DAILY_FIELDS = {
    "OPEN": "open", "CLOSE": "close", "HIGH": "high", "LOW": "low",
    "VOLUME": "volume", "AMOUNT": "amount", "TURN": "turnover",
}
# Core fields verified to return real data on a standard Choice account.
SNAPSHOT_FIELDS = {
    "CLOSE": "close", "PETTM": "pe_ttm", "MV": "mktcap",
    "TURN": "turnover", "AMOUNT": "amount",
}
# Optional fields — attempted best-effort; silently dropped if the account is
# not entitled (northbound holdings / analyst estimates are premium tiers).
OPTIONAL_SNAPSHOT_FIELDS = {
    "HKHOLDVALUE": "north_value", "ESTNETPROFITYOY": "eps_growth_est",
}

_HTTP = requests.Session()
_HTTP.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"})


def _secid(code: str) -> str:
    """`600519.SH` / `300750.SZ` / `688981.SH` -> push2 secid `1.xxx` / `0.xxx`."""
    num, _, mkt = code.partition(".")
    prefix = "1" if mkt.upper() == "SH" else "0"
    return f"{prefix}.{num}"


@dataclass
class MarketData:
    choice: ChoiceClient

    # -- history (Choice) ----------------------------------------------------
    def history(self, codes, start: str, end: str) -> pd.DataFrame:
        """Adjusted daily OHLCV as MultiIndex(code, date). Choice-only.

        csd returns a flat CODES-indexed frame with a DATES column; we rebuild a
        proper (code, date) MultiIndex so factors and per-name slicing work.
        """
        df = self.choice.csd(codes, ",".join(DAILY_FIELDS), start, end)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()

        def _col(c: str) -> str:
            u = str(c).upper()
            if u in ("CODES", "CODE"):
                return "code"
            if u in ("DATES", "DATE"):
                return "date"
            return DAILY_FIELDS.get(u, str(c).lower())

        df = df.rename(columns={c: _col(c) for c in df.columns})
        if "code" not in df.columns or "date" not in df.columns:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df.dropna(subset=["date"]).set_index(["code", "date"]).sort_index()

    def snapshot(self, codes, tradedate: str) -> pd.DataFrame:
        """Cross-section of fundamentals+price on one date. Choice-only.

        Requests core + optional fields; the client drops any field the account
        is not entitled to, so this never crashes on a premium-tier indicator.
        """
        # Core fields in one clean batch (all entitled -> no fallback needed).
        core = self._rename(
            self.choice.css(codes, ",".join(SNAPSHOT_FIELDS), TradeDate=tradedate),
            SNAPSHOT_FIELDS)
        # Optional premium fields best-effort; skip silently if unentitled.
        try:
            opt = self._rename(
                self.choice.css(codes, ",".join(OPTIONAL_SNAPSHOT_FIELDS),
                                TradeDate=tradedate),
                OPTIONAL_SNAPSHOT_FIELDS)
            core = core.join(opt.drop(columns=[c for c in opt.columns
                             if c in core.columns], errors="ignore"), how="left")
        except Exception as e:
            log.info("optional snapshot fields unavailable: %s", e)
        return core

    # -- realtime (push2, free) ---------------------------------------------
    def realtime(self, codes) -> pd.DataFrame:
        """Live last/pctchg/turnover/volume via free push2 API. No auth needed."""
        if isinstance(codes, str):
            codes = [c.strip() for c in codes.split(",") if c.strip()]
        # map numeric code -> full input code so we can re-key the response
        num_to_full = {c.split(".")[0]: c for c in codes}
        secids = ",".join(_secid(c) for c in codes)
        url = ("https://push2.eastmoney.com/api/qt/ulist.np/get"
               f"?secids={secids}&fields=f12,f14,f2,f3,f5,f6,f8,f10&invt=2&fltt=2")
        rows = []
        try:
            j = _HTTP.get(url, timeout=10).json()
            for d in (j.get("data") or {}).get("diff", []) or []:
                full = num_to_full.get(str(d.get("f12")), str(d.get("f12")))
                rows.append({
                    "code": full, "name": d.get("f14"),
                    "last": d.get("f2"), "pctchg": d.get("f3"),
                    "volume": d.get("f5"), "amount": d.get("f6"),
                    "turnover": d.get("f8"), "vol_ratio": d.get("f10"),
                })
        except Exception as e:  # pragma: no cover - network
            log.warning("push2 realtime failed: %s", e)
        df = pd.DataFrame(rows)
        return df.set_index("code") if not df.empty else df

    def klines(self, codes, bars: int = 70) -> pd.DataFrame:
        """Free forward-adjusted daily OHLCV history via push2his. No auth.

        Returns MultiIndex(code, date) with open/close/high/low/volume/amount,
        plus a 'turnover' proxy (amount z-scored per name) so the turnover-heat
        factor still functions offline. This lets the price factor sleeve run
        for any name without a Choice login.
        """
        if isinstance(codes, str):
            codes = [c.strip() for c in codes.split(",") if c.strip()]
        frames = []
        for c in codes:
            # Tencent first: reliable and fast. East Money history host is
            # chronically IP-throttled, so use it only as a fallback.
            rows = self._kline_tencent(c, bars)
            if not rows:
                rows = self._kline_eastmoney(_secid(c), bars)
            if not rows:
                log.warning("kline %s unavailable from all providers", c)
                continue
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            # turnover proxy = activity / trailing-median activity (amount, else volume)
            act = df["amount"] if df["amount"].abs().sum() else df["volume"]
            med = act.rolling(20, min_periods=5).median()
            df["turnover"] = act / med.replace(0, pd.NA)
            df["code"] = c
            frames.append(df.set_index(["code", "date"]))
        return pd.concat(frames).sort_index() if frames else pd.DataFrame()

    # history host mirror shards — used to route around per-host throttling.
    _HIS_HOSTS = ["push2his.eastmoney.com", "1.push2his.eastmoney.com",
                  "13.push2his.eastmoney.com", "23.push2his.eastmoney.com",
                  "push2his.eastmoney.com"]

    def _kline_eastmoney(self, secid: str, bars: int, tries: int = 3) -> list[dict]:
        """East Money klines via mirror hosts + backoff. [] if unavailable."""
        import time
        path = ("/api/qt/stock/kline/get"
                f"?secid={secid}&fields1=f1,f2&fields2=f51,f52,f53,f54,f55,f56,f57"
                f"&klt=101&fqt=1&end=20500101&lmt={bars}")
        for attempt in range(tries):
            host = self._HIS_HOSTS[attempt % len(self._HIS_HOSTS)]
            try:
                d = (_HTTP.get(f"https://{host}{path}", timeout=15).json()
                     .get("data") or {})
                if d.get("klines"):
                    return [{"date": p[0], "open": float(p[1]), "close": float(p[2]),
                             "high": float(p[3]), "low": float(p[4]),
                             "volume": float(p[5]), "amount": float(p[6])}
                            for p in (ln.split(",") for ln in d["klines"])]
            except Exception:
                pass
            time.sleep(1.0 * (attempt + 1))
        return []

    def _kline_tencent(self, code: str, bars: int) -> list[dict]:
        """Independent fallback: Tencent gtimg forward-adjusted daily klines."""
        num, _, mkt = code.partition(".")
        sym = f"{'sh' if mkt.upper() == 'SH' else 'sz'}{num}"
        url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
               f"?param={sym},day,,,{bars},qfq")
        try:
            node = (_HTTP.get(url, timeout=15).json().get("data") or {}).get(sym) or {}
            kl = node.get("qfqday") or node.get("day") or []
            return [{"date": p[0], "open": float(p[1]), "close": float(p[2]),
                     "high": float(p[3]), "low": float(p[4]),
                     "volume": float(p[5]), "amount": 0.0} for p in kl]
        except Exception:
            return []

    @staticmethod
    def _rename(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.drop(columns=[c for c in df.columns if str(c).upper() == "DATES"],
                     errors="ignore")
        low = {c: mapping.get(str(c).upper(), str(c).lower()) for c in df.columns}
        return df.rename(columns=low)
