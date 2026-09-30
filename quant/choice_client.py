"""Robust wrapper around East Money Choice's EmQuantAPI.

Why this exists
---------------
The raw ``EmQuantAPI.c`` class is a thin ctypes shim over a C DLL. It returns
``EmQuantData`` objects with awkward flat-list payloads, has no retry logic, and
happily returns error codes instead of raising. This wrapper gives the rest of
the stack three clean, pandas-native primitives:

    client.css(codes, fields, TradeDate=...)   -> cross-section  (stocks x fields)  one date
    client.csd(codes, fields, start, end)      -> time-series    per code
    client.sector(pukey, date)                 -> index/sector constituents

plus ``login()`` / ``close()`` and an ``is_live`` flag. Everything degrades
gracefully: if Choice is not logged in, callers can fall back to the free
East Money push2 HTTP endpoints (see quant/data.py) so the sentiment stack and
backtester still run.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable, Sequence

import pandas as pd

log = logging.getLogger("ashare.choice")


class ChoiceError(RuntimeError):
    def __init__(self, code, msg):
        super().__init__(f"Choice error {code}: {msg}")
        self.code = code
        self.msg = msg


def _to_list(codes: str | Iterable[str]) -> list[str]:
    if isinstance(codes, str):
        return [c.strip() for c in codes.split(",") if c.strip()]
    return [str(c).strip() for c in codes]


def _join(x: str | Iterable[str]) -> str:
    return ",".join(_to_list(x))


class ChoiceClient:
    """Session-managed, pandas-returning facade over EmQuantAPI."""

    def __init__(self, username: str = "", password: str = "",
                 start_options: str = "ForceLogin=1"):
        self.username = username or os.getenv("EM_USERNAME", "")
        self.password = password or os.getenv("EM_PASSWORD", "")
        self.start_options = start_options
        self._c = None
        self.is_live = False

    # -- lifecycle -----------------------------------------------------------
    def login(self) -> bool:
        """Import the API, load the native lib, authenticate.

        Returns True on success. Never raises on auth failure — sets is_live
        False and logs, so the caller can decide whether to fall back.
        """
        try:
            from EmQuantAPI import c  # registered via installEmQuantAPI.py .pth
        except Exception as e:  # pragma: no cover - env specific
            log.warning("EmQuantAPI import failed (%s). Running in OFFLINE mode.", e)
            self.is_live = False
            return False

        self._c = c

        def _attempt(opts: str, label: str) -> bool:
            try:
                data = c.start(opts, "", None)
            except Exception as e:  # pragma: no cover
                log.warning("Choice start() [%s] raised %s.", label, e)
                return False
            code = str(getattr(data, "ErrorCode", "0"))
            if code in ("0", "10000000"):  # success sentinel
                log.info("Choice login OK via %s.", label)
                return True
            log.info("Choice login via %s failed [%s] %s.",
                     label, code, getattr(data, "ErrorMsg", ""))
            return False

        # 1) Prefer the activation-TOKEN login (no creds). This is what the
        #    LoginActivator provisions and what most Choice accounts support.
        if _attempt(self.start_options, "activation-token"):
            self.is_live = True
            return True

        # 2) Fall back to username/password login if credentials are supplied.
        if self.username and self.password:
            opts = f"UserName={self.username},PassWord={self.password},{self.start_options}"
            if _attempt(opts, "username/password"):
                self.is_live = True
                return True

        log.warning("Choice login failed on all methods. OFFLINE mode "
                    "(run scripts/activate_choice.sh to (re)activate).")
        self.is_live = False
        return False

    def close(self) -> None:
        if self._c is not None and self.is_live:
            try:
                self._c.stop()
            except Exception:
                pass
        self.is_live = False

    def __enter__(self) -> "ChoiceClient":
        self.login()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- helpers -------------------------------------------------------------
    def _require(self):
        if not self.is_live or self._c is None:
            raise ChoiceError("OFFLINE", "Choice not logged in")
        return self._c

    @staticmethod
    def _opts(base: str, kw: dict) -> str:
        extra = ",".join(f"{k}={v}" for k, v in kw.items() if v is not None)
        parts = [p for p in (base, extra) if p]
        return ",".join(parts)

    # -- primitives ----------------------------------------------------------
    def css(self, codes, fields, **kw) -> pd.DataFrame:
        """Cross-section snapshot: one row per code, one column per field.

        Extra kwargs become Choice options (e.g. ``TradeDate='20260717'``).
        """
        c = self._require()
        opts = self._opts("Ispandas=1", kw)
        field_list = _to_list(fields)
        try:
            return self._as_frame(c.css(_join(codes), _join(fields), opts), fields)
        except ChoiceError as e:
            if len(field_list) <= 1:
                raise
            # A single unentitled/invalid field errors the whole batch. Retry
            # field-by-field and keep the columns that succeed.
            log.info("css batch failed (%s); retrying field-by-field", e.msg)
            kept = {}
            for f in field_list:
                try:
                    part = self._as_frame(c.css(_join(codes), f, opts), [f])
                    # css returns a 'DATES' column too — pick the value column.
                    valcols = [c2 for c2 in part.columns
                               if str(c2).upper() != "DATES"]
                    if valcols and not part.empty:
                        kept[f] = part[valcols[0]]
                except ChoiceError:
                    log.info("  dropping unavailable field: %s", f)
            if not kept:
                raise
            out = pd.DataFrame(kept)
            # single-field css can return object dtype; coerce where numeric
            for col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="ignore")
            return out

    def csd(self, codes, fields, start, end, **kw) -> pd.DataFrame:
        """Time-series: MultiIndex (code, date) rows, one column per field."""
        c = self._require()
        opts = self._opts("Ispandas=1,period=1,adjustflag=1,curtype=1,pricetype=1", kw)
        data = c.csd(_join(codes), _join(fields), start, end, opts)
        return self._as_frame(data, fields)

    def sector(self, pukeycode: str, tradedate: str) -> list[str]:
        """Return constituent codes of a Choice sector/index at a date."""
        c = self._require()
        data = c.sector(pukeycode, tradedate, "Ispandas=0")
        if str(getattr(data, "ErrorCode", "0")) not in ("0", "10000000"):
            raise ChoiceError(data.ErrorCode, getattr(data, "ErrorMsg", ""))
        flat = list(getattr(data, "Data", []) or [])
        # Choice returns [code, name, code, name, ...]; take the code slots.
        return [flat[i] for i in range(0, len(flat), 2) if "." in str(flat[i])]

    def tradedates(self, start: str, end: str, market: str = "CNSESH") -> list[str]:
        c = self._require()
        data = c.tradedates(start, end, f"Ispandas=1,market={market}")
        if isinstance(data, pd.DataFrame):
            return [str(x) for x in data.iloc[:, 0].tolist()]
        return [str(x) for x in (getattr(data, "Data", []) or [])]

    # -- conversion ----------------------------------------------------------
    @staticmethod
    def _as_frame(data, fields) -> pd.DataFrame:
        """Coerce an EmQuantData / DataFrame result into a tidy DataFrame."""
        if isinstance(data, pd.DataFrame):
            return ChoiceClient._coerce_numeric(data)
        code = str(getattr(data, "ErrorCode", "0"))
        if code not in ("0", "10000000"):
            raise ChoiceError(code, getattr(data, "ErrorMsg", ""))
        # Manual fallback when Ispandas is unavailable.
        cols = _to_list(fields)
        rows = {}
        for cd in getattr(data, "Codes", []) or []:
            vals = data.Data.get(cd, [])
            rows[cd] = vals[: len(cols)]
        return ChoiceClient._coerce_numeric(
            pd.DataFrame.from_dict(rows, orient="index", columns=cols))

    @staticmethod
    def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
        """Choice with Ispandas=1 often returns object dtype; make numerics
        numeric (leaving genuine text columns like names untouched)."""
        if df is None or df.empty:
            return df
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="ignore")
        return df
