"""Investable-universe construction for the tech / supply-chain book.

We deliberately fish in the *volatile, high-PE growth* pond — semiconductors,
AI compute, the NEV / battery / PV supply chains, consumer electronics — because
that is where A-share retail overreaction (and therefore our reversal + sentiment
edge) is strongest. We then impose liquidity, size and quality gates so the book
is actually tradeable under T+1 with 10–20% price limits.

Choice sector ("板块") codes below are the East Money thematic-index pukey codes.
When Choice is offline we fall back to a small curated seed list of bellwether
names so the pipeline still produces something end-to-end.
"""
from __future__ import annotations

import logging

import pandas as pd

from .config import Config
from .data import MarketData

log = logging.getLogger("ashare.universe")

# theme -> East Money thematic index code (resolved via ChoiceClient.sector).
SECTOR_MAP = {
    "semiconductors":      "007200",   # 半导体
    "ai_computing":        "018880",   # AI算力 / CPO
    "new_energy_vehicle":  "016113",   # 新能源汽车
    "battery_chain":       "016120",   # 锂电池
    "photovoltaic":        "016114",   # 光伏
    "consumer_electronics": "007207",  # 消费电子
}

# Offline seed: liquid bellwethers per theme (used only when Choice is down).
SEED = {
    "semiconductors":       ["688981.SH", "603986.SH", "688012.SH", "002371.SZ", "688041.SH"],
    "ai_computing":         ["300308.SZ", "603019.SH", "300502.SZ", "688256.SH"],
    "new_energy_vehicle":   ["002594.SZ", "601127.SH", "002460.SZ"],
    "battery_chain":        ["300750.SZ", "300014.SZ", "002812.SZ", "688005.SH"],
    "photovoltaic":         ["601012.SH", "300274.SZ", "688599.SH"],
    "consumer_electronics": ["002475.SZ", "002241.SZ", "300136.SZ"],
}


def _board_ok(code: str, cfg: Config) -> bool:
    num = code.split(".")[0]
    if num.startswith("688"):
        return cfg.get("universe.boards.include_star", True)
    if num.startswith("300"):
        return cfg.get("universe.boards.include_chinext", True)
    return cfg.get("universe.boards.include_main", True)


def resolve_candidates(cfg: Config, md: MarketData, tradedate: str) -> pd.DataFrame:
    """Return DataFrame indexed by code with a 'theme' column, pre-filter."""
    rows = []
    themes = cfg.get("universe.themes", list(SECTOR_MAP))
    for theme in themes:
        codes: list[str] = []
        pukey = SECTOR_MAP.get(theme)
        if md.choice.is_live and pukey:
            try:
                codes = md.choice.sector(pukey, tradedate)
            except Exception as e:
                log.warning("sector(%s) failed: %s -> seed", theme, e)
        if not codes:
            codes = SEED.get(theme, [])
        for c in codes:
            if _board_ok(c, cfg):
                rows.append({"code": c, "theme": theme})
    df = pd.DataFrame(rows).drop_duplicates("code")
    return df.set_index("code") if not df.empty else df


def apply_quality_filters(cfg: Config, md: MarketData, cand: pd.DataFrame,
                          tradedate: str) -> pd.DataFrame:
    """Attach fundamentals and drop names that fail size/liquidity/PE gates."""
    if cand.empty:
        return cand
    if not md.choice.is_live:
        # Offline: keep the seed as-is (no fundamentals to filter on).
        cand["pe_ttm"] = float("nan")
        return cand

    snap = md.snapshot(list(cand.index), tradedate)
    df = cand.join(snap, how="left")
    f = cfg.get("universe.filters", {})

    def _keep(row) -> bool:
        mv_yi = (row.get("mktcap") or 0) / 1e8
        if not (f.get("min_mktcap_yi", 0) <= mv_yi <= f.get("max_mktcap_yi", 1e9)):
            return False
        adv_wan = (row.get("amount") or 0) / 1e4
        if adv_wan < f.get("min_adv_20_wan", 0):
            return False
        pe = row.get("pe_ttm")
        if pe is None or pe != pe or pe < f.get("min_pe_ttm", 0):
            return False  # negative/low PE not in our high-growth mandate
        return True

    mask = df.apply(_keep, axis=1)
    kept = df[mask]
    log.info("universe: %d candidates -> %d after filters", len(df), len(kept))
    return kept


def build_universe(cfg: Config, md: MarketData, tradedate: str) -> pd.DataFrame:
    cand = resolve_candidates(cfg, md, tradedate)
    return apply_quality_filters(cfg, md, cand, tradedate)
