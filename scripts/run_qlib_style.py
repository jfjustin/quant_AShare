#!/usr/bin/env python3
"""Qlib-style factor-model backtest on A-share tech names (FREE data, no Choice).

Reproduces Qlib's Alpha158 + LightGBM method and reports the same metrics
(IC / Rank IC / ICIR) plus a top-minus-bottom quantile backtest, so results are
directly comparable in methodology to Qlib's ~IC 0.045 CSI300 benchmark.

    python scripts/run_qlib_style.py                 # default universe, 5d label
    python scripts/run_qlib_style.py --horizon 2 --bars 800
    python scripts/run_qlib_style.py --quantile 0.2

Data: Tencent/East Money free daily klines (~3y). Cross-section is smaller than
CSI300, so expect noisier IC — the point is a real, look-ahead-free pipeline.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.data import MarketData
from quant.choice_client import ChoiceClient
from quant.universe import SEED
from quant.us_market import A2US_PEERS
from quant.alpha158 import make_features
from quant.qlib_model import fit_predict, backtest_long_short

# A broad-ish liquid A-share tech/growth cross-section (more names -> cleaner IC).
EXTRA = [
    "603259.SH", "300760.SZ", "002415.SZ", "000725.SZ", "002230.SZ", "300124.SZ",
    "688111.SH", "688036.SH", "688008.SH", "603259.SH", "300661.SZ", "603160.SH",
    "002049.SZ", "300223.SZ", "688008.SH", "002460.SZ", "300450.SZ", "300751.SZ",
    "002709.SZ", "300496.SZ", "300759.SZ", "688169.SH", "688187.SH", "688516.SH",
    "603160.SH", "002236.SZ", "300454.SZ", "002466.SZ", "600584.SH", "603195.SH",
    "000063.SZ", "600745.SH", "603160.SH", "688396.SH", "603893.SH", "300142.SZ",
    "300347.SZ", "600276.SH", "300015.SZ", "002714.SZ", "300529.SZ", "601138.SH",
]


def build_universe() -> list[str]:
    names = set()
    for lst in SEED.values():
        names.update(lst)
    names.update(A2US_PEERS.keys())
    names.update(EXTRA)
    return sorted(names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=800)
    ap.add_argument("--horizon", type=int, default=5, help="label = h-day fwd return")
    ap.add_argument("--quantile", type=float, default=0.2)
    ap.add_argument("--cost", type=float, default=0.0015, help="per-side turnover cost")
    ap.add_argument("--hold-days", type=int, default=1, dest="hold_days",
                    help="rebalance every N sessions (holding period); cuts turnover ~Nx")
    ap.add_argument("--no-norm", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="ignore cached panel")
    args = ap.parse_args()

    uni = build_universe()
    md = MarketData(choice=ChoiceClient())     # offline: free klines only

    # panel cache (free data pull is slow under throttling) — reuse if <1 day old
    cache = ROOT / "data" / f"panel_{len(uni)}_{args.bars}.pkl"
    cache.parent.mkdir(exist_ok=True)
    panel = pd.DataFrame()
    if not args.refresh and cache.exists() and (time.time() - cache.stat().st_mtime) < 86400:
        panel = pd.read_pickle(cache)
        print(f"Universe: {len(uni)} names. Loaded cached panel {panel.shape} "
              f"({(time.time()-cache.stat().st_mtime)/3600:.1f}h old; --refresh to repull)")
    if panel.empty:
        print(f"Universe: {len(uni)} names. Pulling ~{args.bars} daily bars (free data)...")
        t0 = time.time()
        panel = md.klines(uni, bars=args.bars)
        if panel.empty:
            print("No data (free endpoints throttled?). Try again shortly."); return
        panel.to_pickle(cache)
        print(f"  got {panel.index.get_level_values(0).nunique()} names in {time.time()-t0:.0f}s (cached)")

    print("Computing Alpha158 features + label ...")
    X, y = make_features(panel, label_horizon=args.horizon)
    print(f"  feature matrix: {X.shape[0]} rows x {X.shape[1]} features")

    print("Training LightGBM (Qlib Alpha158 params), evaluating IC/ICIR ...")
    res = fit_predict(X, y, normalize=not args.no_norm)
    bt = backtest_long_short(res.pred, panel, quantile=args.quantile,
                             horizon=args.horizon, cost=args.cost, t_plus_1=True,
                             hold_days=args.hold_days)

    m = res.metrics
    print("\n" + "=" * 66)
    print(f"  QLIB-STYLE RESULTS  (Alpha158 + LightGBM, {args.horizon}d label)")
    print("=" * 66)
    print(f"  IC        {m.get('IC', float('nan')):+.4f}     "
          f"RankIC   {m.get('RankIC', float('nan')):+.4f}")
    print(f"  ICIR      {m.get('ICIR', float('nan')):+.4f}     "
          f"RankICIR {m.get('RankICIR', float('nan')):+.4f}")
    print(f"  test days {m.get('n_days', 0)}   samples {m.get('n_samples', 0)}")
    if bt:
        g, n, lo = bt["long_short_gross"], bt["long_short_net"], bt["long_only_net"]
        print("  " + "-" * 62)
        print(f"  Backtest: T+1 lag, {bt['cost_bps']:.0f}bps/side, hold {bt['hold_days']}d "
              f"({bt['rebalances']} rebalances), avg turnover {bt['avg_turnover']:.2f}/day")
        print(f"  Long-Short GROSS (top/bot {args.quantile:.0%}): "
              f"ann {g['ann_return']:+.1%}  IR {g['ir']:.2f}  MaxDD {g['maxdd']:.1%}")
        print(f"  Long-Short NET   (after costs):   "
              f"ann {n['ann_return']:+.1%}  IR {n['ir']:.2f}  MaxDD {n['maxdd']:.1%}  hit {n['hit']:.0%}")
        print(f"  Long-only  NET   (top {args.quantile:.0%}):      "
              f"ann {lo['ann_return']:+.1%}  IR {lo['ir']:.2f}  MaxDD {lo['maxdd']:.1%}")
        print(f"  cost drag: {bt['cost_drag_annual']:.1%}/yr")
    print("  " + "-" * 62)
    print("  Qlib CSI300 reference (LightGBM/Alpha158): IC 0.045, RankIC 0.047,")
    print("  ICIR 0.37, ann ~9%, IR ~1.0  (300 names vs our small cross-section)")
    print("\n  Top 12 features by importance:")
    for name, imp in res.importance.head(12).items():
        print(f"    {name:10s} {int(imp)}")
    print()


if __name__ == "__main__":
    main()
