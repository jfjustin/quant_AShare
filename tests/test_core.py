"""Offline unit tests — no network, no Choice login required.

    cd ashare_quant && python -m pytest tests/ -v      (or: python tests/test_core.py)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.sentiment.lexicon import score_text
from quant import factors as F


def test_lexicon_polarity():
    assert score_text("涨停 突破 主力 利好 拉升 业绩超预期").polarity > 0.3
    assert score_text("跌停 爆雷 退市 割肉 gou庄 巨亏").polarity < -0.3
    # negation flips
    assert score_text("不看好").polarity < 0
    assert score_text("").polarity == 0.0


def test_reversal_sign():
    # a stock that fell should get a POSITIVE reversal score (buy the dip)
    dates = pd.bdate_range("2025-01-01", periods=10)
    faller = pd.Series(np.linspace(20, 10, 10), index=dates)   # falling
    riser = pd.Series(np.linspace(10, 20, 10), index=dates)    # rising
    frames = []
    for code, s in [("FALL.SZ", faller), ("RISE.SZ", riser)]:
        df = pd.DataFrame({"close": s})
        df.index = pd.MultiIndex.from_product([[code], s.index], names=["code", "date"])
        frames.append(df)
    hist = pd.concat(frames)
    rev = F.st_reversal(hist, lookback=5)
    assert rev["FALL.SZ"] > rev["RISE.SZ"]


def test_zscore_neutralize():
    s = pd.Series([1, 2, 3, 4, 100.0], index=list("abcde"))
    z = F.zscore(F.winsorize(s, 0.1))
    assert abs(z.mean()) < 1e-9
    by = pd.DataFrame({"theme": pd.Categorical(list("xxyyy"))}, index=list("abcde"))
    resid = F.neutralize(s, by)
    assert len(resid) == 5


if __name__ == "__main__":
    test_lexicon_polarity()
    test_reversal_sign()
    test_zscore_neutralize()
    print("ok — all core tests passed")
