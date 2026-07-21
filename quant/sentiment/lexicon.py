"""Chinese financial-sentiment lexicon scorer.

A lightweight, dependency-free (jieba-optional) sentiment model tuned for
retail A-share chatter (股吧 / 雪球 / news headlines). It is intentionally a
*lexicon + rules* model rather than a heavy transformer because:

* it must run at high frequency over thousands of short posts, cheaply;
* retail posts are slangy and emoji-laden — a curated finance lexicon with
  negation + intensifier handling beats a generic sentiment net here;
* it is transparent and auditable (every score decomposes into hit words).

Score is bounded to [-1, +1] per document; the engine aggregates across docs.
Swap in a transformer in ``score_text`` later without touching callers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

try:
    import jieba
    jieba.setLogLevel(60)
    _HAS_JIEBA = True
except Exception:  # pragma: no cover
    _HAS_JIEBA = False

# --- curated finance lexicon (weights are prior magnitudes) -----------------
BULLISH = {
    "涨停": 2.0, "涨": 1.0, "大涨": 1.6, "拉升": 1.3, "突破": 1.2, "新高": 1.4,
    "利好": 1.5, "买入": 1.2, "加仓": 1.3, "满仓": 1.2, "看多": 1.4, "看好": 1.3,
    "牛": 1.2, "牛市": 1.5, "反弹": 1.0, "回暖": 1.0, "放量": 0.8, "主力": 0.5,
    "增持": 1.2, "回购": 1.1, "中标": 1.2, "订单": 0.9, "超预期": 1.6, "业绩": 0.4,
    "龙头": 0.9, "翻倍": 1.6, "起飞": 1.4, "yyds": 1.2, "冲": 0.9, "干": 0.6,
    "强势": 1.1, "机会": 0.7, "低吸": 0.8, "布局": 0.7, "景气": 1.1, "放量拉升": 1.6,
}
BEARISH = {
    "跌停": 2.0, "跌": 1.0, "大跌": 1.6, "暴跌": 1.9, "跳水": 1.6, "破位": 1.4,
    "利空": 1.5, "卖出": 1.2, "减持": 1.3, "清仓": 1.4, "看空": 1.4, "看跌": 1.3,
    "熊": 1.2, "熊市": 1.5, "套牢": 1.3, "被套": 1.3, "割肉": 1.5, "亏": 1.1,
    "亏损": 1.4, "爆雷": 1.9, "退市": 2.0, "商誉": 0.8, "质押": 0.7, "解禁": 1.1,
    "出货": 1.2, "诱多": 1.3, "阴跌": 1.2, "跑路": 1.5, "垃圾": 1.2, "骗": 1.1,
    "崩": 1.5, "凉": 1.0, "深套": 1.4, "站岗": 1.0, "闷杀": 1.5, "低于预期": 1.5,
}
NEGATIONS = {"不", "没", "无", "别", "非", "未", "难", "反", "假"}
INTENSIFIERS = {"很": 1.4, "非常": 1.6, "超": 1.5, "巨": 1.7, "暴": 1.8, "狂": 1.7,
                "大": 1.3, "重": 1.3, "太": 1.4}

_TOKEN_RE = re.compile(r"[一-鿿A-Za-z]+")


def _tokens(text: str) -> list[str]:
    if _HAS_JIEBA:
        return [t for t in jieba.lcut(text) if t.strip()]
    # Fallback: char-bigrams over CJK runs (crude but works for lexicon hits).
    out: list[str] = []
    for run in _TOKEN_RE.findall(text):
        out.append(run)
        out.extend(run[i:i + 2] for i in range(len(run) - 1))
    return out


@dataclass
class DocScore:
    polarity: float      # [-1, 1]
    magnitude: float     # total absolute weight (confidence proxy)
    n_bull: int
    n_bear: int


def score_text(text: str) -> DocScore:
    if not text:
        return DocScore(0.0, 0.0, 0, 0)
    toks = _tokens(text)
    tokset = set(toks)
    pos = neg = 0.0
    nb = nr = 0
    for i, tok in enumerate(toks):
        w = BULLISH.get(tok, 0.0) - BEARISH.get(tok, 0.0)
        if w == 0.0:
            continue
        # look back up to 2 tokens for negation / intensifier
        mult = 1.0
        flip = False
        for j in (i - 1, i - 2):
            if j < 0:
                break
            if toks[j] in NEGATIONS:
                flip = True
            if toks[j] in INTENSIFIERS:
                mult *= INTENSIFIERS[toks[j]]
        val = w * mult * (-1.0 if flip else 1.0)
        if val > 0:
            pos += val
            nb += 1
        else:
            neg += -val
            nr += 1
    total = pos + neg
    polarity = 0.0 if total == 0 else (pos - neg) / total
    # squash magnitude into a confidence-ish [0,1]
    magnitude = 1.0 - 1.0 / (1.0 + total)
    _ = tokset
    return DocScore(round(polarity, 4), round(magnitude, 4), nb, nr)
