# ashare_quant — A-Share Tech & Supply-Chain Alpha Engine

> A regime-gated, sentiment-accelerated **short-term reversal** book for the
> volatile, high-PE corners of the Chinese A-share market — semiconductors,
> AI compute, the NEV / battery / photovoltaic supply chains, and consumer
> electronics. Built on the **East Money Choice (东方财富 Choice)** data API for
> fundamentals/history and on East Money's free real-time endpoints for live
> sentiment, attention and smart-money flow.

This document explains **what the strategy is, why each piece exists, and how to
run it** — written so you can audit every decision, not just execute it.

---

## 0. TL;DR

- **Edge:** A-share turnover is ~2/3 retail. Retail over-reacts to 1–5 day
  moves and to news/attention spikes. That over-reaction mean-reverts. We
  systematically **buy the over-sold, fade the over-bought**, and **accelerate
  the signal with a live sentiment-inflection overlay** that the crowd can't see
  as cleanly as we can.
- **Book:** cross-sectional long/short (or long-only), **theme- and size-neutral**,
  40 names, 5% single-name cap, ≤30% per supply-chain, daily rebalance.
- **Risk governor:** a CSI-300 **volatility-regime gate** scales gross exposure
  from 100% (calm uptrend) down to 30% (high-vol downtrend). This is the single
  most important line of defense in a market that limit-downs in clusters.
- **What runs today, no credentials:** the entire sentiment/attention/flow stack
  and the factor math. **What needs your Choice login:** historical prices,
  fundamentals (PE / mktcap), northbound holdings, and the backtest on real data.

---

## 1. Market thesis — why *this* market, *this* strategy

If you trade A-share tech like it's the S&P, you lose. The microstructure is
different and the differences *are* the alpha:

| A-share reality | Consequence | How we exploit it |
|---|---|---|
| ~60–70% retail turnover | Prices over-shoot on emotion, then revert | **Short-term reversal** is the dominant, most robust anomaly → largest weight |
| T+1 settlement, 10%/20% price limits | Can't scalp intraday; gaps and limit-locks | We hold ≥1 day, respect a 1-day implementation lag, and *fade* limit-up euphoria |
| Retail chases "lottery" names | High-vol / high-MAX stocks are over-bid, underperform | **Short idiosyncratic-vol** and **turnover-heat** sleeves |
| Attention is measurable (股吧 人气榜) | Crowding is observable in near-real-time | **Attention factor** + **euphoria-divergence** short tell |
| News moves retail fast | Sentiment *inflection* leads price | **Sentiment-delta** overlay from live news scoring |
| Northbound / institutions are "smart money" | Their flow front-runs fundamentals | **Main-force net-inflow** and northbound factors |

Classic value and 6–12m momentum are **weak and unstable** here, so we down-weight
them (momentum is off by default; see `config.example.yaml`).

**The named strategy:** *Sentiment-Accelerated Short-Term Reversal with
Volatility-Regime Gating.* Reversal is the engine; sentiment inflection is the
turbo; the regime gate is the brake.

---

## 2. The alpha, factor by factor

Every factor is **cross-sectional** (scored relative to peers on the day) and
signed so **higher = more attractive to LONG**. They are winsorized (2.5% tails),
**neutralized against theme + log-market-cap**, z-scored, then combined with the
weights in `config.yaml`.

| Factor | Sign | What it measures | Why it pays in A-share | Default weight |
|---|---|---|---|---|
| `st_reversal` | − of 1–5d return | Recent over-sold-ness | Retail over-reaction reverts hardest at short horizon | **0.28** |
| `sentiment_delta` | + | Change in fused news+flow sentiment | Inflection leads price; *level* is already priced | **0.22** |
| `ivol_lottery` | − of 20d vol | "Lottery" demand | Gamblers over-pay for high-vol names → they underperform | −0.14 |
| `turnover_heat` | − of 5d/20d turnover | Over-heating / exhaustion | Turnover blow-offs mark local tops | −0.12 |
| `northbound_flow` | + | Smart-money accumulation | Institutional flow front-runs fundamentals | 0.12 |
| `eps_revision` | + | Analyst growth-estimate upgrades | Estimate revisions drift | 0.12 |
| `residual_mom` | + | 60d momentum (skip 5d) | Weak here → **off by default** | 0.00 |

`divergence` (attention high **and** sentiment/flow negative) is carried as a
**risk flag**, not summed into the composite — it flags euphoric names that are
being distributed into, prime short candidates.

**Composite → book:** rank by `alpha`, take the top/bottom names, rank-weight each
leg, apply caps, scale by the regime multiplier, and shrink toward yesterday's
book so we don't churn the edge away in costs (T+1, stamp duty 0.05% sell-side,
commission + impact ≈ 15 bps modeled).

---

## 3. The sentiment engine (the differentiator)

Three **live, free, no-auth** East Money sources, fused per name:

1. **News** (`search-api-web.eastmoney.com`) — most-recent dated headlines per
   name, scored by a curated **Chinese finance-sentiment lexicon** with negation
   and intensifier handling (`quant/sentiment/lexicon.py`), recency-weighted
   (24h half-life). Transparent and fast — right for HF scoring of short posts.
2. **Retail attention** (`emappdata` 股吧人气榜 popularity rank) — East Money's own
   eyeball index. Rising rank = crowding = the fuel for over-reaction.
3. **Smart-money flow** (`push2` `f184` main-force net-inflow %) — large-order net
   buying as a share of turnover.

Fused into per-name **`sentiment`** (level), **`sentiment_delta`** (the tradeable
inflection, computed vs a rolling on-disk cache), **`attention`**, and
**`divergence`**. The *delta* is what enters the alpha, because the crowd has
already priced the *level*.

> Guba's raw *post* API is bot-gated, so we deliberately use the structured
> attention + flow signals above instead of scraping fragile HTML. They are
> cleaner and harder for others to replicate at scale.

---

## 4. Architecture

```
ashare_quant/
├── config.example.yaml     # every knob, documented — copy to config.yaml
├── setup_env.sh            # one-shot: register API, clear quarantine, deps, config
├── requirements.txt
├── quant/
│   ├── config.py           # yaml + env-override loader (secrets never on disk)
│   ├── choice_client.py    # robust pandas-returning wrapper over EmQuantAPI
│   ├── data.py             # Choice history/snapshot  +  free push2 realtime
│   ├── universe.py         # theme → sector → liquidity/size/PE-gated universe
│   ├── factors.py          # cross-sectional factor library (pure pandas)
│   ├── signals.py          # winsorize → neutralize → z → weighted composite
│   ├── regime.py           # CSI-300 vol/trend regime → gross-exposure multiplier
│   ├── portfolio.py        # caps, theme limits, T+1 & cost-aware construction
│   ├── backtest.py         # vectorized daily backtester (price sleeve)
│   ├── strategy.py         # end-to-end orchestrator → TargetBook + blotter
│   └── sentiment/
│       ├── lexicon.py          # Chinese finance sentiment scorer
│       ├── news_scraper.py     # async news + lexicon
│       ├── guba_scraper.py     # async attention (人气榜) + main-force flow
│       └── sentiment_engine.py # fuse → sentiment / delta / divergence
├── scripts/
│   ├── run_daily.py        # produce today's target book + order blotter
│   ├── run_sentiment.py    # LIVE sentiment scan (works with no Choice login)
│   └── run_backtest.py     # backtest (synthetic smoke-test when offline)
└── tests/test_core.py      # offline unit tests
```

**Data-flow (daily):** `login → universe → history+snapshot → factors → live
sentiment → alpha → regime → construct → target weights + blotter`, persisted to
`data/book_YYYY-MM-DD.json` so the next day computes turnover-aware orders.

---

## 5. Setup

```bash
cd ~/Downloads/EMQuantAPI_Python/ashare_quant
bash setup_env.sh
```

`setup_env.sh` does four things (all already verified on this machine):

1. **Registers** `EmQuantAPI` on the Python path (`installEmQuantAPI.py` writes a
   `.pth`) so `from EmQuantAPI import c` works anywhere.
2. **Clears macOS quarantine** on the native libs. *This was required here* —
   the freshly-downloaded `libEMQuantAPIx64.dylib` was blocked by Gatekeeper
   (“library load disallowed by system policy”); `xattr -dr com.apple.quarantine`
   fixes it. After the fix, `c.start()` loads the lib and reaches the auth layer.
3. **Installs** Python deps (pandas, numpy, scikit-learn, aiohttp, jieba, …).
4. **Creates** `config.yaml` from the template.

### Authenticating Choice (the one thing only you can do)
`c.start()` currently returns `10001014: need account activation` — expected,
because no Choice credentials are wired in yet. Do **one** of:

- Put your East Money Choice login in `config.yaml` → `choice.username/password`
  (or `export EM_USERNAME=… EM_PASSWORD=…`), **or**
- Activate once with the bundled GUI tool:
  `~/Downloads/EMQuantAPI_Python/python3/libs/mac/loginactivator_mac`.

> Note: the environment’s Python runs x86_64 under Rosetta; the dylib is a
> universal binary, so the x86_64 slice loads fine. No arch change needed.

---

## 6. Running it

```bash
# LIVE sentiment scan — works RIGHT NOW, no Choice login:
python3 scripts/run_sentiment.py --seed
python3 scripts/run_sentiment.py 300750.SZ 688981.SH 300308.SZ

# Full daily book (uses Choice if creds present; sentiment-only if offline):
python3 scripts/run_daily.py                # prints target weights + blotter
python3 scripts/run_daily.py --no-sentiment # factor sleeve only

# Backtest the price/fundamental sleeve (needs Choice for real history):
python3 scripts/run_backtest.py --start 2024-01-01 --end 2025-06-30

# Offline unit tests:
python3 tests/test_core.py
```

`run_sentiment.py` output (verified live) ranks names by fused sentiment, flags
**bullish inflections** (top `sentiment_delta`) and **euphoria-divergence shorts**
(hot attention, weak flow/news).

---

## 7. Decision log — every non-obvious choice and why

- **Reversal over momentum as the core.** In retail-dominated A-share, the
  short-horizon reversal anomaly is far more robust than price momentum. It gets
  the top weight; momentum is off by default.
- **Sentiment *delta*, not level.** The level is largely priced. The change —
  fresh news turning positive while a name is still depressed and smart money
  starts to accumulate — is the un-priced, tradeable part.
- **Theme + size neutralization.** Without it the book would just be a levered
  bet on whichever supply-chain is hot and on small-caps. We want *within-peer*
  stock selection, so we regress those tilts out before combining factors.
- **Regime gate on gross, not on names.** You can be right on the cross-section
  and still blow up if you're 100% gross into a vol storm. Scaling gross by a
  simple, robust 2×2 (vol percentile × trend sign) is the highest-Sharpe risk
  control available for this market. Down-trend + high-vol → 30% gross.
- **Turnover shrink toward prior book.** Daily reversal signals are noisy; churning
  fully every day donates the edge to brokers and the tax man. We only move when
  the expected edge clears modeled costs.
- **Free endpoints for sentiment, Choice for fundamentals.** Sentiment must be
  high-frequency and cheap; the free push2/search/人气榜 APIs are perfect and need
  no auth. Choice is the authoritative source for adjusted history, PE, mktcap
  and northbound — used where accuracy matters more than latency.
- **Lexicon, not a transformer, for sentiment.** Auditable, millisecond-cheap over
  thousands of short slangy posts, and easy to tune. `score_text()` is a drop-in
  seam if you later want an embedding model.
- **Backtest = price sleeve only.** Free sentiment is point-in-time and can't be
  reconstructed historically, so backtesting it would be look-ahead. We validate
  the sentiment sleeve *forward* (paper/live) and layer it on a clean factor
  backtest. Honest > flattering.

---

## 8. Risk management

- **Gross gate:** regime multiplier 1.0 / 0.7 / 0.6 / 0.3 (calm-up → storm-down).
- **Concentration:** ≤5% per name, ≤30% per theme, ~40 names.
- **Liquidity gate:** min 20-day turnover floor so the book is exitable under T+1.
- **Quality gate:** drop ST/*ST, micro-caps, and names halted too often.
- **Divergence flag:** euphoric-but-distributed names surfaced explicitly.
- **Cost model:** ~15 bps/side (commission + 0.05% sell stamp + impact) charged in
  both construction (turnover shrink) and backtest.

---

## 9. Going to production (live trading path)

This repo produces **target weights and an order blotter**; it does **not** place
orders. To trade for real you would add an execution adapter:

1. **Broker/API:** A-share retail can't use Choice to trade — route via a broker
   that offers an order API (e.g. QMT/miniQMT, Ptrade, or a licensed broker FIX).
2. **Execution layer:** translate `blotter()` deltas into limit orders with
   participation caps (e.g. ≤10% of volume), respect price limits, and handle T+1
   (never sell today's buys).
3. **Live loop:** run `run_strategy()` near the close (14:30–14:55 CST) so signals
   use the full day, place next-open or VWAP orders.
4. **Monitoring:** log fills, slippage vs modeled cost, realized vs predicted
   factor returns; alert on regime flips.

---

## 10. Honest limitations

- **No live trading here** — signals + blotter only (by design).
- **Backtest excludes the sentiment sleeve** (look-ahead avoidance).
- **Sector codes** in `universe.SECTOR_MAP` are East Money thematic-index ids and
  should be reconciled against your Choice terminal's current 板块 list.
- **Sentiment lexicon** is curated, not learned — extend `BULLISH/BEARISH` and/or
  swap in an embedding model behind `score_text()` for more coverage.
- **Free endpoints** are undocumented and can change/rate-limit; the code degrades
  gracefully (empty → contributes nothing) rather than crashing.
- **This is research tooling, not investment advice.** Paper-trade, measure, and
  size to your own risk before committing capital.

---

## 11. Roadmap

- Northbound intraday flow factor (once your Choice tier exposes it).
- Learned sentiment model + per-theme lexicons.
- Intraday reversal sleeve using Choice `cst` tick / `csq` real-time subscribe.
- Portfolio optimizer (risk-model covariance) replacing rank-weighting.
- Walk-forward factor-weight fitting with purged CV.
