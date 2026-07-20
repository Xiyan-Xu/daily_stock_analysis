#!/usr/bin/env python3
"""
backtest_llm.py — LLM-powered backtest that mirrors the actual workflow.

与 backtest.py（规则打分）的区别：
  - 用 LLM 解读技术数据，而非固定公式
  - 输出与真实 pipeline 完全相同的 Decision Dashboard JSON schema
  - 包含价格目标（止损/目标价）、可解释性（analysis_summary/key_points）
  - 额外追踪止损位被穿破的概率

与真实 workflow 仍有的差距（用户要求忽略）：
  - 无 IntelAgent（无新闻/情绪数据）
  - 无 RiskAgent（无风险筛查）
  - 无市场宏观背景

成本估算（gpt-4o-mini）：
  9 只股票 × ~36 个月度样本 = ~324 次 LLM 调用 ≈ $0.10-0.15

运行方法：
  python backtest_llm.py
  OPENAI_API_KEY 从 .env 文件或环境变量读取
"""

import json
import os
import time
import warnings
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Config ─────────────────────────────────────────────────────────────────────

STOCKS = ["VOO", "MSFT", "NVDA", "MSTR", "QQQ", "GOOGL", "MU", "DRAM", "LITE"]
LOOKBACK_DAYS    = 730   # 2 年回测
SAMPLE_INTERVAL  = 20    # 每 20 个交易日取一个信号点（约每月一次）
FORWARD_WINDOWS  = [10, 20, 60]
MIN_HISTORY_BARS = 80
LLM_DELAY_S      = 0.8   # API 调用间隔（避免 rate limit）

MODEL       = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
API_KEY     = os.environ.get("OPENAI_API_KEY", "")

# ── Technical Indicators ───────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    v = (100 - 100 / (1 + rs)).iloc[-1]
    return round(float(v), 1) if not np.isnan(v) else 50.0


def _macd(close: pd.Series) -> dict:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    sig  = line.ewm(span=9, adjust=False).mean()
    hist = line - sig
    return {
        "macd":      round(float(line.iloc[-1]), 4),
        "signal":    round(float(sig.iloc[-1]),  4),
        "histogram": round(float(hist.iloc[-1]), 4),
        "hist_prev": round(float(hist.iloc[-2]), 4) if len(hist) > 1 else 0.0,
    }


def _bollinger(close: pd.Series, period: int = 20) -> dict:
    ma  = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = float((ma + 2 * std).iloc[-1])
    lower = float((ma - 2 * std).iloc[-1])
    mid   = float(ma.iloc[-1])
    price = float(close.iloc[-1])
    pct_b = (price - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
    return {"upper": round(upper, 2), "middle": round(mid, 2),
            "lower": round(lower, 2), "pct_b": round(pct_b, 2)}


def _swing_levels(df: pd.DataFrame, lookback: int = 60) -> dict:
    """Identify support/resistance from recent pivot highs/lows."""
    recent = df.tail(lookback)
    price  = float(df["Close"].iloc[-1])

    pivot_highs, pivot_lows = [], []
    highs = recent["High"].values
    lows  = recent["Low"].values
    for i in range(2, len(recent) - 2):
        if highs[i] == max(highs[i-2:i+3]):
            pivot_highs.append(float(highs[i]))
        if lows[i] == min(lows[i-2:i+3]):
            pivot_lows.append(float(lows[i]))

    supports    = sorted([p for p in pivot_lows  if p < price], reverse=True)
    resistances = sorted([p for p in pivot_highs if p > price])

    return {
        "support":      round(supports[0],    2) if supports    else round(price * 0.95, 2),
        "resistance":   round(resistances[0], 2) if resistances else round(price * 1.05, 2),
        "support_2":    round(supports[1],    2) if len(supports)    > 1 else None,
        "resistance_2": round(resistances[1], 2) if len(resistances) > 1 else None,
    }


def build_technical_context(df: pd.DataFrame, ticker: str, signal_date: str) -> tuple:
    """
    Compute technical indicators and format them as a rich context string —
    mirroring what TechnicalAgent's tool calls (get_daily_history, analyze_trend,
    calculate_ma, get_volume_analysis, etc.) would return to the LLM.
    """
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]
    price  = round(float(close.iloc[-1]), 2)

    # Moving averages
    ma5   = round(float(close.rolling(5).mean().iloc[-1]),   2)
    ma10  = round(float(close.rolling(10).mean().iloc[-1]),  2)
    ma20  = round(float(close.rolling(20).mean().iloc[-1]),  2)
    ma50  = round(float(close.rolling(50).mean().iloc[-1]),  2) if len(close) >= 50  else None
    ma200 = round(float(close.rolling(200).mean().iloc[-1]), 2) if len(close) >= 200 else None

    ma_align = ("bullish" if ma5 > ma10 > ma20 else
                "bearish" if ma5 < ma10 < ma20 else "mixed")
    bias_ma20 = round((price / ma20 - 1) * 100, 1)

    # Volume
    avg_vol = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio  = round(float(volume.iloc[-1]) / avg_vol, 2) if avg_vol > 0 else 1.0
    vol_status = "heavy" if vol_ratio > 1.5 else ("light" if vol_ratio < 0.7 else "normal")

    # RSI
    rsi_val  = _rsi(close)
    rsi_zone = ("oversold" if rsi_val < 30 else "overbought" if rsi_val > 70 else "neutral")

    # MACD
    m = _macd(close)
    macd_dir = ("bullish_accel"  if m["histogram"] > 0 and m["histogram"] > m["hist_prev"] else
                "bullish_decel"  if m["histogram"] > 0 else
                "bearish_accel"  if m["histogram"] < 0 and m["histogram"] < m["hist_prev"] else
                "bearish_decel")

    # Bollinger
    bb = _bollinger(close)

    # Swing levels
    levels = _swing_levels(df)

    # 52-week context
    w52h = round(float(high.tail(252).max()), 2)
    w52l = round(float(low.tail(252).min()),  2)
    pct_from_high = round((price / w52h - 1) * 100, 1)
    pct_from_low  = round((price / w52l - 1) * 100, 1)

    # Short-term returns
    ret_5d  = round((price / float(close.iloc[-6])  - 1) * 100, 1) if len(close) > 5  else None
    ret_20d = round((price / float(close.iloc[-21]) - 1) * 100, 1) if len(close) > 20 else None

    # Recent 10-session OHLCV table
    recent_rows = []
    for dt, row in df.tail(10).iterrows():
        chg = round((float(row["Close"]) / float(row["Open"]) - 1) * 100, 1)
        recent_rows.append(
            f"  {dt.strftime('%Y-%m-%d')}  "
            f"O={row['Close']:.2f}  H={row['High']:.2f}  "
            f"L={row['Low']:.2f}  C={row['Close']:.2f}  "
            f"chg={chg:+.1f}%  V={int(row['Volume']):>12,}"
        )

    ctx = f"""=== Technical Data: {ticker}  |  Analysis date: {signal_date} ===

[Price Action]
Current Price : {price}
5d Return     : {ret_5d}%
20d Return    : {ret_20d}%
52w High      : {w52h}  ({pct_from_high}% from high)
52w Low       : {w52l}  (+{pct_from_low}% from low)

[Moving Averages]
MA5={ma5}  MA10={ma10}  MA20={ma20}  MA50={ma50 or 'N/A'}  MA200={ma200 or 'N/A'}
Alignment  : {ma_align.upper()}
Bias vs MA20: {bias_ma20:+.1f}%

[Momentum Indicators]
RSI(14)        : {rsi_val}  ({rsi_zone.upper()})
MACD Line      : {m['macd']}
MACD Signal    : {m['signal']}
MACD Histogram : {m['histogram']}  prev={m['hist_prev']}  → {macd_dir.upper()}

[Volatility — Bollinger Bands(20,2)]
Upper : {bb['upper']}   Middle : {bb['middle']}   Lower : {bb['lower']}
%B    : {bb['pct_b']}   (>0.8 = overbought zone, <0.2 = oversold zone)

[Volume]
Volume Ratio (vs 20d avg) : {vol_ratio}x  → {vol_status.upper()}

[Key Price Levels — Swing Analysis]
Immediate Support   : {levels['support']}
Secondary Support   : {levels['support_2'] or 'N/A'}
Immediate Resistance: {levels['resistance']}
Secondary Resistance: {levels['resistance_2'] or 'N/A'}

[Recent 10 Sessions]
{"".join(chr(10) + r for r in recent_rows)}
""".strip()

    return ctx, levels


# ── System Prompt (mirrors TechnicalAgent + DecisionAgent combined) ────────────

SYSTEM_PROMPT = """\
You are a quantitative Technical Analysis and Decision Agent for US equities.

You receive pre-computed technical indicator data for a stock at a specific
historical date. Your task: interpret the data and produce a complete investment
decision in the exact same JSON format used by a production multi-agent trading system.

## Signal Weighting
- MA alignment + trend : ~40%
- RSI + MACD momentum  : ~35%
- Volume confirmation  : ~15%
- Pattern / structure  : ~10%

## Scoring Rules
- 80-100 → buy   (strong conviction, all signals aligned)
- 60-79  → buy   (positive, minor caveats)
- 40-59  → hold  (mixed signals, wait for confirmation)
- 20-39  → sell  (negative trend, elevated risk)
- 0-19   → sell  (major technical breakdown)

decision_type MUST be consistent with sentiment_score:
  sentiment_score ≥ 60 → "buy"
  sentiment_score 40-59 → "hold"
  sentiment_score ≤ 39 → "sell"

## Price Level Rules
- stop_loss  : MUST be below current price (for buy signals: near immediate support)
- take_profit: for buy signals, near resistance; null for sell signals
- Be conservative with stop_loss — place it where the thesis is invalidated

## Output Format
Return ONLY a valid JSON object with these exact keys:
{
  "sentiment_score": <int 0-100>,
  "decision_type": "buy|hold|sell",
  "confidence_level": "高|中|低",
  "analysis_summary": "<2-3 sentence summary in Chinese>",
  "trend_prediction": "<short-to-medium term trend outlook>",
  "operation_advice": "<specific actionable advice in Chinese>",
  "key_levels": {
    "support": <float>,
    "resistance": <float>,
    "stop_loss": <float>,
    "take_profit": <float or null>
  },
  "key_points": ["<point1>", "<point2>", "<point3>"],
  "risk_warning": "<primary technical risk in Chinese>",
  "ma_alignment": "bullish|neutral|bearish",
  "trend_score": <int 0-100>
}
"""


def call_llm(context: str) -> Optional[dict]:
    """Send technical context to LLM, parse and return the decision JSON."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed — run: pip install openai")

    client = OpenAI(api_key=API_KEY)
    user_msg = (
        "Analyse the following technical data and produce the decision JSON.\n"
        "Do not add any text outside the JSON object.\n\n"
        + context
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=900,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as exc:
        print(f"    [LLM ERROR] {exc}")
        return None


# ── Base Rate ─────────────────────────────────────────────────────────────────

def compute_base_rates(data: pd.DataFrame, test_start: pd.Timestamp) -> dict:
    close = data[data.index >= test_start]["Close"]
    rates = {}
    for fwd in FORWARD_WINDOWS:
        pos = sum(
            1 for i in range(len(close) - fwd)
            if float(close.iloc[i + fwd]) > float(close.iloc[i])
        )
        total = max(len(close) - fwd, 1)
        rates[fwd] = round(pos / total * 100, 1)
    return rates


# ── Backtest Engine ───────────────────────────────────────────────────────────

def backtest_stock(ticker: str, data: pd.DataFrame) -> tuple:
    test_start = data.index[-1] - pd.Timedelta(days=LOOKBACK_DAYS)
    base_rates  = compute_base_rates(data, test_start)
    test_dates  = data.index[data.index >= test_start][::SAMPLE_INTERVAL]

    rows = []
    for i, signal_date in enumerate(test_dates):
        loc = data.index.get_loc(signal_date)
        df_slice = data.iloc[:loc + 1]
        if len(df_slice) < MIN_HISTORY_BARS:
            continue

        date_str = signal_date.strftime("%Y-%m-%d")
        print(f"  [{i+1:02d}/{len(test_dates)}] {date_str}", end=" ", flush=True)

        ctx, levels = build_technical_context(df_slice, ticker, date_str)
        result = call_llm(ctx)
        time.sleep(LLM_DELAY_S)

        if result is None:
            print("skip (LLM error)")
            continue

        signal       = result.get("decision_type", "hold")
        score        = int(result.get("sentiment_score", 50))
        signal_price = float(data["Close"].iloc[loc])
        kl           = result.get("key_levels", {})

        print(f"{signal.upper():4s} score={score:3d}  "
              f"stop={kl.get('stop_loss','?')}  tp={kl.get('take_profit','?')}")

        row = {
            "ticker":           ticker,
            "date":             signal_date.date(),
            "price":            round(signal_price, 2),
            "signal":           signal,
            "sentiment_score":  score,
            "confidence":       result.get("confidence_level", "中"),
            "ma_alignment":     result.get("ma_alignment", ""),
            "trend_score":      result.get("trend_score", 50),
            "stop_loss":        kl.get("stop_loss"),
            "take_profit":      kl.get("take_profit"),
            "support":          kl.get("support"),
            "resistance":       kl.get("resistance"),
            "analysis_summary": result.get("analysis_summary", ""),
            "risk_warning":     result.get("risk_warning", ""),
            "operation_advice": result.get("operation_advice", ""),
        }

        for fwd in FORWARD_WINDOWS:
            future_loc = loc + fwd
            if future_loc < len(data):
                future_price = float(data["Close"].iloc[future_loc])
                ret = (future_price / signal_price - 1) * 100
                row[f"ret_{fwd}d"] = round(ret, 2)
                row[f"ok_{fwd}d"]  = (
                    (signal == "buy"  and ret > 0) or
                    (signal == "sell" and ret < 0)
                ) if signal != "hold" else None

                # Did price ever touch the stop-loss level during the window?
                sl = row["stop_loss"]
                if signal == "buy" and sl is not None:
                    future_low = float(data["Low"].iloc[loc + 1: future_loc + 1].min())
                    row[f"sl_hit_{fwd}d"] = future_low < float(sl)
                else:
                    row[f"sl_hit_{fwd}d"] = None
            else:
                row[f"ret_{fwd}d"]  = None
                row[f"ok_{fwd}d"]   = None
                row[f"sl_hit_{fwd}d"] = None

        rows.append(row)

    return pd.DataFrame(rows), base_rates


# ── Terminal Output ───────────────────────────────────────────────────────────

def _rich():
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        return Console(), Table, box
    except ImportError:
        return None, None, None


def _row_cells(sub: pd.DataFrame, fwd: int, base: float, sig: str) -> list:
    valid = sub[sub[f"ok_{fwd}d"].notna()]
    if not len(valid):
        return ["—", "—", "—"]
    acc  = valid[f"ok_{fwd}d"].mean() * 100
    avg  = valid[f"ret_{fwd}d"].mean()
    lift = acc - (base if sig == "buy" else (100 - base))
    c  = "green" if acc >= 55 else ("yellow" if acc >= 45 else "red")
    lc = "green" if lift >= 5 else ("yellow" if lift >= 0 else "red")
    r  = "green" if avg > 0 else "red"
    return [f"[{c}]{acc:.0f}%[/{c}]",
            f"[{lc}]{lift:+.0f}pp[/{lc}]",
            f"[{r}]{avg:+.1f}%[/{r}]"]


def print_stock_summary(ticker: str, df: pd.DataFrame, base_rates: dict) -> None:
    console, Table, box = _rich()
    br_str = "  ".join(f"{fwd}d={base_rates.get(fwd,50):.0f}%" for fwd in FORWARD_WINDOWS)

    if console:
        console.print(f"\n[bold cyan]── {ticker} ──[/bold cyan]")
        console.print(f"  [dim]Random-buy baseline: {br_str}[/dim]")

        tbl = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
        tbl.add_column("Signal")
        tbl.add_column("N", justify="right")
        for fwd in FORWARD_WINDOWS:
            tbl.add_column(f"Acc {fwd}d",    justify="right")
            tbl.add_column(f"Lift {fwd}d",   justify="right")
            tbl.add_column(f"AvgRet {fwd}d", justify="right")

        for sig in ["buy", "hold", "sell"]:
            sub = df[df["signal"] == sig]
            cells = [sig.upper(), str(len(sub))]
            if sig == "hold":
                cells += ["—", "—", "—"] * len(FORWARD_WINDOWS)
            else:
                for fwd in FORWARD_WINDOWS:
                    cells += _row_cells(sub, fwd, base_rates.get(fwd, 50.0), sig)
            tbl.add_row(*cells)
        console.print(tbl)

        # Stop-loss breach stats
        buy_df = df[df["signal"] == "buy"]
        for fwd in FORWARD_WINDOWS:
            valid = buy_df[buy_df[f"sl_hit_{fwd}d"].notna()]
            if len(valid):
                rate = valid[f"sl_hit_{fwd}d"].mean() * 100
                color = "red" if rate > 30 else ("yellow" if rate > 15 else "green")
                console.print(f"  [dim]Stop-loss hit rate ({fwd}d): [{color}]{rate:.0f}%[/{color}][/dim]")
    else:
        print(f"\n=== {ticker} ===  (baseline: {br_str})")
        for sig in ["buy", "sell"]:
            sub = df[df["signal"] == sig]
            print(f"  {sig.upper()} ({len(sub)}):")
            for fwd in FORWARD_WINDOWS:
                valid = sub[sub[f"ok_{fwd}d"].notna()]
                if not len(valid):
                    continue
                acc  = valid[f"ok_{fwd}d"].mean() * 100
                avg  = valid[f"ret_{fwd}d"].mean()
                base = base_rates.get(fwd, 50.0)
                lift = acc - (base if sig == "buy" else (100 - base))
                print(f"    {fwd}d: acc={acc:.0f}%  lift={lift:+.0f}pp  avg_ret={avg:+.1f}%")


def print_portfolio_summary(results: dict, all_base_rates: dict) -> None:
    console, Table, box = _rich()
    all_rows = pd.concat([v[0] for v in results.values()], ignore_index=True)
    avg_base = {
        fwd: np.mean([all_base_rates[t][fwd] for t in all_base_rates
                      if fwd in all_base_rates[t]])
        for fwd in FORWARD_WINDOWS
    }

    br_str = "  ".join(f"{fwd}d={avg_base[fwd]:.0f}%" for fwd in FORWARD_WINDOWS)
    if console:
        console.rule("[bold]Portfolio Summary (all stocks combined)[/bold]")
        console.print(f"  [dim]Avg random baseline: {br_str}[/dim]")

        tbl = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
        tbl.add_column("Signal")
        tbl.add_column("Total N", justify="right")
        for fwd in FORWARD_WINDOWS:
            tbl.add_column(f"Acc {fwd}d",    justify="right")
            tbl.add_column(f"Lift {fwd}d",   justify="right")
            tbl.add_column(f"AvgRet {fwd}d", justify="right")

        for sig in ["buy", "sell"]:
            sub = all_rows[all_rows["signal"] == sig]
            cells = [sig.upper(), str(len(sub))]
            for fwd in FORWARD_WINDOWS:
                cells += _row_cells(sub, fwd, avg_base.get(fwd, 50.0), sig)
            tbl.add_row(*cells)
        console.print(tbl)
    else:
        print(f"\n=== PORTFOLIO SUMMARY ===  (baseline: {br_str})")
        for sig in ["buy", "sell"]:
            sub = all_rows[all_rows["signal"] == sig]
            print(f"  {sig.upper()} ({len(sub)}):")
            for fwd in FORWARD_WINDOWS:
                valid = sub[sub[f"ok_{fwd}d"].notna()]
                if not len(valid):
                    continue
                acc  = valid[f"ok_{fwd}d"].mean() * 100
                avg  = valid[f"ret_{fwd}d"].mean()
                base = avg_base.get(fwd, 50.0)
                lift = acc - (base if sig == "buy" else (100 - base))
                print(f"    {fwd}d: acc={acc:.0f}%  lift={lift:+.0f}pp  avg_ret={avg:+.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        print("ERROR: OPENAI_API_KEY not set.")
        print("  Option 1: export OPENAI_API_KEY=sk-...")
        print("  Option 2: add OPENAI_API_KEY=sk-... to your .env file")
        return

    console, _, _ = _rich()
    n_est = len(STOCKS) * (LOOKBACK_DAYS // SAMPLE_INTERVAL)
    header = (f"DSA LLM Backtester  |  model={MODEL}  |  "
              f"{LOOKBACK_DAYS}d lookback  |  ~{n_est} LLM calls")
    if console:
        console.print(f"\n[bold green]{header}[/bold green]")
        console.print(f"Stocks: {', '.join(STOCKS)}")
        console.print(f"Cost estimate: < $0.15 total with gpt-4o-mini\n")
    else:
        print(f"\n{header}")
        print(f"Stocks: {', '.join(STOCKS)}\n")

    fetch_start = (date.today() - timedelta(days=LOOKBACK_DAYS + 300)).strftime("%Y-%m-%d")
    results:         dict = {}
    all_base_rates:  dict = {}

    for ticker in STOCKS:
        print(f"\n{'─'*60}")
        print(f"Downloading {ticker} ...", end=" ", flush=True)
        try:
            raw = yf.download(ticker, start=fetch_start, auto_adjust=True, progress=False)
            if raw.empty:
                print("no data, skipping")
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            # Ensure High/Low columns exist
            if "High" not in raw.columns:
                raw["High"] = raw["Close"]
            if "Low" not in raw.columns:
                raw["Low"] = raw["Close"]
            print(f"{len(raw)} bars")

            df, base_rates = backtest_stock(ticker, raw)
            results[ticker]       = (df, base_rates)
            all_base_rates[ticker] = base_rates
            print_stock_summary(ticker, df, base_rates)

        except KeyboardInterrupt:
            print("\nStopped by user. Saving partial results...")
            break
        except Exception as exc:
            print(f"ERROR: {exc}")

    if not results:
        return

    print_portfolio_summary(results, all_base_rates)

    out = "backtest_llm_results.csv"
    combined = pd.concat([v[0] for v in results.values()], ignore_index=True)
    combined.to_csv(out, index=False)

    if console:
        console.print(f"\n[green]Saved → {out}[/green]")
        console.print("[dim]Columns include: signal, sentiment_score, stop_loss, "
                      "take_profit, analysis_summary (LLM reasoning)[/dim]")
    else:
        print(f"\nSaved → {out}")
        print("Columns: signal, sentiment_score, stop_loss, take_profit, analysis_summary")

    print("\nHow to read Lift:")
    print("  Lift = Accuracy - random baseline")
    print("  Lift > +5pp = signal has real edge beyond market drift")
    print("  Lift ≈ 0    = signal adds no value vs just buying randomly")
    print("  Lift < 0    = signal is WORSE than random (anti-predictive)")


if __name__ == "__main__":
    main()
