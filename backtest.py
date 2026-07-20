#!/usr/bin/env python3
"""
backtest.py — Technical signal backtest for DSA watchlist.

Simulates TechnicalAgent's signal logic at weekly intervals over the past year
using only historical price/volume data (yfinance). Zero API keys needed.

Run:
    python backtest.py

Output:
    - Per-stock accuracy table printed to terminal
    - backtest_results.csv  (detailed per-signal records)
"""

import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

STOCKS = ["VOO", "MSFT", "NVDA", "MSTR", "QQQ", "GOOGL", "MU", "DRAM", "LITE"]
LOOKBACK_DAYS = 730        # how far back to test (2 years → more samples)
SAMPLE_INTERVAL = 5        # generate signal every N trading days
FORWARD_WINDOWS = [10, 20, 60, 120]  # longer windows capture trend-level moves
MIN_HISTORY_BARS = 60      # minimum bars needed before computing indicators

# ── Technical Indicators ──────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line - signal_line  # histogram


# ── Signal Logic (mirrors TechnicalAgent scoring) ────────────────────────────

def score_to_signal(score: float) -> str:
    if score >= 65:
        return "buy"
    if score <= 35:
        return "sell"
    return "hold"


def generate_signal(df_slice: pd.DataFrame) -> dict:
    """
    Score 0-100 from 4 components — higher = more bullish:
      MA alignment  : 0-40
      RSI zone      : 0-30
      MACD momentum : 0-20
      Volume conf.  : 0-10
    """
    close = df_slice["Close"]
    volume = df_slice["Volume"]

    if len(close) < MIN_HISTORY_BARS:
        return {"signal": "hold", "score": 50, "rsi": None, "ma_alignment": "unknown"}

    price = close.iloc[-1]
    ma5  = close.rolling(5).mean().iloc[-1]
    ma10 = close.rolling(10).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]

    ma_bullish = bool(ma5 > ma10 > ma20)
    ma_bearish = bool(ma5 < ma10 < ma20)

    if ma_bullish:
        ma_score = 35 if price > ma20 else 22
    elif ma_bearish:
        ma_score = 5 if price < ma20 else 15
    else:
        ma_score = 18

    rsi_series = compute_rsi(close)
    rsi = float(rsi_series.iloc[-1])
    if np.isnan(rsi):
        rsi_score, rsi = 15, None
    elif rsi < 30:
        rsi_score = 28   # oversold
    elif rsi < 50:
        rsi_score = 22
    elif rsi < 65:
        rsi_score = 14
    elif rsi < 75:
        rsi_score = 8    # overbought caution
    else:
        rsi_score = 3    # strongly overbought

    hist = compute_macd(close)
    h_now  = float(hist.iloc[-1])
    h_prev = float(hist.iloc[-2]) if len(hist) > 1 else 0.0
    if np.isnan(h_now):
        macd_score = 10
    elif h_now > 0 and h_now > h_prev:
        macd_score = 18   # bullish, accelerating
    elif h_now > 0:
        macd_score = 13
    elif h_now < 0 and h_now < h_prev:
        macd_score = 2    # bearish, accelerating
    else:
        macd_score = 7

    avg_vol = volume.rolling(20).mean().iloc[-1]
    vol_ratio = float(volume.iloc[-1] / avg_vol) if avg_vol > 0 else 1.0
    if np.isnan(vol_ratio):
        vol_score = 5
    elif vol_ratio > 1.5 and ma_bullish:
        vol_score = 10
    elif vol_ratio > 1.5 and ma_bearish:
        vol_score = 2
    else:
        vol_score = 5

    score = ma_score + rsi_score + macd_score + vol_score

    return {
        "signal": score_to_signal(score),
        "score": score,
        "rsi": round(rsi, 1) if rsi is not None else None,
        "ma_alignment": "bullish" if ma_bullish else ("bearish" if ma_bearish else "mixed"),
    }


# ── Base Rate (controls for bull/bear market bias) ────────────────────────────

def compute_base_rates(data: pd.DataFrame, test_start: pd.Timestamp) -> dict:
    """
    What % of ALL days had positive returns over each forward window?
    This is the 'random buy' baseline. A signal only has real edge when
    its accuracy meaningfully exceeds this rate.
    """
    test_data = data[data.index >= test_start]["Close"]
    rates = {}
    for fwd in FORWARD_WINDOWS:
        positives = 0
        total = 0
        for i in range(len(test_data) - fwd):
            ret = test_data.iloc[i + fwd] / test_data.iloc[i] - 1
            positives += int(ret > 0)
            total += 1
        rates[fwd] = (positives / total * 100) if total > 0 else 50.0
    return rates


# ── Backtest Engine ───────────────────────────────────────────────────────────

def backtest_stock(ticker: str, data: pd.DataFrame) -> tuple:
    """Returns (signals_df, base_rates_dict)."""
    test_start = data.index[-1] - pd.Timedelta(days=LOOKBACK_DAYS)
    base_rates = compute_base_rates(data, test_start)
    test_dates = data.index[data.index >= test_start][::SAMPLE_INTERVAL]

    rows = []
    for signal_date in test_dates:
        loc = data.index.get_loc(signal_date)
        df_slice = data.iloc[: loc + 1]  # no lookahead: only history up to this date
        sig = generate_signal(df_slice)
        signal_price = float(data["Close"].iloc[loc])

        row = {
            "ticker": ticker,
            "date": signal_date.date(),
            "price": round(signal_price, 2),
            "signal": sig["signal"],
            "score": sig["score"],
            "rsi": sig["rsi"],
            "ma_alignment": sig["ma_alignment"],
        }

        for fwd in FORWARD_WINDOWS:
            future_loc = loc + fwd
            if future_loc < len(data):
                future_price = float(data["Close"].iloc[future_loc])
                ret = (future_price / signal_price - 1) * 100
                correct = (
                    (sig["signal"] == "buy"  and ret > 0)
                    or (sig["signal"] == "sell" and ret < 0)
                )
                row[f"ret_{fwd}d"] = round(ret, 2)
                row[f"ok_{fwd}d"] = correct if sig["signal"] != "hold" else None
            else:
                row[f"ret_{fwd}d"] = None
                row[f"ok_{fwd}d"]  = None

        rows.append(row)

    return pd.DataFrame(rows), base_rates


# ── Terminal Output ───────────────────────────────────────────────────────────

def _try_rich():
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        return Console(), Table, box
    except ImportError:
        return None, None, None


def print_stock_summary(ticker: str, df: pd.DataFrame, base_rates: dict) -> None:
    console, Table, box = _try_rich()
    signals = ["buy", "sell"]

    if console:
        console.print(f"\n[bold cyan]── {ticker} ──[/bold cyan]")
        # Base rate row header
        br_str = "  ".join(
            f"{fwd}d base={base_rates.get(fwd, 50):.0f}%" for fwd in FORWARD_WINDOWS
        )
        console.print(f"  [dim]Random-buy baseline: {br_str}[/dim]")

        tbl = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
        tbl.add_column("Signal")
        tbl.add_column("N", justify="right")
        for fwd in FORWARD_WINDOWS:
            tbl.add_column(f"Acc {fwd}d", justify="right")
            tbl.add_column(f"Lift {fwd}d", justify="right")   # acc − base_rate
            tbl.add_column(f"AvgRet {fwd}d", justify="right")

        for sig in signals:
            sub = df[df["signal"] == sig]
            cells = [sig.upper(), str(len(sub))]
            for fwd in FORWARD_WINDOWS:
                valid = sub[sub[f"ok_{fwd}d"].notna()]
                base = base_rates.get(fwd, 50.0)
                if len(valid) == 0:
                    cells += ["—", "—", "—"]
                else:
                    acc = valid[f"ok_{fwd}d"].mean() * 100
                    avg = valid[f"ret_{fwd}d"].mean()
                    lift = acc - (base if sig == "buy" else (100 - base))
                    c = "green" if acc >= 55 else ("yellow" if acc >= 45 else "red")
                    lc = "green" if lift >= 5 else ("yellow" if lift >= 0 else "red")
                    r = "green" if avg > 0 else "red"
                    cells.append(f"[{c}]{acc:.0f}%[/{c}]")
                    cells.append(f"[{lc}]{lift:+.0f}pp[/{lc}]")
                    cells.append(f"[{r}]{avg:+.1f}%[/{r}]")
            tbl.add_row(*cells)

        hold_n = len(df[df["signal"] == "hold"])
        tbl.add_row("HOLD", str(hold_n), *["—"] * (len(FORWARD_WINDOWS) * 3))
        console.print(tbl)
    else:
        print(f"\n=== {ticker} ===")
        br_str = "  ".join(f"{fwd}d={base_rates.get(fwd,50):.0f}%" for fwd in FORWARD_WINDOWS)
        print(f"  Random baseline: {br_str}")
        for sig in signals:
            sub = df[df["signal"] == sig]
            print(f"  {sig.upper()} ({len(sub)}):")
            for fwd in FORWARD_WINDOWS:
                valid = sub[sub[f"ok_{fwd}d"].notna()]
                base = base_rates.get(fwd, 50.0)
                if not len(valid):
                    print(f"    {fwd}d: —")
                else:
                    acc = valid[f"ok_{fwd}d"].mean() * 100
                    avg = valid[f"ret_{fwd}d"].mean()
                    lift = acc - (base if sig == "buy" else (100 - base))
                    print(f"    {fwd}d: acc={acc:.0f}%  lift={lift:+.0f}pp  avg_ret={avg:+.1f}%")


def print_portfolio_summary(results: dict, all_base_rates: dict) -> None:
    console, Table, box = _try_rich()
    all_rows = pd.concat([v[0] for v in results.values()], ignore_index=True)
    # Average base rates across all stocks
    avg_base = {
        fwd: np.mean([all_base_rates[t][fwd] for t in all_base_rates if fwd in all_base_rates[t]])
        for fwd in FORWARD_WINDOWS
    }

    if console:
        console.rule("[bold]Portfolio-Wide Summary (all stocks combined)[/bold]")
        br_str = "  ".join(f"{fwd}d={avg_base[fwd]:.0f}%" for fwd in FORWARD_WINDOWS)
        console.print(f"  [dim]Average random-buy baseline: {br_str}[/dim]")

        tbl = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
        tbl.add_column("Signal")
        tbl.add_column("Total N", justify="right")
        for fwd in FORWARD_WINDOWS:
            tbl.add_column(f"Acc {fwd}d", justify="right")
            tbl.add_column(f"Lift {fwd}d", justify="right")
            tbl.add_column(f"AvgRet {fwd}d", justify="right")

        for sig in ["buy", "sell"]:
            sub = all_rows[all_rows["signal"] == sig]
            cells = [sig.upper(), str(len(sub))]
            for fwd in FORWARD_WINDOWS:
                valid = sub[sub[f"ok_{fwd}d"].notna()]
                base = avg_base.get(fwd, 50.0)
                if not len(valid):
                    cells += ["—", "—", "—"]
                else:
                    acc = valid[f"ok_{fwd}d"].mean() * 100
                    avg = valid[f"ret_{fwd}d"].mean()
                    lift = acc - (base if sig == "buy" else (100 - base))
                    c = "green" if acc >= 55 else ("yellow" if acc >= 45 else "red")
                    lc = "green" if lift >= 5 else ("yellow" if lift >= 0 else "red")
                    r = "green" if avg > 0 else "red"
                    cells.append(f"[{c}]{acc:.0f}%[/{c}]")
                    cells.append(f"[{lc}]{lift:+.0f}pp[/{lc}]")
                    cells.append(f"[{r}]{avg:+.1f}%[/{r}]")
            tbl.add_row(*cells)

        console.print(tbl)
    else:
        print("\n=== PORTFOLIO SUMMARY ===")
        br_str = "  ".join(f"{fwd}d={avg_base[fwd]:.0f}%" for fwd in FORWARD_WINDOWS)
        print(f"  Average random baseline: {br_str}")
        for sig in ["buy", "sell"]:
            sub = all_rows[all_rows["signal"] == sig]
            print(f"  {sig.upper()} ({len(sub)} total):")
            for fwd in FORWARD_WINDOWS:
                valid = sub[sub[f"ok_{fwd}d"].notna()]
                base = avg_base.get(fwd, 50.0)
                if not len(valid):
                    print(f"    {fwd}d: —")
                else:
                    acc = valid[f"ok_{fwd}d"].mean() * 100
                    avg = valid[f"ret_{fwd}d"].mean()
                    lift = acc - (base if sig == "buy" else (100 - base))
                    print(f"    {fwd}d: acc={acc:.0f}%  lift={lift:+.0f}pp  avg_ret={avg:+.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    console, _, _ = _try_rich()
    header = f"DSA Technical Backtester  |  {LOOKBACK_DAYS}d lookback  |  sample every {SAMPLE_INTERVAL} bars"
    if console:
        console.print(f"\n[bold green]{header}[/bold green]")
        console.print(f"Stocks: {', '.join(STOCKS)}\n")
    else:
        print(f"\n{header}")
        print(f"Stocks: {', '.join(STOCKS)}\n")

    fetch_start = (date.today() - timedelta(days=LOOKBACK_DAYS + 250)).strftime("%Y-%m-%d")
    results = {}       # ticker → (df, base_rates)
    all_base_rates = {}

    for ticker in STOCKS:
        print(f"Downloading {ticker} ...", end=" ", flush=True)
        try:
            raw = yf.download(ticker, start=fetch_start, auto_adjust=True, progress=False)
            if raw.empty:
                print("no data, skipping")
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            print(f"{len(raw)} bars")
            df, base_rates = backtest_stock(ticker, raw)
            results[ticker] = (df, base_rates)
            all_base_rates[ticker] = base_rates
            print_stock_summary(ticker, df, base_rates)
        except Exception as exc:
            print(f"ERROR: {exc}")

    if not results:
        print("No results to show.")
        return

    print_portfolio_summary(results, all_base_rates)

    out = "backtest_results.csv"
    pd.concat([v[0] for v in results.values()], ignore_index=True).to_csv(out, index=False)
    print(f"\nDetailed results → {out}")
    print("\nHow to read:")
    print("  Accuracy = % of BUY signals where price rose / SELL signals where price fell")
    print("  AvgRet   = average actual return over that window (positive = signal made money)")
    print("  >55% accuracy + positive AvgRet = signal has edge")
    print("  ~50% accuracy = random (no edge)")


if __name__ == "__main__":
    main()
