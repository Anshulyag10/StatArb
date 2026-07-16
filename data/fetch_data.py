"""
fetch_data.py -- Download and cache historical prices via yfinance.

Supports single-pair and multi-ticker downloads with CSV caching.
"""

import os
import pandas as pd
import yfinance as yf


def fetch_prices(ticker_y: str, ticker_x: str,
                 start: str, end: str,
                 cache_dir: str | None = None) -> pd.DataFrame:
    """
    Download adjusted close prices for two tickers and return a DataFrame
    with columns [ticker_y, ticker_x] indexed by date.
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(
        cache_dir, f"{ticker_y}_{ticker_x}_{start}_{end}.csv"
    )

    if os.path.exists(cache_path):
        print(f"[data] Loading cached prices from {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    print(f"[data] Downloading {ticker_y} and {ticker_x} "
          f"from {start} to {end} ...")

    raw = yf.download(
        [ticker_y, ticker_x],
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        df = raw["Close"][[ticker_y, ticker_x]].copy()
    else:
        df = raw[["Close"]].copy()
        df.columns = [ticker_y]

    df.dropna(inplace=True)
    df.to_csv(cache_path)
    print(f"[data] Cached {len(df)} rows -> {cache_path}")
    return df


def fetch_multi_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """
    Download adjusted close prices for multiple tickers.

    Returns DataFrame with columns = ticker symbols, indexed by date.
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)

    ticker_key = "_".join(sorted(tickers))
    cache_path = os.path.join(cache_dir, f"multi_{ticker_key}_{start}_{end}.csv")

    if os.path.exists(cache_path):
        print(f"[data] Loading cached multi-ticker data from {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    print(f"[data] Downloading {len(tickers)} tickers: {', '.join(tickers)} ...")

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        df = raw["Close"][tickers].copy()
    else:
        df = raw[["Close"]].copy()
        df.columns = tickers[:1]

    df.dropna(inplace=True)
    df.to_csv(cache_path)
    print(f"[data] Cached {len(df)} rows x {len(df.columns)} tickers -> {cache_path}")
    return df


def fetch_vix(start: str, end: str, cache_dir: str | None = None) -> pd.Series:
    """
    Download VIX index data for regime filtering.
    Returns a Series of VIX closing values.
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"VIX_{start}_{end}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df.squeeze()

    print("[data] Downloading VIX data ...")
    raw = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        vix = raw["Close"].squeeze()
    else:
        vix = raw["Close"].squeeze()

    vix = vix.dropna()
    vix.to_csv(cache_path)
    print(f"[data] Cached VIX data -> {cache_path}")
    return vix


def fetch_volumes(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Download daily volume data for realistic cost modeling."""
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)

    ticker_key = "_".join(sorted(tickers))
    cache_path = os.path.join(cache_dir, f"vol_{ticker_key}_{start}_{end}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    print(f"[data] Downloading volume data for {', '.join(tickers)} ...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        df = raw["Volume"][tickers].copy()
    else:
        df = raw[["Volume"]].copy()
        df.columns = tickers[:1]

    df.dropna(inplace=True)
    df.to_csv(cache_path)
    return df


# Quick smoke test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config import TICKER_Y, TICKER_X, START_DATE, END_DATE

    prices = fetch_prices(TICKER_Y, TICKER_X, START_DATE, END_DATE)
    print(prices.head())
    print(f"\nShape: {prices.shape}")
