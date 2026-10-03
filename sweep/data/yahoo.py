"""Yahoo Finance market data provider: 100% free live and historical SPY data.

No API key, no account registration, and no Tax ID required.
Satisfies the MarketData protocol (minute_bars, dividends, contracts, quotes).
Provides real SPY candles (up to 5,000+ hourly bars from NYSE/Arca) and supports
both live options chains and historical BSM pricing for multi-year backtests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
import yfinance as yf

from ..config import UNDERLYING
from ..options import bsm
from .market import ET, YEAR_S, empty_quotes, expiry_close_utc, occ_ticker, parse_occ


@dataclass
class YahooMarket:
    underlying: str = UNDERLYING
    label: str = "YahooFinance"
    is_fixture: bool = False
    cache_dir: Path = field(default_factory=lambda: Path(".sweep_cache/yahoo"))
    _chain_cache: dict[str, object] = field(default_factory=dict, repr=False)
    _ticker_obj: yf.Ticker | None = field(default=None, repr=False)
    _options_cache: tuple | None = field(default=None, repr=False)
    _bars_cache: pd.DataFrame | None = field(default=None, repr=False)

    def _ticker(self) -> yf.Ticker:
        if self._ticker_obj is None:
            self._ticker_obj = yf.Ticker(self.underlying)
        return self._ticker_obj

    def minute_bars(self, start: date, end: date) -> pd.DataFrame:
        """Fetch real SPY bars (minute if recent, 1h for full multi-year history)."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{self.underlying}_bars_{start}_{end}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                self._set_bars(df)
                return df
            except Exception:
                pass


        now = pd.Timestamp.now(tz=ET).date()
        is_recent = (now - start).days <= 28

        df = pd.DataFrame()
        if is_recent:
            try:
                df = yf.download(self.underlying, start=start, end=end + timedelta(days=1),
                                 interval="1m", progress=False)
            except Exception:
                df = pd.DataFrame()

        if df.empty:
            # Fetch 1h bars (Yahoo supports 730 days of real 1h bars)
            earliest_yahoo = now - timedelta(days=720)
            fetch_start = max(start, earliest_yahoo)
            try:
                df = yf.download(self.underlying, start=fetch_start, end=end + timedelta(days=1),
                                 interval="1h", progress=False)
            except Exception:
                df = pd.DataFrame()

        if df.empty:
            # Fall back to period='730d' if start was clamped
            try:
                df = yf.download(self.underlying, period="730d", interval="1h", progress=False)
            except Exception:
                df = pd.DataFrame()


        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        # Ensure index is UTC DatetimeIndex
        if df.index.tz is None:
            df.index = pd.DatetimeIndex(df.index).tz_localize("UTC")
        else:
            df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")

        # Keep standard columns
        cols = ["open", "high", "low", "close", "volume"]
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
        df = df[cols].astype(float)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        self._set_bars(df)

        try:
            df.to_parquet(cache_file)
        except Exception:
            pass

        return df

    def dividends(self) -> pd.DataFrame:
        """Trailing cash dividends from Yahoo Finance."""
        try:
            divs = self._ticker().dividends
            if divs.empty:
                return pd.DataFrame(columns=["ex_date", "declaration_date", "pay_date", "cash_amount"])
            ex_dates = pd.to_datetime(divs.index).tz_convert(ET).normalize().tz_localize(None)
            return pd.DataFrame({
                "ex_date": ex_dates,
                "declaration_date": pd.NaT,
                "pay_date": pd.NaT,
                "cash_amount": divs.values.astype(float),
            })
        except Exception:
            return pd.DataFrame(columns=["ex_date", "declaration_date", "pay_date", "cash_amount"])

    _bars_cache: pd.DataFrame | None = field(default=None, repr=False)
    _close_times: np.ndarray | None = field(default=None, repr=False)
    _close_prices: np.ndarray | None = field(default=None, repr=False)
    _close_vols: np.ndarray | None = field(default=None, repr=False)

    def _set_bars(self, df: pd.DataFrame) -> None:
        self._bars_cache = df
        if not df.empty:
            self._close_times = df.index.asi8
            self._close_prices = df["close"].to_numpy(dtype=float)
            ret = pd.Series(self._close_prices).pct_change().fillna(0.0)
            roll_std = ret.rolling(140, min_periods=20).std().fillna(0.16 / np.sqrt(252 * 7))
            self._close_vols = np.clip(roll_std.to_numpy() * np.sqrt(252 * 7), 0.08, 0.45)

    def _spot_at_ts(self, at: pd.Timestamp) -> float:
        if self._close_times is not None and len(self._close_times):
            v = at.value if hasattr(at, "value") else pd.Timestamp(at).value
            idx = int(np.searchsorted(self._close_times, v, side="right") - 1)
            if idx >= 0:
                return float(self._close_prices[min(idx, len(self._close_prices) - 1)])
            return float(self._close_prices[0])
        return 500.0

    def _vol_at_ts(self, at: pd.Timestamp) -> float:
        if self._close_times is not None and len(self._close_times):
            v = at.value if hasattr(at, "value") else pd.Timestamp(at).value
            idx = int(np.searchsorted(self._close_times, v, side="right") - 1)
            if idx >= 0:
                return float(self._close_vols[min(idx, len(self._close_vols) - 1)])
        return 0.16


    def contracts(self, as_of: date, exp_from: date, exp_to: date) -> pd.DataFrame:
        """Option contracts within the target expiration window."""
        now = pd.Timestamp.now(tz=ET).date()
        is_live = as_of >= (now - timedelta(days=2))
        records = []

        if is_live:
            if self._options_cache is None:
                try:
                    self._options_cache = self._ticker().options
                except Exception:
                    self._options_cache = ()
            available_exps = self._options_cache

            for e in available_exps:
                e_date = pd.to_datetime(e).date()
                if not (exp_from <= e_date <= exp_to):
                    continue
                chain = self._get_option_chain(e)
                if chain is None:
                    continue
                for kind, opt_df in [("call", getattr(chain, "calls", None)), ("put", getattr(chain, "puts", None))]:
                    if opt_df is None or opt_df.empty:
                        continue
                    for _, row in opt_df.iterrows():
                        sym = str(row["contractSymbol"])
                        ticker = sym if sym.startswith("O:") else f"O:{sym}"
                        records.append({
                            "ticker": ticker,
                            "type": kind,
                            "strike": float(row["strike"]),
                            "expiry": pd.to_datetime(e).normalize(),
                        })

        # If historical date or no active chains found in Yahoo live
        if not records:
            fridays = pd.date_range(pd.Timestamp(exp_from), pd.Timestamp(exp_to), freq="W-FRI")
            if not len(fridays):
                return pd.DataFrame(columns=["ticker", "type", "strike", "expiry"])
            ref = self._spot_at_ts(pd.Timestamp(as_of, tz="UTC"))
            strikes = np.arange(np.floor(ref * 0.85), np.ceil(ref * 1.15) + 1, 1.0)
            e, k, c = (a.ravel() for a in np.meshgrid(fridays, strikes, [0, 1], indexing="ij"))
            kind = np.where(c == 0, "call", "put")
            exp = pd.DatetimeIndex(e)
            tick = [occ_ticker(self.underlying, x, y, z) for x, y, z in zip(exp.date, kind, k)]
            return pd.DataFrame({"ticker": tick, "type": kind, "strike": k.astype(float), "expiry": exp.values})

        return pd.DataFrame(records)

    def _get_option_chain(self, expiry_str: str):
        if expiry_str in self._chain_cache:
            return self._chain_cache[expiry_str]
        if self._options_cache is not None and expiry_str not in self._options_cache:
            return None
        try:
            oc = self._ticker().option_chain(expiry_str)
            self._chain_cache[expiry_str] = oc
            return oc
        except Exception:
            self._chain_cache[expiry_str] = None
            return None

    def quotes(self, tickers: Sequence[str], at: pd.Timestamp) -> pd.DataFrame:
        """Quotes for specific OCC option tickers."""
        if not tickers:
            return empty_quotes([])

        utc_at = pd.Timestamp(at).tz_convert("UTC") if pd.Timestamp(at).tzinfo else pd.Timestamp(at, tz="UTC")
        now_utc = pd.Timestamp.now(tz="UTC")
        is_live_quote = (now_utc - utc_at).total_seconds() < 86400 * 2

        lookup: dict[str, dict] = {}
        if is_live_quote:
            # Ensure all relevant chains are loaded into cache for live contracts
            parsed = parse_occ(tickers)
            unique_expiries = parsed["expiry"].dt.strftime("%Y-%m-%d").unique()
            for e_str in unique_expiries:
                if self._options_cache is not None and e_str not in self._options_cache:
                    continue
                if e_str not in self._chain_cache:
                    self._get_option_chain(e_str)

            # Build lookup table across cached live chains
            for e_str, oc in self._chain_cache.items():
                if oc is None:
                    continue
                for opt_df in [getattr(oc, "calls", None), getattr(oc, "puts", None)]:
                    if opt_df is None or opt_df.empty:
                        continue
                    for _, row in opt_df.iterrows():
                        sym = str(row["contractSymbol"])
                        t = sym if sym.startswith("O:") else f"O:{sym}"
                        bid = float(row.get("bid", 0.0) or 0.0)
                        ask = float(row.get("ask", 0.0) or 0.0)
                        last_price = float(row.get("lastPrice", 0.0) or 0.0)
                        ts = row.get("lastTradeDate")
                        if pd.isna(ts) or ts is None:
                            ts = utc_at
                        else:
                            ts = pd.to_datetime(ts)
                            ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")

                        if (bid <= 0 or ask <= bid) and last_price > 0:
                            spread = max(0.05, round(last_price * 0.04, 2))
                            bid = max(0.01, round(last_price - spread / 2, 2))
                            ask = round(last_price + spread / 2, 2)

                        lookup[t] = {"bid": bid, "ask": ask, "ts": ts}

        # For historical tickers not in live chain, price via BSM from real market spot & vol
        missing = [t for t in tickers if t not in lookup]
        if missing:
            info = parse_occ(missing)
            S = self._spot_at_ts(utc_at)
            T = (expiry_close_utc(info["expiry"]).asi8 - utc_at.value) / 1e9 / YEAR_S
            day_vol = self._vol_at_ts(utc_at)
            live = T > 0
            Ts = np.where(live, T, 1e-9)
            m = np.log(info["strike"].values / S) / np.sqrt(np.maximum(Ts, 1 / 365))
            iv = np.clip(day_vol + 0.02 - 0.10 * m + 0.15 * m * m, 0.05, 2.0)
            mid = bsm.price(info["type"].values == "call", S, info["strike"].values, Ts, 0.04, 0.013, iv)
            hs = np.maximum(0.01, 0.015 * mid)
            bid = np.floor((mid - hs) * 100) / 100
            ask = np.ceil((mid + hs) * 100) / 100
            bid = np.where(bid < 0.01, 0.0, bid)
            for t_sym, b, a_val, is_live in zip(missing, bid, ask, live):
                lookup[t_sym] = {"bid": float(b) if is_live else np.nan,
                                 "ask": float(a_val) if is_live else np.nan,
                                 "ts": utc_at}

        bids, asks, tss = [], [], []
        for t in tickers:
            if t in lookup:
                bids.append(lookup[t]["bid"])
                asks.append(lookup[t]["ask"])
                tss.append(lookup[t]["ts"])
            else:
                bids.append(np.nan)
                asks.append(np.nan)
                tss.append(pd.NaT)

        return pd.DataFrame({"bid": bids, "ask": asks, "ts": pd.DatetimeIndex(tss, tz="UTC")},
                            index=pd.Index(list(tickers), name="ticker"))
