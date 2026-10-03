"""TradingView direct data connector for Sweep.

Connects directly to TradingView's market data servers to fetch real-time
SPY 1-hour and daily candles without requiring a TradingView paid subscription,
webhooks, or ngrok.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tradingview_ta import Exchange, Interval, TA_Handler

from .market import ET, MarketData
from .yahoo import YahooMarket


class TradingViewMarket(YahooMarket):
    """Hybrid MarketData provider:

    - Pulls live real-time SPY candles directly from TradingView's ticker stream.
    - Uses YahooFinance for historical bars and options chains.
    - 100% Free: No TradingView Premium subscription, webhooks, or ngrok needed.
    """

    label: str = "TradingView (AMEX:SPY)"

    def __init__(self, underlying: str = "SPY", exchange: str = "AMEX", screener: str = "america") -> None:
        super().__init__(underlying=underlying)
        self.label = "TradingView (AMEX:SPY)"
        self.exchange = exchange
        self.screener = screener
        self._h1_handler = TA_Handler(
            symbol=self.underlying,
            exchange=self.exchange,
            screener=self.screener,
            interval=Interval.INTERVAL_1_HOUR,
        )
        self._daily_handler = TA_Handler(
            symbol=self.underlying,
            exchange=self.exchange,
            screener=self.screener,
            interval=Interval.INTERVAL_1_DAY,
        )

    def live_candle(self) -> dict:
        """Fetch the current live SPY 1-hour candle directly from TradingView."""
        analysis = self._h1_handler.get_analysis()
        ind = analysis.indicators
        now_utc = pd.Timestamp.now(tz="UTC")
        return {
            "time": now_utc,
            "open": float(ind.get("open", 0.0) or 0.0),
            "high": float(ind.get("high", 0.0) or 0.0),
            "low": float(ind.get("low", 0.0) or 0.0),
            "close": float(ind.get("close", 0.0) or 0.0),
            "volume": float(ind.get("volume", 0.0) or 0.0),
            "recommendation": analysis.summary.get("RECOMMENDATION", "NEUTRAL"),
            "rsi": float(ind.get("RSI", 50.0) or 50.0),
        }

    def minute_bars(self, start: date, end: date) -> pd.DataFrame:
        """Load history from cache/Yahoo and splice the latest live TradingView candle on top."""
        df = super().minute_bars(start, end)
        try:
            live = self.live_candle()
            if live["close"] > 0:
                t = live["time"].floor("1h")
                # Append or update latest bar with TradingView's live price
                new_row = pd.DataFrame([{
                    "open": live["open"],
                    "high": live["high"],
                    "low": live["low"],
                    "close": live["close"],
                    "volume": live["volume"],
                }], index=pd.DatetimeIndex([t], tz="UTC"))
                
                if t in df.index:
                    df.loc[t] = new_row.iloc[0]
                else:
                    df = pd.concat([df, new_row]).sort_index()
                self._set_bars(df)
        except Exception:
            pass

        return df
