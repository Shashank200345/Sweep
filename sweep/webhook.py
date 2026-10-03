"""TradingView Webhook Listener for Sweep.

Receives live SPY 1h candle alerts from TradingView via HTTP POST requests,
updates Sweep's market state, triggers SMC structure analysis, asks TypeSafe Jev
System One if structure changed, and executes defined-risk credit spreads paper orders.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable
import pandas as pd

from .config import DEFAULT, Config
from .data.market import ET, MarketData
from .paper import PaperLoop
from .typesafe import SystemOne


class TradingViewWebhookHandler(BaseHTTPRequestHandler):
    """HTTP request handler for TradingView alert webhooks."""

    loop: PaperLoop | None = None
    verbose: bool = True

    def log_message(self, format: str, *args) -> None:
        """Suppress default HTTP request logging to keep output clean."""
        pass

    def do_GET(self) -> None:
        """Health check endpoint."""
        if self.path in ("/health", "/ping", "/"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status = {
                "status": "running",
                "service": "Sweep TradingView Listener",
                "symbol": "SPY",
                "time_utc": datetime.now(timezone.utc).isoformat(),
            }
            self.wfile.write(json.dumps(status).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        """Receive TradingView alert webhook payload."""
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            payload = json.loads(post_data.decode("utf-8"))
        except Exception:
            # Handle plain text or malformed JSON
            payload = {"raw": post_data.decode("utf-8", errors="ignore")}

        # Extract timestamp
        t_raw = payload.get("time") or payload.get("timestamp")
        t_parsed = None
        if t_raw:
            try:
                # Handle numeric epoch ms or s
                val = float(t_raw)
                unit = "ms" if val > 1e11 else "s"
                t_parsed = pd.to_datetime(val, unit=unit, utc=True)
            except Exception:
                try:
                    t_parsed = pd.to_datetime(t_raw, utc=True)
                except Exception:
                    pass

        if t_parsed is None:
            t_parsed = pd.Timestamp.now(tz="UTC")

        # Normalize to the closest 1h bar close if slight delay
        ticker = payload.get("ticker", "SPY")
        close_px = payload.get("close")
        
        print(f"\n[TradingView Webhook Received] {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
        print(f"  Ticker: {ticker} | Bar Time: {t_parsed} | Close: {close_px if close_px is not None else 'market'}")

        response_data = {
            "status": "received",
            "ticker": ticker,
            "timestamp": str(t_parsed),
        }

        # Process through Sweep PaperLoop
        if self.loop is not None:
            try:
                # Run the tick for this timestamp
                log = self.loop.tick(t_parsed)
                if log:
                    action = log.get("action", "hold")
                    structure = log.get("structure")
                    reason = log.get("reason", "")
                    execution = log.get("execution")
                    called = "Jev Model" if log.get("jev_called") else "Materiality Reused"
                    
                    print(f"  [Sweep Decision] {action.upper()} ({structure or 'none'}) via {called}")
                    print(f"  Reason: {reason}")
                    if execution:
                        print(f"  Execution: {execution}")
                    
                    response_data.update({
                        "decision": action,
                        "structure": structure,
                        "reason": reason,
                        "execution": execution,
                        "equity": log.get("equity"),
                    })
                else:
                    print("  [Sweep Status] Waiting for complete hourly candle or outside market hours.")
            except Exception as ex:
                print(f"  [Sweep Error] Processing error: {ex}")
                response_data["error"] = str(ex)

        # Return 200 OK to TradingView
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode("utf-8"))


def start_tradingview_listener(
    md: MarketData,
    jev: SystemOne,
    state_dir: Path | str = "paper_state",
    cfg: Config = DEFAULT,
    host: str = "0.0.0.0",
    port: int = 8767,
    verbose: bool = True,
) -> None:
    """Start the TradingView Webhook HTTP server listener."""
    state_dir = Path(state_dir)
    loop = PaperLoop(md, jev, state_dir, cfg, verbose=verbose)

    TradingViewWebhookHandler.loop = loop
    TradingViewWebhookHandler.verbose = verbose

    server = HTTPServer((host, port), TradingViewWebhookHandler)
    print("=" * 70)
    print(f"  Sweep TradingView Webhook Listener Active")
    print(f"  Listening on: http://{host}:{port}/webhook")
    print(f"  Data Source: {md.label} (SPY)")
    print(f"  Model: {getattr(jev, 'model', '?')}")
    print(f"  State Directory: {state_dir.resolve()}")
    print("=" * 70)
    print("\nInstructions for TradingView:")
    print(f"1. If running locally with ngrok: `ngrok http {port}`")
    print("   Webhook URL: https://<your-ngrok-subdomain>.ngrok-free.app/webhook")
    print("2. Set Alert in TradingView: Condition: SPY 1h | Frequency: Once Per Bar Close")
    print('   Message payload: {"ticker": "{{ticker}}", "time": "{{time}}", "close": {{close}}, "open": {{open}}, "high": {{high}}, "low": {{low}}}')
    print("\nWaiting for incoming TradingView alerts... (Press Ctrl+C to stop)\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping TradingView Webhook Listener...")
        server.server_close()
