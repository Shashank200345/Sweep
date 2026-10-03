from __future__ import annotations

import json
import urllib.request
from http.server import HTTPServer
from threading import Thread

import pytest

from sweep.webhook import TradingViewWebhookHandler


def test_tradingview_webhook_handler_get():
    # Start server on ephemeral port for testing
    server = HTTPServer(("127.0.0.1", 0), TradingViewWebhookHandler)
    port = server.server_address[1]
    t = Thread(target=server.handle_request, daemon=True)
    t.start()

    req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["status"] == "running"
        assert data["symbol"] == "SPY"

    server.server_close()


def test_tradingview_webhook_handler_post():
    server = HTTPServer(("127.0.0.1", 0), TradingViewWebhookHandler)
    port = server.server_address[1]
    t = Thread(target=server.handle_request, daemon=True)
    t.start()

    payload = json.dumps({"ticker": "SPY", "close": 570.5, "time": "2026-10-02T14:30:00Z"}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/webhook", data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["status"] == "received"
        assert data["ticker"] == "SPY"

    server.server_close()
