import os
import time
import hmac
import hashlib
import logging
from typing import Dict, Any, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing environment variable: {name}")
    return v


class PionexClient:
    """
    Minimal Pionex REST client for PRIVATE GET endpoints (read-only).
    Implements signing per Pionex docs.
    """
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://api.pionex.com"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "PIONEX-KEY": self.api_key,
            "Content-Type": "application/json",
        })

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _build_sorted_query(params: Dict[str, Any]) -> str:
        # IMPORTANT: Signature-related values must not be URL-encoded (per docs).
        # We keep it simple for now: timestamp is numeric.
        items = [(k, str(v)) for k, v in params.items() if v is not None]
        items.sort(key=lambda x: x[0])  # ASCII by key
        return "&".join([f"{k}={v}" for k, v in items])

    def _sign(self, method: str, path: str, query: str, body: Optional[str] = None) -> str:
        method = method.upper()
        path_url = f"{path}?{query}" if query else path
        payload = f"{method}{path_url}"
        if body:
            payload += body  # POST/DELETE only; not used now
        sig = hmac.new(self.api_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return sig

    def private_get(self, path: str, extra_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = {"timestamp": self._now_ms()}
        if extra_params:
            params.update(extra_params)

        query = self._build_sorted_query(params)
        signature = self._sign("GET", path, query)

        headers = {"PIONEX-SIGNATURE": signature}
        url = f"{self.base_url}{path}?{query}"

        resp = self.session.get(url, headers=headers, timeout=15)
        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(f"Non-JSON response from Pionex (status={resp.status_code}): {resp.text[:200]}")

        if not resp.ok:
            raise RuntimeError(f"HTTP {resp.status_code} from Pionex: {data}")

        return data

    def get_balances(self) -> Dict[str, Any]:
        return self.private_get("/api/v1/account/balances")


def main():
    logging.info("Booting Pionex bot (READ-ONLY)...")

    api_key = require_env("PIONEX_API_KEY")
    api_secret = require_env("PIONEX_API_SECRET")

    trading_mode = os.getenv("TRADING_MODE", "paper")
    symbols = os.getenv("SYMBOLS", "SOL/USDT,ARB/USDT")
    max_usdt_per_trade = float(os.getenv("MAX_USDT_PER_TRADE", "20"))
    max_daily_loss = float(os.getenv("MAX_DAILY_LOSS_USDT", "10"))

    logging.info(f"Mode: {trading_mode} (should be 'paper' for now)")
    logging.info(f"Symbols: {symbols}")
    logging.info(f"Risk: max_usdt_per_trade={max_usdt_per_trade}, max_daily_loss={max_daily_loss}")

    client = PionexClient(api_key=api_key, api_secret=api_secret)

    # Read-only polling loop
    while True:
        try:
            r = client.get_balances()
            if not r.get("result", False):
                logging.warning(f"Pionex returned result=false: {r}")
            else:
                balances = (r.get("data") or {}).get("balances", [])
                # Log a small, useful subset:
                interesting = {"USDT", "SOL", "ARB"}
                lines = []
                for b in balances:
                    coin = b.get("coin")
                    if coin in interesting:
                        lines.append(f"{coin}: free={b.get('free')} frozen={b.get('frozen')}")
                if lines:
                    logging.info("Balances: " + " | ".join(lines))
                else:
                    logging.info(f"Balances fetched OK. Coins={len(balances)} (not logging all).")

        except Exception as e:
            logging.error(f"Balance check failed: {e}")

        logging.info("Heartbeat: bot is alive.")
        time.sleep(30)


if __name__ == "__main__":
    main()
