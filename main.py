import os
import time
import hmac
import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing environment variable: {name}")
    return v


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def parse_symbols(raw: str) -> List[str]:
    # "SOL/USDT,ARB/USDT" -> ["SOL/USDT", "ARB/USDT"]
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def normalize_symbol_variants(symbol: str) -> List[str]:
    """
    Pionex kan bruke ulike symbol-formater på public endpoints.
    Vi prøver noen varianter for å gjøre det robust.
    """
    s = symbol.strip()
    variants = [s]
    if "/" in s:
        variants.append(s.replace("/", "_"))  # SOL_USDT
        variants.append(s.replace("/", ""))   # SOLUSDT (hvis noen bruker det)
    return list(dict.fromkeys(variants))  # de-dupe


class PionexClient:
    """
    Minimal Pionex REST client for PRIVATE GET endpoints (read-only) + PUBLIC GET for priser.
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
        items = [(k, str(v)) for k, v in params.items() if v is not None]
        items.sort(key=lambda x: x[0])
        return "&".join([f"{k}={v}" for k, v in items])

    def _sign(self, method: str, path: str, query: str, body: Optional[str] = None) -> str:
        method = method.upper()
        path_url = f"{path}?{query}" if query else path
        payload = f"{method}{path_url}"
        if body:
            payload += body
        return hmac.new(
            self.api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

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
            raise RuntimeError(f"Non-JSON response (status={resp.status_code}): {resp.text[:200]}")

        if not resp.ok:
            raise RuntimeError(f"HTTP {resp.status_code}: {data}")
        return data

    def public_get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Public endpoints trenger normalt ikke signatur
        query = ""
        if params:
            query = self._build_sorted_query(params)
        url = f"{self.base_url}{path}" + (f"?{query}" if query else "")
        resp = self.session.get(url, timeout=15)
        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(f"Non-JSON public response (status={resp.status_code}): {resp.text[:200]}")
        if not resp.ok:
            raise RuntimeError(f"HTTP {resp.status_code} (public): {data}")
        return data

    def get_balances(self) -> Dict[str, Any]:
        return self.private_get("/api/v1/account/balances")

    def get_prices_best_effort(self, symbols: List[str]) -> Dict[str, float]:
        """
        Henter priser på en robust måte:
        - Prøver å hente alle tickers, og filtrerer (hvis endpointet finnes)
        - Hvis det feiler, prøver vi symbol-spesifikk ticker (hvis endpointet finnes)
        Returnerer dict: { "SOL/USDT": price, ... } (for de vi klarer)
        """
        # 1) Prøv "alle tickers"-variant
        prices: Dict[str, float] = {}
        try:
            # Noen API-er har /api/v1/market/tickers
            r = self.public_get("/api/v1/market/tickers")
            if r.get("result") is True:
                data = r.get("data") or {}
                tickers = data.get("tickers") or data.get("data") or data.get("list") or []
                # tickers forventes å være liste av dicts med symbol + last/close
                for sym in symbols:
                    for variant in normalize_symbol_variants(sym):
                        for t in tickers:
                            ts = t.get("symbol") or t.get("symbolName") or t.get("market")
                            if ts == variant:
                                last = t.get("last") or t.get("close") or t.get("price")
                                if last is not None:
                                    try:
                                        prices[sym] = float(last)
                                        break
                                    except Exception:
                                        pass
                        if sym in prices:
                            break
        except Exception as e:
            logging.info(f"Public tickers endpoint not usable (ok): {e}")

        # 2) Hvis vi mangler noen, prøv symbol-spesifikt
        missing = [s for s in symbols if s not in prices]
        for sym in missing:
            ok = False
            for variant in normalize_symbol_variants(sym):
                try:
                    # Noen API-er har /api/v1/market/ticker?symbol=...
                    r = self.public_get("/api/v1/market/ticker", params={"symbol": variant})
                    if r.get("result") is True:
                        data = r.get("data") or {}
                        last = data.get("last") or data.get("close") or data.get("price")
                        if last is not None:
                            prices[sym] = float(last)
                            ok = True
                            break
                except Exception:
                    continue
            if not ok:
                logging.warning(f"Could not fetch price for {sym} (tried variants).")

        return prices


class TradeGuard:
    """
    Enkle guardrails:
    - Kill switch (BOT_ENABLED)
    - Trading mode må være 'live' for å plassere ordre (vi plasserer ikke ordre i denne versjonen)
    - Minimum USDT før vi i det hele tatt vurderer trading
    - Rate limit (maks trades per time) – forberedt til senere
    """
    def __init__(self, min_usdt_to_trade: float, max_trades_per_hour: int):
        self.min_usdt_to_trade = min_usdt_to_trade
        self.max_trades_per_hour = max_trades_per_hour
        self.trade_timestamps: List[int] = []

    def can_trade(self, bot_enabled: bool, trading_mode: str, usdt_free: float) -> Tuple[bool, str]:
        if not bot_enabled:
            return False, "BOT_ENABLED=false (kill-switch aktiv)"
        if trading_mode.strip().lower() != "live":
            return False, f"TRADING_MODE={trading_mode} (ikke live)"
        if usdt_free < self.min_usdt_to_trade:
            return False, f"USDT free {usdt_free:.4f} < MIN_USDT_TO_TRADE {self.min_usdt_to_trade:.2f}"
        # rate limit (til senere)
        now = int(time.time())
        one_hour_ago = now - 3600
        self.trade_timestamps = [t for t in self.trade_timestamps if t >= one_hour_ago]
        if len(self.trade_timestamps) >= self.max_trades_per_hour:
            return False, f"Rate limit: {len(self.trade_timestamps)}/{self.max_trades_per_hour} trades siste time"
        return True, "OK"

    def record_trade(self):
        self.trade_timestamps.append(int(time.time()))


def main():
    logging.info("Booting Pionex bot (READ-ONLY + PRICES + GUARDS)...")

    api_key = require_env("PIONEX_API_KEY")
    api_secret = require_env("PIONEX_API_SECRET")

    trading_mode = os.getenv("TRADING_MODE", "paper")
    symbols = parse_symbols(os.getenv("SYMBOLS", "SOL/USDT,ARB/USDT"))
    max_usdt_per_trade = float(os.getenv("MAX_USDT_PER_TRADE", "20"))
    max_daily_loss = float(os.getenv("MAX_DAILY_LOSS_USDT", "10"))

    # Guardrails
    bot_enabled = env_bool("BOT_ENABLED", default=False)  # default OFF for safety
    min_usdt_to_trade = float(os.getenv("MIN_USDT_TO_TRADE", "10"))
    max_trades_per_hour = int(os.getenv("MAX_TRADES_PER_HOUR", "6"))

    logging.info(f"Mode: {trading_mode}")
    logging.info(f"Symbols: {','.join(symbols)}")
    logging.info(f"Risk: max_usdt_per_trade={max_usdt_per_trade}, max_daily_loss={max_daily_loss}")
    logging.info(f"Guards: BOT_ENABLED={bot_enabled}, MIN_USDT_TO_TRADE={min_usdt_to_trade}, MAX_TRADES_PER_HOUR={max_trades_per_hour}")

    client = PionexClient(api_key=api_key, api_secret=api_secret)
    guard = TradeGuard(min_usdt_to_trade=min_usdt_to_trade, max_trades_per_hour=max_trades_per_hour)

    while True:
        usdt_free = 0.0

        # 1) Balances (private)
        try:
            r = client.get_balances()
            if r.get("result") is not True:
                logging.warning(f"Balances result=false: {r}")
            else:
                balances = (r.get("data") or {}).get("balances", [])
                for b in balances:
                    if b.get("coin") == "USDT":
                        try:
                            usdt_free = float(b.get("free", 0))
                        except Exception:
                            usdt_free = 0.0
                        logging.info(f"Balances: USDT free={usdt_free} frozen={b.get('frozen')}")
                        break
        except Exception as e:
            logging.error(f"Balance check failed: {e}")

        # 2) Prices (public)
        try:
            prices = client.get_prices_best_effort(symbols)
            if prices:
                pretty = " | ".join([f"{s}={prices[s]}" for s in prices])
                logging.info(f"Prices: {pretty}")
            else:
                logging.warning("Prices: could not fetch any prices (public endpoints may differ).")
        except Exception as e:
            logging.error(f"Price fetch failed: {e}")

        # 3) Trading gate (ingen ordre i denne versjonen, men vi logger om vi ville vært 'armed')
        ok, reason = guard.can_trade(bot_enabled=bot_enabled, trading_mode=trading_mode, usdt_free=usdt_free)
        if ok:
            logging.warning("Trading is ARMED (live + enabled + min balance). Still NOT placing orders in this version.")
        else:
            logging.info(f"Trading disabled: {reason}")

        logging.info("Heartbeat: bot is alive.")
        time.sleep(30)


if __name__ == "__main__":
    main()
