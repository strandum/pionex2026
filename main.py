import os
import time
import hmac
import hashlib
import logging
from typing import Dict, Any, Optional, List

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
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


# ---------- indicators ----------
def ema(values: List[float], period: int) -> Optional[float]:
    if period <= 0 or len(values) < period:
        return None
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e


def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return None
    trs: List[float] = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / period


def vwap(highs: List[float], lows: List[float], closes: List[float], volumes: List[float], lookback: int = 100) -> Optional[float]:
    n = min(lookback, len(closes))
    if n <= 5:
        return None
    tpv = 0.0
    vol = 0.0
    for i in range(len(closes) - n, len(closes)):
        typical = (highs[i] + lows[i] + closes[i]) / 3.0
        v = volumes[i]
        tpv += typical * v
        vol += v
    if vol <= 0:
        return None
    return tpv / vol


# ----------(SIM): trend + pullback signal
def trend_state_15m(closes_15m: List[float]) -> str:
    e50 = ema(closes_15m, 50)
    e200 = ema(closes_15m, 200)
    if e50 is None or e200 is None:
        return "NO_TRADE"
    last = closes_15m[-1]
    if e50 > e200 and last > e50 and last > e200:
        return "BULL"
    if e50 < e200 and last < e50 and last < e200:
        return "BEAR"
    return "NO_TRADE"


def pullback_signal_1m(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
    trend: str,
    vwap_lookback: int = 100,
    atr_period: int = 14,
    atr_min: float = 0.0,
) -> Optional[Dict[str, float]]:
    vw = vwap(highs, lows, closes, volumes, lookback=vwap_lookback)
    a = atr(highs, lows, closes, period=atr_period)
    if vw is None or a is None:
        return None
    if a <= atr_min:
        return None
    if len(closes) < 3:
        return None

    c_2, c_1, c_0 = closes[-3], closes[-2], closes[-1]

    # LONG in BULL: above -> dip to/below -> reclaim above VWAP
    if trend == "BULL":
        if (c_2 > vw) and (c_1 <= vw) and (c_0 > vw):
            entry = c_0
            sl = entry - (1.2 * a)
            tp = entry + (2.0 * a)
            return {"side": 1, "entry": entry, "sl": sl, "tp": tp, "atr": a, "vwap": vw}

    # SHORT in BEAR: below -> pop to/above -> reject below VWAP
    if trend == "BEAR":
        if (c_2 < vw) and (c_1 >= vw) and (c_0 < vw):
            entry = c_0
            sl = entry + (1.2 * a)
            tp = entry - (2.0 * a)
            return {"side": -1, "entry": entry, "sl": sl, "tp": tp, "atr": a, "vwap": vw}

    return None


# ---------- Pionex client ----------
class PionexClient:
    """
    Minimal Pionex client:
    - Private GET for balances (signed)
    - Public GET for tickers + klines (no signature)
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
        data = resp.json()
        if not resp.ok:
            raise RuntimeError(f"HTTP {resp.status_code}: {data}")
        return data

    def public_get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        query = ""
        if params:
            query = self._build_sorted_query(params)
        url = f"{self.base_url}{path}" + (f"?{query}" if query else "")
        resp = self.session.get(url, timeout=15)
        data = resp.json()
        if not resp.ok:
            raise RuntimeError(f"HTTP {resp.status_code} (public): {data}")
        return data

    # private
    def get_balances(self) -> Dict[str, Any]:
        return self.private_get("/api/v1/account/balances")

    # public
    def get_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Uses market/tickers and filters in-code. Your logs showed this already works.
        """
        prices: Dict[str, float] = {}
        r = self.public_get("/api/v1/market/tickers")
        if r.get("result") is not True:
            return prices
        data = r.get("data") or {}
        tickers = data.get("tickers") or data.get("data") or data.get("list") or []
        if not isinstance(tickers, list):
            return prices

        wanted = {s.replace("/", "_"): s for s in symbols}  # SOL_USDT -> SOL/USDT
        for t in tickers:
            ts = t.get("symbol") or t.get("symbolName") or t.get("market")
            if ts in wanted:
                last = t.get("last") or t.get("close") or t.get("price")
                if last is not None:
                    prices[wanted[ts]] = float(last)
        return prices

    def get_candles(self, symbol: str, interval_seconds: int, limit: int) -> Optional[List[Dict[str, float]]]:
        """
        Pionex klines:
          GET /api/v1/market/klines?symbol=SOL_USDT&interval=60&limit=200
        Returns rows like: [time, open, close, high, low, volume]
        """
        sym = symbol.replace("/", "_")  # SOL_USDT
        r = self.public_get(
            "/api/v1/market/klines",
            params={"symbol": sym, "interval": interval_seconds, "limit": limit},
        )
        if r.get("result") is not True:
            return None

        raw = (r.get("data") or {}).get("klines")
        if raw is None:
            # some APIs return list directly in data
            raw = r.get("data")
        if not isinstance(raw, list) or not raw:
            return None

        candles: List[Dict[str, float]] = []
        for row in raw:
            if not isinstance(row, (list, tuple)) or len(row) < 6:
                continue
            _, o, c, h, l, v = row[:6]
            candles.append({
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(v),
            })

        if len(candles) < 50:
            return None
        return candles


def main():
    logging.info("Booting Pionex bot (SIM STRATEGY D)...")

    api_key = require_env("PIONEX_API_KEY")
    api_secret = require_env("PIONEX_API_SECRET")

    trading_mode = os.getenv("TRADING_MODE", "paper")
    symbols = parse_symbols(os.getenv("SYMBOLS", "SOL/USDT,ARB/USDT"))

    # Guardrails (still SIM-only, but keep for safety)
    bot_enabled = env_bool("BOT_ENABLED", default=False)  # keep false
    min_usdt_to_trade = float(os.getenv("MIN_USDT_TO_TRADE", "10"))
    max_trades_per_hour = int(os.getenv("MAX_TRADES_PER_HOUR", "6"))

    # Strategy params
    vwap_lookback = int(os.getenv("VWAP_LOOKBACK", "100"))
    atr_period = int(os.getenv("ATR_PERIOD", "14"))
    atr_min = float(os.getenv("ATR_MIN", "0"))
    candle_limit_1m = int(os.getenv("CANDLE_LIMIT_1M", "240"))
    candle_limit_15m = int(os.getenv("CANDLE_LIMIT_15M", "260"))  # needs >= 200 for EMA200
    poll_seconds = int(os.getenv("POLL_SECONDS", "30"))

    client = PionexClient(api_key=api_key, api_secret=api_secret)

    logging.info(f"Mode: {trading_mode}")
    logging.info(f"Symbols: {','.join(symbols)}")
    logging.info(f"Guards: BOT_ENABLED={bot_enabled}, MIN_USDT_TO_TRADE={min_usdt_to_trade}, MAX_TRADES_PER_HOUR={max_trades_per_hour}")
    logging.info(f"Strategy: EMA(50/200) on 15m, Pullback+VWAP on 1m, ATR({atr_period}), VWAP_LOOKBACK={vwap_lookback}")

    while True:
        usdt_free = 0.0

        # balances
        try:
            r = client.get_balances()
            if r.get("result") is True:
                balances = (r.get("data") or {}).get("balances", [])
                for b in balances:
                    if b.get("coin") == "USDT":
                        usdt_free = float(b.get("free", 0) or 0)
                        logging.info(f"Balances: USDT free={usdt_free} frozen={b.get('frozen')}")
                        break
            else:
                logging.warning(f"Balances result=false: {r}")
        except Exception as e:
            logging.error(f"Balance check failed: {e}")

        # prices
        try:
            prices = client.get_prices(symbols)
            if prices:
                pretty = " | ".join([f"{s}={prices[s]}" for s in prices])
                logging.info(f"Prices: {pretty}")
        except Exception as e:
            logging.error(f"Price fetch failed: {e}")

        # strategy loop
        for sym in symbols:
            try:
                c15 = client.get_candles(sym, interval_seconds=900, limit=candle_limit_15m)  # 15m
                c1 = client.get_candles(sym, interval_seconds=60, limit=candle_limit_1m)    # 1m

                if not c15 or not c1:
                    logging.warning(f"[SIM] {sym}: candle fetch failed (15m={bool(c15)} 1m={bool(c1)}).")
                    continue

                closes_15 = [c["close"] for c in c15]
                trend = trend_state_15m(closes_15)

                highs_1 = [c["high"] for c in c1]
                lows_1 = [c["low"] for c in c1]
                closes_1 = [c["close"] for c in c1]
                vols_1 = [c["volume"] for c in c1]

                sig = pullback_signal_1m(
                    highs=highs_1,
                    lows=lows_1,
                    closes=closes_1,
                    volumes=vols_1,
                    trend=trend,
                    vwap_lookback=vwap_lookback,
                    atr_period=atr_period,
                    atr_min=atr_min,
                )

                last_price = closes_1[-1]
                if sig is None:
                    logging.info(f"[SIM] {sym}: TREND={trend} price={last_price}")
                else:
                    side = "BUY" if sig["side"] == 1 else "SELL"
                    logging.warning(
                        f"[SIM] SIGNAL {side} {sym} | TREND={trend} "
                        f"entry={sig['entry']:.6f} vwap={sig['vwap']:.6f} atr={sig['atr']:.6f} "
                        f"SL={sig['sl']:.6f} TP={sig['tp']:.6f} | "
                        f"USDT_free={usdt_free:.2f} BOT_ENABLED={bot_enabled} MODE={trading_mode}"
                    )

            except Exception as e:
                logging.error(f"[SIM] {sym}: strategy loop error: {e}")

        if bot_enabled and trading_mode.strip().lower() == "live":
            logging.warning("Trading is ARMED but this build is SIM-only (no orders implemented yet).")
        else:
            logging.info("Trading disabled (SIM-only / guards).")

        logging.info("Heartbeat: bot is alive.")
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
