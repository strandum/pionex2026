import os
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing environment variable: {name}")
    return v

def main():
    logging.info("Booting Pionex bot...")

    # Required secrets (set in Render)
    _api_key = require_env("PIONEX_API_KEY")
    _api_secret = require_env("PIONEX_API_SECRET")

    # Non-secret settings
    trading_mode = os.getenv("TRADING_MODE", "paper")
    symbols = os.getenv("SYMBOLS", "SOL/USDT,ARB/USDT")
    max_usdt_per_trade = float(os.getenv("MAX_USDT_PER_TRADE", "20"))
    max_daily_loss = float(os.getenv("MAX_DAILY_LOSS_USDT", "10"))

    logging.info(f"Mode: {trading_mode}")
    logging.info(f"Symbols: {symbols}")
    logging.info(f"Risk: max_usdt_per_trade={max_usdt_per_trade}, max_daily_loss={max_daily_loss}")

    while True:
        logging.info("Heartbeat: bot is alive.")
        time.sleep(30)

if __name__ == "__main__":
    main()
