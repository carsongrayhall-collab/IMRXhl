#!/usr/bin/env python3
"""
2.1 Adds sell funcitonality IMRX 10-minute Prev-Low / Prev-High strategy (no lookahead) with Kelly sizing
and Alpaca order execution.

Exact logic implemented:
- Build 30-second candles from live trades.
- Every completed 10-minute block, compute its OHLC.
- For the NEXT 10-minute block:
    support = previous 10-min low
    resistance = previous 10-min high
  Place a BUY LIMIT at support while flat.
  After fill, place a SELL LIMIT at resistance.
  If resistance not hit by the end of the current 10-min block:
      cancel sell limit (if any) and exit at market (time-stop).

Position sizing:
- Expanding-window Kelly fraction from realized trade returns (wins/losses and payoff ratio).
- f* clipped to [0, 1]; multiply by kelly_multiplier in {1.0, 0.5, 0.25}.
- Uses account equity * (f* * kelly_multiplier) to size shares.

Notes:
- This script uses Alpaca's streaming trade feed to build 30s candles.
- You must have an Alpaca data subscription that provides real-time trades for IMRX.
- Fill behavior: assumes limit orders may fill partially; it handles fills via trade updates stream.

Install:
    pip install alpaca-py python-dateutil

Run (Paper recommended):
    export ALPACA_KEY="..."
    export ALPACA_SECRET="..."
    export ALPACA_PAPER="1"   # 1=paper, 0=live
    python imrx_10m_prevlow_prevhigh_kelly.py

Optional:
    export KELLY_MULT="0.5"   # 1.0 / 0.5 / 0.25
    export MAX_FRACTION="1.0" # cap Kelly fraction (default 1.0)
"""

import os
import math
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Deque, List, Tuple
from collections import deque

from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.live import StockDataStream

# ----------------------------
# Config
# ----------------------------
SYMBOL = "IMRX"

ALPACA_KEY = os.getenv("ALPACA_KEY", "").strip()
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "").strip()
PAPER = os.getenv("ALPACA_PAPER", "1").strip() != "0"

KELLY_MULT = float(os.getenv("KELLY_MULT", "1.0"))
MAX_FRACTION = float(os.getenv("MAX_FRACTION", "1.0"))

# Hardwired bootstrap sizing (2% of equity) until Kelly is ready
BOOTSTRAP_FRACTION = 0.02

# Safety / sanity defaults
MIN_KELLY_TRADES = 30
MIN_WINS_LOSSES = 5
MAX_SHARES = int(os.getenv("MAX_SHARES", "1000000"))  # hard cap
MIN_SHARES = int(os.getenv("MIN_SHARES", "1"))

# Candle building
CANDLE_SECONDS = 30
BLOCK_MINUTES = 10

# If you want to force a "flat" start (cancel existing orders, close position), set to 1
FLATTEN_ON_START = os.getenv("FLATTEN_ON_START", "1").strip() == "1"


# ----------------------------
# Helpers
# ----------------------------
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def floor_time(dt: datetime, seconds: int) -> datetime:
    """Floor dt to nearest interval in seconds."""
    epoch = int(dt.timestamp())
    floored = epoch - (epoch % seconds)
    return datetime.fromtimestamp(floored, tz=timezone.utc)


def floor_10min(dt: datetime) -> datetime:
    """Floor dt to 10-minute boundary."""
    epoch = int(dt.timestamp())
    block = BLOCK_MINUTES * 60
    floored = epoch - (epoch % block)
    return datetime.fromtimestamp(floored, tz=timezone.utc)


def normalize_limit_price(px: float) -> float:
    """Snap limit prices to common US equity tick sizes.

    Typical rule: >= $1.00 trades in $0.01 increments; < $1.00 may trade in $0.0001.
    Alpaca will reject sub-penny prices that violate the symbol's tick size.
    """
    if px is None:
        raise ValueError('price is None')
    if px >= 1.0:
        return round(float(px) + 1e-12, 2)
    return round(float(px) + 1e-12, 4)



@dataclass
class Candle:
    start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0

    def update(self, px: float, size: int):
        self.high = max(self.high, px)
        self.low = min(self.low, px)
        self.close = px
        self.volume += int(size)


@dataclass
class TenMinBar:
    start: datetime
    open: float
    high: float
    low: float
    close: float


def compute_kelly_fraction(returns: List[float]) -> float:
    """
    Standard "edge-based" Kelly fraction using win rate and payoff ratio:
      f = (b*p - (1-p)) / b
    where b = avg_win/avg_loss.
    """
    if len(returns) < MIN_KELLY_TRADES:
        return 0.0

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]

    if len(wins) < MIN_WINS_LOSSES or len(losses) < MIN_WINS_LOSSES:
        return 0.0

    p = len(wins) / len(returns)
    avg_win = sum(wins) / len(wins)
    avg_loss = abs(sum(losses) / len(losses))
    if avg_win <= 0 or avg_loss <= 0:
        return 0.0

    b = avg_win / avg_loss
    f = (b * p - (1 - p)) / b
    if math.isnan(f) or math.isinf(f):
        return 0.0

    f = max(0.0, min(MAX_FRACTION, f))
    return f


# ----------------------------
# Strategy / Execution State
# ----------------------------
class StrategyState:
    def __init__(self):
        # Candle aggregation
        self.cur_30s: Optional[Candle] = None
        self.candles_30s: Deque[Candle] = deque(maxlen=4000)

        # 10-min bars
        self.cur_10m_start: Optional[datetime] = None
        self.cur_10m_ohlc: Optional[TenMinBar] = None
        self.prev_10m: Optional[TenMinBar] = None

        # Levels for current 10-min block (derived from prev_10m)
        self.support: Optional[float] = None
        self.resistance: Optional[float] = None
        self.block_end: Optional[datetime] = None

        # Trading state
        self.in_position: bool = False
        self.position_qty: int = 0

        self.buy_order_id: Optional[str] = None
        self.sell_order_id: Optional[str] = None

        # For Kelly
        self.trade_returns: List[float] = []
        self.last_entry_price: Optional[float] = None
        self.last_entry_qty: int = 0

        # Last trade price seen (for sizing / sanity)
        self.last_px: Optional[float] = None

    def __repr__(self):
        return (
            f"State(in_position={self.in_position}, qty={self.position_qty}, "
            f"support={self.support}, resistance={self.resistance}, "
            f"buy_id={self.buy_order_id}, sell_id={self.sell_order_id})"
        )


# ----------------------------
# Alpaca Bot
# ----------------------------
class AlpacaIMRXBot:
    def __init__(self):
        if not ALPACA_KEY or not ALPACA_SECRET:
            raise SystemExit("Set ALPACA_KEY and ALPACA_SECRET environment variables.")

        self.trading = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=PAPER)
        self.trade_stream = TradingStream(ALPACA_KEY, ALPACA_SECRET, paper=PAPER)
        self.data_stream = StockDataStream(ALPACA_KEY, ALPACA_SECRET)

        self.state = StrategyState()
        self._lock = asyncio.Lock()

    # ------------- trading utilities -------------
    def _log(self, msg: str):
        ts = utc_now().isoformat()
        print(f"[{ts}] {msg}", flush=True)

    def flatten_on_start(self):
        # Cancel open orders and close position (if any)
        self._log("Flattening on start: cancelling open orders and closing any position...")
        try:
            orders = self.trading.get_orders()
            for o in orders:
                try:
                    self.trading.cancel_order_by_id(o.id)
                except Exception:
                    pass
        except Exception as e:
            self._log(f"Warning: could not list/cancel orders: {e}")

        try:
            pos = self.trading.get_open_position(SYMBOL)
            qty = int(float(pos.qty))
            if qty != 0:
                side = OrderSide.SELL if qty > 0 else OrderSide.BUY
                req = MarketOrderRequest(symbol=SYMBOL, qty=abs(qty), side=side, time_in_force=TimeInForce.DAY)
                try:
                    self.trading.submit_order(req)
                except Exception as e:
                    self._log(f"Flatten submit failed: {e}")
                else:
                    self._log(f"Submitted market order to flatten {SYMBOL}: {side} {abs(qty)}")
        except Exception:
            # No position is fine
            pass

        # Reset local state
        self.state.in_position = False
        self.state.position_qty = 0
        self.state.buy_order_id = None
        self.state.sell_order_id = None
        self.state.last_entry_price = None
        self.state.last_entry_qty = 0

    def get_equity(self) -> float:
        acct = self.trading.get_account()
        return float(acct.equity)

    def compute_order_qty(self, price: float) -> int:
        # Kelly sizing based on realized trade returns
        f_star = compute_kelly_fraction(self.state.trade_returns)

        # Bootstrap to avoid Kelly "deadlock" at startup (no trades -> Kelly=0 -> qty=0 forever)
        if f_star == 0.0:
            f = BOOTSTRAP_FRACTION
        else:
            f = max(0.0, min(MAX_FRACTION, f_star)) * KELLY_MULT

        equity = self.get_equity()
        notional = equity * f
        if notional <= 0 or price <= 0:
            return 0

        qty = int(notional // price)
        qty = max(0, min(MAX_SHARES, qty))
        if qty < MIN_SHARES:
            return 0
        return qty

    def submit_buy_limit(self, limit_price: float):
        if self.state.buy_order_id or self.state.in_position:
            return

        last_px = self.state.last_px
        if last_px is None:
            return

        qty = self.compute_order_qty(last_px)
        if qty <= 0:
            self._log("Kelly sizing produced qty=0; skipping buy placement.")
            return

        req = LimitOrderRequest(
            symbol=SYMBOL,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            limit_price=normalize_limit_price(limit_price),
        )
        try:
            o = self.trading.submit_order(req)
        except Exception as e:
            self._log(f"BUY submit failed (limit={normalize_limit_price(limit_price):.4f}): {e}")
            return
        self.state.buy_order_id = str(o.id)
        self._log(f"Placed BUY LIMIT {SYMBOL} qty={qty} @ {normalize_limit_price(limit_price):.4f} (order_id={o.id})")

    def submit_sell_limit(self, qty: int, limit_price: float):
        if self.state.sell_order_id or not self.state.in_position:
            return
        req = LimitOrderRequest(
            symbol=SYMBOL,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            limit_price=normalize_limit_price(limit_price),
        )
        try:
            o = self.trading.submit_order(req)
        except Exception as e:
            self._log(f"SELL submit failed (limit={normalize_limit_price(limit_price):.4f}): {e}")
            return
        self.state.sell_order_id = str(o.id)
        self._log(f"Placed SELL LIMIT {SYMBOL} qty={qty} @ {normalize_limit_price(limit_price):.4f} (order_id={o.id})")

    def cancel_order(self, order_id: Optional[str]):
        if not order_id:
            return
        try:
            self.trading.cancel_order_by_id(order_id)
        except Exception:
            pass

    def exit_market(self):
        if not self.state.in_position or self.state.position_qty <= 0:
            return
        qty = self.state.position_qty
        req = MarketOrderRequest(symbol=SYMBOL, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
        try:
            o = self.trading.submit_order(req)
        except Exception as e:
            self._log(f"TIME-STOP market exit submit failed: {e}")
            return
        self._log(f"TIME-STOP: Market exit SELL {SYMBOL} qty={qty} (order_id={o.id})")

    # ------------- strategy core -------------
    def on_trade_tick_build_candles(self, trade) -> None:
        """
        trade has .timestamp, .price, .size
        """
        px = float(trade.price)
        sz = int(trade.size) if trade.size is not None else 0
        ts = trade.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        self.state.last_px = px

        # 30s candle bucket
        bucket = floor_time(ts, CANDLE_SECONDS)
        if self.state.cur_30s is None or self.state.cur_30s.start != bucket:
            # finalize old
            if self.state.cur_30s is not None:
                self.state.candles_30s.append(self.state.cur_30s)
            # new candle
            self.state.cur_30s = Candle(start=bucket, open=px, high=px, low=px, close=px, volume=sz)
        else:
            self.state.cur_30s.update(px, sz)

        # Update 10-min bar aggregation (based on 30s candles)
        ten_start = floor_10min(ts)
        if self.state.cur_10m_start is None or ten_start != self.state.cur_10m_start:
            # new 10-min block started -> finalize previous block if exists
            if self.state.cur_10m_ohlc is not None:
                self.state.prev_10m = self.state.cur_10m_ohlc
                self._log(
                    f"10m close {self.state.prev_10m.start.isoformat()} "
                    f"O={self.state.prev_10m.open:.4f} H={self.state.prev_10m.high:.4f} "
                    f"L={self.state.prev_10m.low:.4f} C={self.state.prev_10m.close:.4f}"
                )

            # start new 10-min OHLC using current price as seed
            self.state.cur_10m_start = ten_start
            self.state.cur_10m_ohlc = TenMinBar(start=ten_start, open=px, high=px, low=px, close=px)

            # Define tradable levels for this NEW block from the PREVIOUS completed 10m bar
            if self.state.prev_10m is not None:
                self.state.support = float(self.state.prev_10m.low)
                self.state.resistance = float(self.state.prev_10m.high)
                self.state.block_end = ten_start + timedelta(minutes=BLOCK_MINUTES)
                self._log(f"New block: support={self.state.support:.4f} resistance={self.state.resistance:.4f} end={self.state.block_end.isoformat()}")

                # Place buy limit at support if flat
                if not self.state.in_position and self.state.buy_order_id is None:
                    self.submit_buy_limit(self.state.support)
            else:
                self.state.support = None
                self.state.resistance = None
                self.state.block_end = ten_start + timedelta(minutes=BLOCK_MINUTES)

        else:
            # update ongoing 10m OHLC
            b = self.state.cur_10m_ohlc
            if b is not None:
                b.high = max(b.high, px)
                b.low = min(b.low, px)
                b.close = px

    async def check_time_stop(self):
        """
        Called periodically. If we are in position and block ended and sell not filled,
        cancel sell and exit at market.
        """
        async with self._lock:
            if self.state.block_end is None:
                return
            now = utc_now()
            if now < self.state.block_end:
                return

            # block ended
            if self.state.in_position:
                # if sell working, cancel it then market exit
                if self.state.sell_order_id:
                    self._log(f"Block ended: cancelling sell limit {self.state.sell_order_id}")
                    self.cancel_order(self.state.sell_order_id)
                    self.state.sell_order_id = None
                self.exit_market()

            # If we are flat but buy limit is still working from prior block, cancel it.
            if (not self.state.in_position) and self.state.buy_order_id:
                self._log(f"Block ended: cancelling buy limit {self.state.buy_order_id}")
                self.cancel_order(self.state.buy_order_id)
                self.state.buy_order_id = None

            # Advance block_end forward to prevent repeated triggers (until next tick sets it)
            self.state.block_end = now + timedelta(seconds=1)


    async def reconcile_broker_state(self):
        """Periodically reconcile local state with Alpaca brokerage.

        This fixes cases where trade update events are missed/delayed, which can
        otherwise prevent placing the SELL order and allow repeated buys.
        """
        async with self._lock:
            # 1) Check actual open position
            pos_qty = 0
            pos_avg = None
            try:
                pos = self.trading.get_open_position(SYMBOL)
                pos_qty = int(float(pos.qty))
                pos_avg = float(pos.avg_entry_price) if getattr(pos, "avg_entry_price", None) else None
            except Exception:
                pos_qty = 0
                pos_avg = None

            if pos_qty > 0:
                # We are long in reality
                if not self.state.in_position or self.state.position_qty != pos_qty:
                    self._log(f"RECONCILE: detected live position qty={pos_qty} avg={pos_avg}")
                self.state.in_position = True
                self.state.position_qty = pos_qty
                if self.state.last_entry_price is None and pos_avg is not None:
                    self.state.last_entry_price = pos_avg
                    self.state.last_entry_qty = pos_qty

                # Cancel any working buy order to prevent stacking
                if self.state.buy_order_id:
                    self._log(f"RECONCILE: cancelling buy order {self.state.buy_order_id} (position already open)")
                    self.cancel_order(self.state.buy_order_id)
                    self.state.buy_order_id = None

                # 2) Detect if a sell order is already working
                working_sell_id = None
                try:
                    orders = self.trading.get_orders()
                    for o in orders:
                        try:
                            if str(getattr(o, "symbol", "")).upper() != SYMBOL:
                                continue
                            if str(getattr(o, "side", "")).lower() == "sell" and str(getattr(o, "status", "")).lower() in ("new", "accepted", "held", "partially_filled"):
                                working_sell_id = str(o.id)
                                break
                        except Exception:
                            continue
                except Exception:
                    pass

                if working_sell_id:
                    self.state.sell_order_id = working_sell_id
                else:
                    # No sell working: place one at the strategy resistance for the CURRENT block
                    target = self.state.resistance
                    if target is None:
                        # fall back to last completed 10m high if available
                        if self.state.prev_10m is not None:
                            target = float(self.state.prev_10m.high)
                    if target is not None:
                        self._log(f"RECONCILE: no sell working; placing SELL for qty={pos_qty} @ {target:.4f}")
                        self.submit_sell_limit(pos_qty, target)
            else:
                # No open position in reality
                if self.state.in_position or self.state.position_qty != 0:
                    self._log("RECONCILE: no live position; resetting local position state")
                self.state.in_position = False
                self.state.position_qty = 0
                self.state.sell_order_id = None
                self.state.last_entry_price = None
                self.state.last_entry_qty = 0

    # ------------- stream callbacks -------------
    async def on_trade(self, trade):
        async with self._lock:
            try:
                self.on_trade_tick_build_candles(trade)
            except Exception as e:
                self._log(f"on_trade handler error (continuing): {e}")

    async def on_trade_update(self, data):
        """
        Handles order fill lifecycle.
        data has fields: event, order (with id, side, filled_qty, filled_avg_price, status)
        """
        async with self._lock:
            event = getattr(data, "event", None)
            order = getattr(data, "order", None)
            if order is None:
                return

            oid = str(order.id)
            side = str(order.side).lower()
            status = str(order.status).lower()
            filled_qty = int(float(order.filled_qty)) if getattr(order, "filled_qty", None) else 0
            avg_price = float(order.filled_avg_price) if getattr(order, "filled_avg_price", None) else None

            # BUY fills -> enter position and place SELL limit
            if self.state.buy_order_id and oid == self.state.buy_order_id:
                if status in ("filled", "partially_filled"):
                    if filled_qty > 0 and avg_price is not None:
                        self.state.in_position = True
                        self.state.position_qty = filled_qty
                        self.state.last_entry_price = avg_price
                        self.state.last_entry_qty = filled_qty
                        self._log(f"BUY fill update: qty={filled_qty} avg_px={avg_price:.4f} status={status}")

                        # Place sell limit at resistance (if defined)
                        if self.state.resistance is not None and self.state.sell_order_id is None:
                            self.submit_sell_limit(filled_qty, self.state.resistance)

                if status in ("filled", "canceled", "rejected", "expired"):
                    if status != "filled":
                        self._log(f"BUY order done with status={status}")
                    # clear buy id when it is no longer working
                    if status != "partially_filled":
                        self.state.buy_order_id = None

            # SELL fills -> realize return and reset
            if self.state.sell_order_id and oid == self.state.sell_order_id:
                if status in ("filled", "partially_filled"):
                    if filled_qty > 0 and avg_price is not None:
                        self._log(f"SELL fill update: qty={filled_qty} avg_px={avg_price:.4f} status={status}")

                if status == "filled":
                    if self.state.last_entry_price is not None and avg_price is not None and self.state.last_entry_qty > 0:
                        # compute realized return on the position
                        ret = (avg_price / self.state.last_entry_price) - 1.0
                        self.state.trade_returns.append(ret)
                        self._log(f"REALIZED trade return: {ret*100:.3f}% (n={len(self.state.trade_returns)})")

                    self.state.in_position = False
                    self.state.position_qty = 0
                    self.state.sell_order_id = None
                    self.state.last_entry_price = None
                    self.state.last_entry_qty = 0

                if status in ("canceled", "rejected", "expired"):
                    self._log(f"SELL order done with status={status}")
                    self.state.sell_order_id = None

    # ------------- run loop -------------
    async def run(self):
        if FLATTEN_ON_START:
            await asyncio.to_thread(self.flatten_on_start)

        self._log(f"Starting bot for {SYMBOL} | paper={PAPER} | KELLY_MULT={KELLY_MULT}")

        # Subscribe to live trades for candle building
        self.data_stream.subscribe_trades(self.on_trade, SYMBOL)

        # Subscribe to order updates for fill tracking
        self.trade_stream.subscribe_trade_updates(self.on_trade_update)

        # Periodic time-stop watcher
        async def watcher():
            last_reconcile = utc_now()
            while True:
                await self.check_time_stop()
                # Reconcile every 15 seconds
                if (utc_now() - last_reconcile).total_seconds() >= 15:
                    await self.reconcile_broker_state()
                    last_reconcile = utc_now()
                await asyncio.sleep(1)

        await asyncio.gather(
            self.data_stream._run_forever(),
            self.trade_stream._run_forever(),
            watcher(),
        )


def main():
    bot = AlpacaIMRXBot()
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()

