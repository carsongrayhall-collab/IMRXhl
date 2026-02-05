#!/usr/bin/env python3
"""
2.3 IMRX 10-minute Prev-Low / Prev-High strategy (no lookahead) with *risk-per-trade* sizing
and Alpaca order execution.

Core trade logic (unchanged):
- Build 30-second candles from live trades.
- Every completed 10-minute block, compute its OHLC.
- For the NEXT 10-minute block:
    support = previous 10-min low
    resistance = previous 10-min high
  Place a BUY LIMIT at support while flat.
  After fill, place a SELL LIMIT at resistance.
  If resistance not hit by the end of the current 10-min block:
      cancel working exit orders and exit at market (time-stop).

What changed vs your prior script:
- Position sizing is now "true risk-per-trade":
    shares = floor( risk_dollars / (entry_price - stop_price) )
  where:
    entry_price ~= support (your buy limit)
    stop_price  = support - max(support*STOP_PCT, STOP_ABS)

- After entry fill, the bot places TWO exit orders:
    1) Take-profit SELL LIMIT at resistance
    2) Protective SELL STOP (market) at stop_price
  If one exit fills, the other is canceled.

Configuration (env vars):
- RISK_DOLLARS: fixed dollars to risk per trade (e.g. 20000)
- RISK_FRACTION: alternatively, risk as a fraction of equity (e.g. 0.15 for 15%)
  If both are set, RISK_DOLLARS wins.
- STOP_PCT: stop distance as % of support (default 0.01 = 1%)
- STOP_ABS: absolute stop distance in dollars (default 0.00). Effective stop distance is max(STOP_PCT*support, STOP_ABS)
- MAX_BP_FRACTION: cap notional by buying power (default 0.95)
- MAX_NOTIONAL_FRACTION: cap notional by equity (default 1.0)
- MAX_SHARES / MIN_SHARES: hard caps
- FLATTEN_ON_START: 1 cancels open orders and closes any position on startup (default 1)

Install:
    pip install alpaca-py python-dateutil

Run (Paper recommended):
    export ALPACA_KEY="..."
    export ALPACA_SECRET="..."
    export ALPACA_PAPER="1"   # 1=paper, 0=live
    export RISK_DOLLARS="20000"
    export STOP_PCT="0.01"
    python imrx_10m_prevlow_prevhigh_risk.py
"""

import os
import math
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Deque, List
from collections import deque

from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.live import StockDataStream

# ----------------------------
# Config
# ----------------------------
SYMBOL = "IMRX"

ALPACA_KEY = os.getenv("ALPACA_KEY", "").strip()
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "").strip()
PAPER = os.getenv("ALPACA_PAPER", "1").strip() != "0"

# Risk sizing
RISK_DOLLARS = os.getenv("RISK_DOLLARS", "").strip()
RISK_DOLLARS = float(RISK_DOLLARS) if RISK_DOLLARS else None

RISK_FRACTION = os.getenv("RISK_FRACTION", "").strip()
RISK_FRACTION = float(RISK_FRACTION) if RISK_FRACTION else None

STOP_PCT = float(os.getenv("STOP_PCT", "0.01"))     # 1% below support by default
STOP_ABS = float(os.getenv("STOP_ABS", "0.00"))     # absolute dollars below support (max with pct)

# Notional caps
MAX_BP_FRACTION = float(os.getenv("MAX_BP_FRACTION", "0.95"))         # of buying power
MAX_NOTIONAL_FRACTION = float(os.getenv("MAX_NOTIONAL_FRACTION", "1.0"))  # of equity

# Safety / sanity defaults
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


def normalize_price(px: float) -> float:
    """Snap prices to common US equity tick sizes."""
    if px is None:
        raise ValueError("price is None")
    if px >= 1.0:
        return round(float(px) + 1e-12, 2)
    return round(float(px) + 1e-12, 4)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


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

        # Order ids
        self.buy_order_id: Optional[str] = None
        self.tp_order_id: Optional[str] = None     # take-profit (limit)
        self.sl_order_id: Optional[str] = None     # stop-loss (stop)

        # Entry info
        self.last_entry_price: Optional[float] = None
        self.last_entry_qty: int = 0

        # Last trade price seen (for sanity / logging)
        self.last_px: Optional[float] = None

    def __repr__(self):
        return (
            f"State(in_position={self.in_position}, qty={self.position_qty}, "
            f"support={self.support}, resistance={self.resistance}, "
            f"buy_id={self.buy_order_id}, tp_id={self.tp_order_id}, sl_id={self.sl_order_id})"
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
            pass

        # Reset local state
        self.state.in_position = False
        self.state.position_qty = 0
        self.state.buy_order_id = None
        self.state.tp_order_id = None
        self.state.sl_order_id = None
        self.state.last_entry_price = None
        self.state.last_entry_qty = 0

    def get_equity(self) -> float:
        acct = self.trading.get_account()
        return float(acct.equity)

    def get_buying_power(self) -> float:
        acct = self.trading.get_account()
        bp = getattr(acct, "buying_power", None)
        try:
            return float(bp) if bp is not None else float(acct.cash)
        except Exception:
            return float(acct.cash)

    def risk_dollars(self) -> float:
        equity = self.get_equity()
        if RISK_DOLLARS is not None and RISK_DOLLARS > 0:
            return float(RISK_DOLLARS)
        if RISK_FRACTION is not None and RISK_FRACTION > 0:
            return float(equity * RISK_FRACTION)
        # sensible default: 12% of equity
        return float(equity * 0.12)

    def compute_stop_price(self, support: float) -> float:
        dist = max(abs(support) * STOP_PCT, STOP_ABS)
        # prevent nonsense
        dist = max(dist, 0.0001)
        stop = support - dist
        # never negative
        stop = max(stop, 0.0001)
        return normalize_price(stop)

    def compute_order_qty_from_risk(self, entry: float, stop: float) -> int:
        """
        Shares = floor( risk_dollars / (entry - stop) ), then clipped by notional caps.
        """
        entry = float(entry)
        stop = float(stop)

        risk_per_share = entry - stop
        if risk_per_share <= 0:
            return 0

        rd = self.risk_dollars()
        qty = int(rd // risk_per_share)
        qty = max(0, qty)

        # Apply notional caps (equity and buying power)
        equity = self.get_equity()
        buying_power = self.get_buying_power()

        max_notional_by_equity = max(0.0, equity * clamp(MAX_NOTIONAL_FRACTION, 0.0, 10.0))
        max_notional_by_bp = max(0.0, buying_power * clamp(MAX_BP_FRACTION, 0.0, 1.0))
        max_notional = min(max_notional_by_equity, max_notional_by_bp) if max_notional_by_bp > 0 else max_notional_by_equity

        if entry > 0 and max_notional > 0:
            qty = min(qty, int(max_notional // entry))

        qty = min(qty, MAX_SHARES)
        if qty < MIN_SHARES:
            return 0
        return qty

    def cancel_order(self, order_id: Optional[str]):
        if not order_id:
            return
        try:
            self.trading.cancel_order_by_id(order_id)
        except Exception:
            pass

    def cancel_all_symbol_orders(self):
        try:
            orders = self.trading.get_orders()
            for o in orders:
                try:
                    if str(getattr(o, "symbol", "")).upper() == SYMBOL and str(getattr(o, "status", "")).lower() in (
                        "new", "accepted", "held", "partially_filled"
                    ):
                        self.trading.cancel_order_by_id(o.id)
                except Exception:
                    continue
        except Exception:
            pass

    def submit_buy_limit(self, support: float, resistance: float):
        if self.state.buy_order_id or self.state.in_position:
            return

        entry = normalize_price(support)
        stop = self.compute_stop_price(entry)
        qty = self.compute_order_qty_from_risk(entry, stop)
        if qty <= 0:
            self._log(f"Risk sizing produced qty=0 (entry={entry:.4f}, stop={stop:.4f}); skipping buy placement.")
            return

        req = LimitOrderRequest(
            symbol=SYMBOL,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            limit_price=entry,
        )
        try:
            o = self.trading.submit_order(req)
        except Exception as e:
            self._log(f"BUY submit failed (limit={entry:.4f}): {e}")
            return

        self.state.buy_order_id = str(o.id)
        self._log(
            f"Placed BUY LIMIT {SYMBOL} qty={qty} @ {entry:.4f} | stop={stop:.4f} | "
            f"risk_dollars≈{self.risk_dollars():.2f} | (order_id={o.id})"
        )

    def submit_take_profit(self, qty: int, resistance: float):
        if self.state.tp_order_id or not self.state.in_position:
            return
        px = normalize_price(resistance)
        req = LimitOrderRequest(
            symbol=SYMBOL,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            limit_price=px,
        )
        try:
            o = self.trading.submit_order(req)
        except Exception as e:
            self._log(f"TAKE-PROFIT submit failed (limit={px:.4f}): {e}")
            return
        self.state.tp_order_id = str(o.id)
        self._log(f"Placed TAKE-PROFIT SELL LIMIT {SYMBOL} qty={qty} @ {px:.4f} (order_id={o.id})")

    def submit_stop_loss(self, qty: int, stop_price: float):
        if self.state.sl_order_id or not self.state.in_position:
            return
        spx = normalize_price(stop_price)
        req = StopOrderRequest(
            symbol=SYMBOL,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            stop_price=spx,
        )
        try:
            o = self.trading.submit_order(req)
        except Exception as e:
            self._log(f"STOP-LOSS submit failed (stop={spx:.4f}): {e}")
            return
        self.state.sl_order_id = str(o.id)
        self._log(f"Placed STOP-LOSS SELL STOP {SYMBOL} qty={qty} stop={spx:.4f} (order_id={o.id})")

    def exit_market(self):
        if not self.state.in_position or self.state.position_qty <= 0:
            return
        qty = self.state.position_qty
        req = MarketOrderRequest(symbol=SYMBOL, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
        try:
            o = self.trading.submit_order(req)
        except Exception as e:
            self._log(f"Market exit submit failed: {e}")
            return
        self._log(f"Market exit SELL {SYMBOL} qty={qty} (order_id={o.id})")

    # ------------- strategy core -------------
    def on_trade_tick_build_candles(self, trade) -> None:
        px = float(trade.price)
        sz = int(trade.size) if trade.size is not None else 0
        ts = trade.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        self.state.last_px = px

        # 30s candle bucket
        bucket = floor_time(ts, CANDLE_SECONDS)
        if self.state.cur_30s is None or self.state.cur_30s.start != bucket:
            if self.state.cur_30s is not None:
                self.state.candles_30s.append(self.state.cur_30s)
            self.state.cur_30s = Candle(start=bucket, open=px, high=px, low=px, close=px, volume=sz)
        else:
            self.state.cur_30s.update(px, sz)

        # Update 10-min bar aggregation
        ten_start = floor_10min(ts)
        if self.state.cur_10m_start is None or ten_start != self.state.cur_10m_start:
            # finalize previous block
            if self.state.cur_10m_ohlc is not None:
                self.state.prev_10m = self.state.cur_10m_ohlc
                self._log(
                    f"10m close {self.state.prev_10m.start.isoformat()} "
                    f"O={self.state.prev_10m.open:.4f} H={self.state.prev_10m.high:.4f} "
                    f"L={self.state.prev_10m.low:.4f} C={self.state.prev_10m.close:.4f}"
                )

            # start new 10-min OHLC
            self.state.cur_10m_start = ten_start
            self.state.cur_10m_ohlc = TenMinBar(start=ten_start, open=px, high=px, low=px, close=px)

            # define levels for new block
            if self.state.prev_10m is not None:
                self.state.support = float(self.state.prev_10m.low)
                self.state.resistance = float(self.state.prev_10m.high)
                self.state.block_end = ten_start + timedelta(minutes=BLOCK_MINUTES)
                self._log(
                    f"New block: support={self.state.support:.4f} resistance={self.state.resistance:.4f} "
                    f"end={self.state.block_end.isoformat()}"
                )

                if not self.state.in_position and self.state.buy_order_id is None:
                    self.submit_buy_limit(self.state.support, self.state.resistance)
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
        If we are in position and block ended and exits not filled,
        cancel working exit orders and exit at market.
        """
        async with self._lock:
            if self.state.block_end is None:
                return
            now = utc_now()
            if now < self.state.block_end:
                return

            if self.state.in_position:
                # cancel TP/SL then exit
                if self.state.tp_order_id:
                    self._log(f"Block ended: cancelling take-profit {self.state.tp_order_id}")
                    self.cancel_order(self.state.tp_order_id)
                    self.state.tp_order_id = None
                if self.state.sl_order_id:
                    self._log(f"Block ended: cancelling stop-loss {self.state.sl_order_id}")
                    self.cancel_order(self.state.sl_order_id)
                    self.state.sl_order_id = None
                self.exit_market()

            # if flat but buy limit still working, cancel it
            if (not self.state.in_position) and self.state.buy_order_id:
                self._log(f"Block ended: cancelling buy limit {self.state.buy_order_id}")
                self.cancel_order(self.state.buy_order_id)
                self.state.buy_order_id = None

            self.state.block_end = now + timedelta(seconds=1)

    async def reconcile_broker_state(self):
        """
        Reconcile local state with Alpaca brokerage (position + working orders).
        """
        async with self._lock:
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
                if not self.state.in_position or self.state.position_qty != pos_qty:
                    self._log(f"RECONCILE: detected live position qty={pos_qty} avg={pos_avg}")
                self.state.in_position = True
                self.state.position_qty = pos_qty
                if self.state.last_entry_price is None and pos_avg is not None:
                    self.state.last_entry_price = pos_avg
                    self.state.last_entry_qty = pos_qty

                # cancel any working buy order to prevent stacking
                if self.state.buy_order_id:
                    self._log(f"RECONCILE: cancelling buy order {self.state.buy_order_id} (position already open)")
                    self.cancel_order(self.state.buy_order_id)
                    self.state.buy_order_id = None

                # detect working TP/SL orders
                tp_id = None
                sl_id = None
                try:
                    orders = self.trading.get_orders()
                    for o in orders:
                        try:
                            if str(getattr(o, "symbol", "")).upper() != SYMBOL:
                                continue
                            st = str(getattr(o, "status", "")).lower()
                            if st not in ("new", "accepted", "held", "partially_filled"):
                                continue
                            side = str(getattr(o, "side", "")).lower()
                            otype = str(getattr(o, "type", "")).lower()
                            if side == "sell" and otype == "limit":
                                tp_id = str(o.id)
                            if side == "sell" and otype == "stop":
                                sl_id = str(o.id)
                        except Exception:
                            continue
                except Exception:
                    pass

                self.state.tp_order_id = tp_id
                self.state.sl_order_id = sl_id

                # if missing exits, recreate based on current block levels
                if self.state.resistance is not None and self.state.tp_order_id is None:
                    self._log(f"RECONCILE: no take-profit working; placing TP for qty={pos_qty} @ {self.state.resistance:.4f}")
                    self.submit_take_profit(pos_qty, self.state.resistance)

                # stop based on current support (best available) or entry price
                if self.state.sl_order_id is None:
                    base = None
                    if self.state.support is not None:
                        base = float(self.state.support)
                    elif self.state.last_entry_price is not None:
                        base = float(self.state.last_entry_price)
                    if base is not None:
                        sp = self.compute_stop_price(base)
                        self._log(f"RECONCILE: no stop-loss working; placing SL for qty={pos_qty} stop={sp:.4f}")
                        self.submit_stop_loss(pos_qty, sp)
            else:
                # no open position
                if self.state.in_position or self.state.position_qty != 0:
                    self._log("RECONCILE: no live position; resetting local position state")
                self.state.in_position = False
                self.state.position_qty = 0
                self.state.tp_order_id = None
                self.state.sl_order_id = None
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
        data has fields: event, order (with id, side, filled_qty, filled_avg_price, status, type)
        """
        async with self._lock:
            order = getattr(data, "order", None)
            if order is None:
                return

            oid = str(order.id)
            status = str(order.status).lower()
            side = str(order.side).lower()
            otype = str(getattr(order, "type", "")).lower()
            filled_qty = int(float(order.filled_qty)) if getattr(order, "filled_qty", None) else 0
            avg_price = float(order.filled_avg_price) if getattr(order, "filled_avg_price", None) else None

            # BUY fills -> enter position and place TP + SL
            if self.state.buy_order_id and oid == self.state.buy_order_id:
                if status in ("filled", "partially_filled"):
                    if filled_qty > 0 and avg_price is not None:
                        self.state.in_position = True
                        self.state.position_qty = filled_qty
                        self.state.last_entry_price = avg_price
                        self.state.last_entry_qty = filled_qty
                        self._log(f"BUY fill update: qty={filled_qty} avg_px={avg_price:.4f} status={status}")

                        # Place TP at resistance and SL below support/entry
                        if self.state.resistance is not None and self.state.tp_order_id is None:
                            self.submit_take_profit(filled_qty, self.state.resistance)

                        base = self.state.support if self.state.support is not None else avg_price
                        stop_px = self.compute_stop_price(float(base))
                        if self.state.sl_order_id is None:
                            self.submit_stop_loss(filled_qty, stop_px)

                if status in ("filled", "canceled", "rejected", "expired"):
                    if status != "filled":
                        self._log(f"BUY order done with status={status}")
                    if status != "partially_filled":
                        self.state.buy_order_id = None

            # TAKE-PROFIT filled -> cancel stop-loss and reset
            if self.state.tp_order_id and oid == self.state.tp_order_id:
                if status == "filled":
                    self._log(f"TAKE-PROFIT filled: qty={filled_qty} avg_px={avg_price if avg_price else 0:.4f}")
                    if self.state.sl_order_id:
                        self._log(f"Cancelling stop-loss {self.state.sl_order_id} after TP fill")
                        self.cancel_order(self.state.sl_order_id)
                        self.state.sl_order_id = None

                    self.state.in_position = False
                    self.state.position_qty = 0
                    self.state.tp_order_id = None
                    self.state.last_entry_price = None
                    self.state.last_entry_qty = 0

                if status in ("canceled", "rejected", "expired"):
                    self._log(f"TAKE-PROFIT order done with status={status}")
                    self.state.tp_order_id = None

            # STOP-LOSS filled -> cancel take-profit and reset
            if self.state.sl_order_id and oid == self.state.sl_order_id:
                if status == "filled":
                    self._log(f"STOP-LOSS filled: qty={filled_qty} avg_px={avg_price if avg_price else 0:.4f}")
                    if self.state.tp_order_id:
                        self._log(f"Cancelling take-profit {self.state.tp_order_id} after SL fill")
                        self.cancel_order(self.state.tp_order_id)
                        self.state.tp_order_id = None

                    self.state.in_position = False
                    self.state.position_qty = 0
                    self.state.sl_order_id = None
                    self.state.last_entry_price = None
                    self.state.last_entry_qty = 0

                if status in ("canceled", "rejected", "expired"):
                    self._log(f"STOP-LOSS order done with status={status}")
                    self.state.sl_order_id = None

    # ------------- run loop -------------
    async def run(self):
        if FLATTEN_ON_START:
            await asyncio.to_thread(self.flatten_on_start)

        self._log(
            f"Starting bot for {SYMBOL} | paper={PAPER} | "
            f"risk_dollars={self.risk_dollars():.2f} | STOP_PCT={STOP_PCT} STOP_ABS={STOP_ABS} | "
            f"MAX_BP_FRACTION={MAX_BP_FRACTION} MAX_NOTIONAL_FRACTION={MAX_NOTIONAL_FRACTION}"
        )

        self.data_stream.subscribe_trades(self.on_trade, SYMBOL)
        self.trade_stream.subscribe_trade_updates(self.on_trade_update)

        async def watcher():
            last_reconcile = utc_now()
            while True:
                await self.check_time_stop()
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
