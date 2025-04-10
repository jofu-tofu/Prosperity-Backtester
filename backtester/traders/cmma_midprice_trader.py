from backtester.datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple
import numpy as np
from collections import deque
import jsonpickle

class Trader:
    """
    CMMA Midprice Trader for SQUID_INK

    This trader uses Cumulative Moving Average Momentum (CMMA) to determine position direction
    and only trades at or near the midprice based on the max_spread parameter.

    Key parameters:
    - lookback: Number of periods to use for CMMA calculation (default: 20)
    - exponent: Exponent for position sizing calculation (default: 2.0)
    - max_position: Maximum allowed position size (default: 50)
    - max_spread: Maximum spread willing to pay (0, 1, or 2) (default: 1)
    - fair_price: Fair price for SQUID_INK (default: 2000)
    - allow_counter_fair: If True, allows positions against fair price direction (default: False)
      When False, prevents going long above fair price or short below fair price
    """
    def __init__(self):
        # CMMA parameters
        self.lookback = 20
        self.upper_threshold = 0.8
        self.lower_threshold = 0.2
        self.exponent = 1.0
        self.max_position = 50  # Maximum allowed position

        # Trading parameters
        self.max_spread = 0  # Maximum spread willing to pay (0, 1, or 2)
        self.fair_price = 2000  # Fair price for SQUID_INK
        self.allow_counter_fair = True  # If True, allows positions against fair price direction

        # Price history for CMMA calculation
        self.price_history: Dict[str, deque] = {}
        self.products = ["SQUID_INK"]  # Add more products as needed

        for product in self.products:
            self.price_history[product] = deque(maxlen=self.lookback + 1)

    def calculate_cmma(self, prices: deque) -> float:
        """
        Compute Cumulative Moving Average Momentum (CMMA)
        """
        if len(prices) < self.lookback:
            return 0.5  # Default to neutral when insufficient data

        price_list = list(prices)
        current_price = price_list[-1]
        ma = sum(price_list[-self.lookback-1:-1]) / self.lookback

        # Calculate raw CMMA
        raw_cmma = (current_price - ma) / np.sqrt(self.lookback + 1)

        # Normalize using sigmoid function
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        return sigmoid(raw_cmma)

    def calculate_position_size(self, cmma: float, current_price: float) -> int:
        """
        Calculate desired position size based on CMMA value and current price
        """
        # Center CMMA around 0.5 and scale to [-1, 1]
        cmma_scaled = 2 * (cmma - 0.5)

        # Calculate base position using exponential scaling
        if cmma_scaled > 0:
            # High CMMA -> short position
            base_position = -min(abs(cmma_scaled) ** self.exponent, 1.0)
            # Don't short below fair price unless allowed
            if current_price < self.fair_price and not self.allow_counter_fair:
                base_position = 0
        else:
            # Low CMMA -> long position
            base_position = min(abs(cmma_scaled) ** self.exponent, 1.0)
            # Don't go long above fair price unless allowed
            if current_price > self.fair_price and not self.allow_counter_fair:
                base_position = 0

        # Scale to max position size and round to integer
        target_position = int(base_position * self.max_position)

        return target_position

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        """
        Calculate mid price from order depth
        """
        mm_ask = max(order_depth.sell_orders.keys()) if len(order_depth.sell_orders) > 0 else None
        mm_bid = min(order_depth.buy_orders.keys()) if len(order_depth.buy_orders) > 0 else None

        if mm_ask is None or mm_bid is None:
            return None

        # Calculate midprice and round to nearest integer
        return round((mm_ask + mm_bid) / 2)

    def get_adjusted_prices(self, order_depth: OrderDepth) -> tuple:
        """
        Calculate adjusted prices based on midprice and max spread
        """
        mid_price = self.get_mid_price(order_depth)
        if mid_price is None:
            return None, None

        # Adjust prices based on max spread parameter
        buy_price = mid_price - self.max_spread
        sell_price = mid_price + self.max_spread

        return buy_price, sell_price

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """
        Main trading logic - only trades at midprice
        """
        # Initialize trader data from state or create new if none exists
        trader_data = {}
        if state.traderData:
            try:
                trader_data = jsonpickle.decode(state.traderData)
            except:
                trader_data = {}

        # Store CMMA values for each product in trader_data
        if 'cmma_values' not in trader_data:
            trader_data['cmma_values'] = {}

        result = {}

        for product in self.products:
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                mid_price = self.get_mid_price(order_depth)
                if mid_price is None:
                    continue

                # Get adjusted prices based on spread parameter
                buy_price, sell_price = self.get_adjusted_prices(order_depth)
                if buy_price is None or sell_price is None:
                    continue

                # Store prices in trader_data
                if 'prices' not in trader_data:
                    trader_data['prices'] = {}
                trader_data['prices'][product] = {
                    'mid_price': mid_price,
                    'buy_price': buy_price,
                    'sell_price': sell_price
                }

                self.price_history[product].append(mid_price)
                if len(self.price_history[product]) < self.lookback:
                    continue

                cmma = self.calculate_cmma(self.price_history[product])

                # Store CMMA value in trader_data
                trader_data['cmma_values'][product] = cmma

                current_position = state.position.get(product, 0)
                target_position = self.calculate_position_size(cmma, mid_price)
                position_difference = target_position - current_position

                # Create orders at adjusted prices based on direction
                orders: List[Order] = []
                if position_difference > 0:  # Need to buy
                    orders.append(Order(product, buy_price, position_difference))
                elif position_difference < 0:  # Need to sell
                    orders.append(Order(product, sell_price, position_difference))

                if orders:
                    result[product] = orders

        # Store parameters and timestamp for reference
        trader_data['last_timestamp'] = state.timestamp
        trader_data['parameters'] = {
            'lookback': self.lookback,
            'exponent': self.exponent,
            'max_position': self.max_position,
            'max_spread': self.max_spread,
            'fair_price': self.fair_price,
            'allow_counter_fair': self.allow_counter_fair
        }

        # Encode trader_data to string
        serialized_trader_data = jsonpickle.encode(trader_data)

        # No conversions needed for SQUID_INK
        conversions = 0

        return result, conversions, serialized_trader_data
