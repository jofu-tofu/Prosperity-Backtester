# backtester/constants.py
"""
This module defines constants and default configurations used by the backtester.
It includes fair value calculation methods, position limits, and other simulation parameters.
"""

import numpy as np
from typing import Callable, Dict, Any

# Attempt to import OrderDepth, handle potential ImportError if datamodel changes location
try:
    from .datamodel import OrderDepth
except ImportError:
    try:
        from datamodel import OrderDepth # Fallback for direct execution or different structure
    except ImportError:
        print("Warning: Could not import OrderDepth from datamodel in constants.py. PnL calculations might fail if functions require it.")
        # Define a dummy type for type hints if import fails
        OrderDepth = Any


# --- Simulation Parameters ---

MAX_TIMESTAMP = 199900  # Default maximum timestamp for a standard round (adjust if needed)
# --- Data Loading Parameters ---
TIMESTAMPS_PER_DAY = 1_000_000 # Standard duration for a trading day in competition timestamps
CSV_FILENAME_FORMAT = "prices_round_{round_num}_day_{day_num}.csv" # Customizable format
# --- End Data Loading ---

# --- Fair Market Value Calculation Functions ---

# Default fair value calculation: Mid-price of the best bid and ask.
def mid_price(order_depth: OrderDepth) -> float:
    """Calculates the mid-price from the best bid and ask. Returns NaN if book is thin."""
    if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
        # Return NaN if the OrderDepth object is None or sides are empty
        return np.nan
    try:
        best_bid = min(order_depth.buy_orders.keys())
        best_ask = max(order_depth.sell_orders.keys())
        return float(best_bid + best_ask) / 2.0
    except Exception:
        # Handles potential errors like empty keys after filtering etc.
        return np.nan

# Define more complex fair value functions for specific products below if needed.
# Example structure:
# def calculate_fair_orchids(order_depth: OrderDepth) -> float:
#     """Calculates a custom fair value for ORCHIDS based on volume, etc."""
#     # Implement custom logic here...
#     # Example: Volume-weighted mid-price or regression-based value
#     if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
#         return np.nan
#     # ... complex calculation ...
#     calculated_value = mid_price(order_depth) # Placeholder
#     return calculated_value

# --- Fair Market Value Dictionary ---
# Maps product symbols (str) to the function used for fair value calculation (Callable).
# The callable should accept an OrderDepth object and return a float.
# PnL calculations will use the function assigned here.
FAIR_MKT_VALUE: Dict[str, Callable[[OrderDepth], float]] = {
    "SQUID_INK": mid_price,
     "KELP": mid_price,
     "RAINFOREST_RESIN": lambda _: 10000, # Special case: Fixed value (Verify if correct)
     # Add any other known *traded* products here
}

# Note: Non-traded items like TRANSPORT_FEES, EXPORT_TARIFF, IMPORT_TARIFF,
# SUNLIGHT, HUMIDITY are generally not included in FAIR_MKT_VALUE as they don't
# have a standard OrderDepth for PnL calculation based on market price.


# --- Position Limits Dictionary ---
# Maps product symbols (str) to their maximum allowed absolute position (int).
# !! IMPORTANT: These values are EXAMPLES based on previous rounds or guesses. !!
# !! YOU MUST UPDATE these limits based on the official competition rules !!
# !! for the specific round you are trading.                        !!
POSITION_LIMITS: Dict[str, int] = {'KELP': 50, 'RAINFOREST_RESIN': 50, 'SQUID_INK': 50, 'CROISSANTS': 250,
                             "JAMS": 350, "DJEMBES": 60, "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100}
