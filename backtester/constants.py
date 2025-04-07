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


# --- Fair Market Value Calculation Functions ---

# Default fair value calculation: Mid-price of the best bid and ask.
def mid_price(order_depth: OrderDepth) -> float:
    """Calculates the mid-price from the best bid and ask. Returns NaN if book is thin."""
    if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
        # Return NaN if the OrderDepth object is None or sides are empty
        return np.nan
    try:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
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
    # Round 1
    "PEARLS": mid_price,
    "BANANAS": mid_price,
    # Round 2
    "COCONUTS": mid_price,
    "PINA_COLADAS": mid_price,
    # Round 3
    "DIVING_GEAR": mid_price,
    "BERRIES": mid_price,
    # Round 4
    "DIP": mid_price,
    "UKULELE": mid_price,
    "PICNIC_BASKET": mid_price,
    # Round 5
     "ROSES": mid_price,
     "GIFT_BASKET": mid_price,
     "CHOCOLATE": mid_price,
     "STRAWBERRIES": mid_price,
     # Round 6 (Based on log data)
     "ORCHIDS": mid_price, # Replace with calculate_fair_orchids if defined
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
POSITION_LIMITS: Dict[str, int] = {
    # # Round 1
    # "PEARLS": 20,
    # "BANANAS": 20,
    # # Round 2
    # "COCONUTS": 600,
    # "PINA_COLADAS": 300,
    # # Round 3
    # "DIVING_GEAR": 50,
    # "BERRIES": 250,
    # # Round 4
    # "DIP": 300, # Check R4 rules
    # "UKULELE": 70, # Check R4 rules
    # "PICNIC_BASKET": 70, # Check R4 rules
    # # Round 5
    #  "ROSES": 60,
    #  "GIFT_BASKET": 60,
    #  "CHOCOLATE": 250,
    #  "STRAWBERRIES": 350,
    #  # Round 6 (Based on log data - VERIFY THESE)
    #  "ORCHIDS": 100,
     "KELP": 50, # Example limit, VERIFY
     "RAINFOREST_RESIN": 50, # Example limit, VERIFY
     "SQUID INK": 50
     # Add any other known traded products here with their CORRECT limits
}
