# backtester/constants.py
import numpy as np
from .datamodel import OrderDepth # Assuming datamodel.py is in the same directory or accessible

# --- Fair Market Value Calculation Functions ---
# Define functions to calculate fair market value for each product if desired.
# They should take an OrderDepth object as input.
# Example: Use mid-price as default if no specific function is given
def mid_price(order_depth: OrderDepth) -> float:
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return np.nan # Or handle differently (e.g., last trade price)
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    return (best_bid + best_ask) / 2

# Add more complex fair value calculators here if needed for specific products
# def calculate_fair_orchids(order_depth: OrderDepth) -> float: ...

FAIR_MKT_VALUE = {
    "PEARLS": mid_price,  # Example default
    "BANANAS": mid_price, # Example default
    # Add other products as they appear
    "COCONUTS": mid_price,
    "PINA_COLADAS": mid_price,
    "DIVING_GEAR": mid_price,
    "BERRIES": mid_price,
    "DIP": mid_price,
    "UKULELE": mid_price,
    "PICNIC_BASKET": mid_price,
     "ROSES": mid_price,
     "GIFT_BASKET": mid_price,
     "CHOCOLATE": mid_price,
     "STRAWBERRIES": mid_price,
     "ORCHIDS": mid_price, # Placeholder - add complex logic if needed
     "TRANSPORT_FEES": mid_price, # These usually aren't traded, handle specially if needed
     "EXPORT_TARIFF": mid_price,
     "IMPORT_TARIFF": mid_price,
     "SUNLIGHT": mid_price,
     "HUMIDITY": mid_price,
     # Add all known products - KELP and RAINFOREST_RESIN from the log
     "KELP": mid_price,
     "RAINFOREST_RESIN": lambda _: 10000,
}

# --- Position Limits ---
POSITION_LIMITS = {
    "PEARLS": 20,
    "BANANAS": 20,
    "COCONUTS": 600,
    "PINA_COLADAS": 300,
    "DIVING_GEAR": 50,
    "BERRIES": 250,
    "DIP": 300,
    "UKULELE": 70,
    "PICNIC_BASKET": 70,
     "ROSES": 60,
     "GIFT_BASKET": 60,
     "CHOCOLATE": 250,
     "STRAWBERRIES": 350,
     "ORCHIDS": 100, # Assuming a limit based on R3 info
     # Add all known products
     "KELP": 500, # Example - adjust based on competition rules!
     "RAINFOREST_RESIN": 300 # Example - adjust!
}

# --- Bot Matching Behavior Logic ---
# Helper functions to determine if a bot trade matches a player order
BOT_BEHAVIOR_MATCH = {
    "none": lambda order_price, bot_price: False,  # Never match
    "eq": lambda order_price, bot_price: order_price == bot_price, # Only if equal
    "lt": lambda order_price, bot_price: order_price > bot_price,  # Player Buy wants < bot sell | Player Sell wants > bot buy
    "lte": lambda order_price, bot_price: order_price >= bot_price, # Player Buy wants <= bot sell | Player Sell wants >= bot buy
}