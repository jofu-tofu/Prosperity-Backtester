# backtester/backtester.py
import copy
import io
import json
import os
import pickle
import time
import numpy as np
from collections import defaultdict
from contextlib import redirect_stdout
from typing import Any, Dict, List, Literal, TextIO, Tuple

import pandas as pd

# Assuming these are in the same directory or package
from . import constants
from . import util
from .datamodel import Order, Listing, Observation, OrderDepth, Trade, TradingState
Symbol = str
# backtester/backtester.py

# --- Add at the top ---
import os
script_dir_backtester = os.path.dirname(os.path.abspath(__file__)) # Directory of backtester.py
BACKTESTER_BASE_DIR = os.path.dirname(script_dir_backtester) # Go up one level to the main project dir

TRADER_DIR_ABS = os.path.join(BACKTESTER_BASE_DIR, "backtester", "traders") # <<< Correct trader dir
DATA_DIR_ABS = os.path.join(BACKTESTER_BASE_DIR, "data") # <<< Correct data dir
# ---

class Backtester:
    # ... attributes ...

    def __init__(
        self,
        trader_fname: str,           # Expect just the filename
        data_fnames: List[str],      # Expect list of filenames
        timerange: tuple[int, int] = (0, float('inf')),
        bot_behavior: Literal["none", "eq", "lt", "lte"] = "lte",
        ignore_limits: bool = False,
    ):
        """Initializes the backtester."""
        # Construct full paths INTERNALLY using the corrected base paths
        self.trader_path = os.path.join(TRADER_DIR_ABS, trader_fname) # <<< Use correct base
        self.data_paths = [os.path.join(DATA_DIR_ABS, fname) for fname in data_fnames] # <<< Use correct base

        print(f"Initializing backtester with trader '{self.trader_path}' and data {self.data_paths}")
        print(f"Time range: {timerange}, Bot Behavior: {bot_behavior}, Ignore Limits: {ignore_limits}")

        self.trader = util.get_trader(self.trader_path) # Pass full path
        self.start_time, self.end_time = timerange
        self.bot_behavior = bot_behavior
        self.bot_matching_func = constants.BOT_BEHAVIOR_MATCH[bot_behavior]
        self.ignore_limits = ignore_limits

        self.listings = {}
        self.position_limit = constants.POSITION_LIMITS
        self.fair_price_calc = constants.FAIR_MKT_VALUE

        # Now _load_and_prepare_data uses self.data_paths which are correct absolute paths
        self._load_and_prepare_data()

        self.products = list(self.listings.keys())
        self.current_position = {product: 0 for product in self.products}

        self.pnl = {product: 0.0 for product in self.products}
        self.cash = {product: 0.0 for product in self.products}
        self.pnl_over_time = {product: [] for product in self.products}
        self.total_pnl_over_time = []
        self.sandbox_logs_capture = []
        self.lambda_logs_capture = []
        self.all_trades_log_output = []
        self.last_processed_timestamp = -1
        self.activity_log_output = self.market_data.copy()
        for prod in self.products:
              self.activity_log_output[f'pnl_{prod}'] = np.nan
        self.activity_log_output['total_pnl'] = np.nan
        print("Initialization complete.")
    # ... (_load_and_prepare_data, _precompute_order_depths remain the same) ...

    def run(self):
        """Runs the backtest simulation."""
        print("Starting simulation run...")
        traderData = ""
        perf_timer_start = time.time()

        timestamps = sorted(self.market_data.index.unique())

        for i, timestamp in enumerate(timestamps):
            # --- (Keep initial part: getting trades, preparing state, running trader) ---
            if timestamp < self.start_time or timestamp > self.end_time:
                continue

            current_step_market_trades_dict = defaultdict(list)
            current_step_own_trades_dict = defaultdict(list)

            start_range = self.last_processed_timestamp + 1
            end_range = timestamp
            trades_in_interval = self.raw_log_trades[
                (self.raw_log_trades.index >= start_range) & (self.raw_log_trades.index <= end_range)
            ]

            for _, trade_row in trades_in_interval.iterrows():
                  trade = Trade(
                      trade_row['symbol'], int(trade_row['price']), int(trade_row['quantity']),
                      trade_row['buyer'], trade_row['seller'], int(trade_row.name)
                  )
                  current_step_market_trades_dict[trade.symbol].append(trade)

            current_order_depths = copy.deepcopy(self.order_depths_cache.get(timestamp, {}))
            for prod in self.products:
                  if prod not in current_order_depths:
                      current_order_depths[prod] = OrderDepth()

            state = TradingState(
                traderData, timestamp, self.listings, current_order_depths,
                {}, dict(current_step_market_trades_dict), self.current_position.copy(),
                Observation({}, {}),
            )

            log_stream = io.StringIO()
            sandbox_log_output = ""
            trader_orders_dict = {}
            conversions = 0
            start_run_time = time.time()

            try:
                  with redirect_stdout(log_stream):
                     trader_orders_dict, conversions, traderData = self.trader.run(state)
            except Exception as e:
                 print(f"\n--- ERROR RUNNING TRADER AT TIMESTAMP {timestamp} ---")
                 print(f"Exception: {e}")
                 import traceback
                 traceback.print_exc()
                 print("--- END ERROR ---")
                 sandbox_log_output += f"\n TRADER CRASHED: {e}"
                 trader_orders_dict = {}

            run_time = time.time() - start_run_time
            if run_time * 1000 > 900:
                 sandbox_log_output += f"\n >> WARNING: Trader execution time exceeded 900ms ({run_time*1000:.2f}ms) <<"

            self.lambda_logs_capture.append(log_stream.getvalue())

            # --- Simulate Execution ---
            order_depths_for_matching = copy.deepcopy(current_order_depths)
            executed_own_trades_dict = defaultdict(list)
            bot_reaction_trades_at_t = self.raw_log_trades[self.raw_log_trades.index == timestamp]

            all_submitted_orders = []
            if trader_orders_dict:
                for product, orders in trader_orders_dict.items():
                   if product in self.listings:
                       for order in orders:
                           all_submitted_orders.append(order)

            grouped_player_orders = defaultdict(list)
            for order in all_submitted_orders:
                grouped_player_orders[order.symbol].append(order)

            # <<< START POSITION LIMIT CHECK AND EXECUTION >>>
            for product, product_orders in grouped_player_orders.items():
                 limit_breached = False # Flag for this product at this timestamp
                 if not self.ignore_limits and product in self.position_limit:
                     # Aggregate Check
                     total_buy_vol = sum(o.quantity for o in product_orders if o.quantity > 0)
                     total_sell_vol = sum(abs(o.quantity) for o in product_orders if o.quantity < 0)
                     current_pos = self.current_position[product]
                     limit = self.position_limit[product]

                     potential_long_pos = current_pos + total_buy_vol
                     potential_short_pos = current_pos - total_sell_vol

                     if potential_long_pos > limit or potential_short_pos < -limit:
                          sandbox_log_output += (
                               f"\nPOSITION LIMIT BREACH for {product} at {timestamp}. "
                               f"Limit: {limit}, Current: {current_pos}, Orders BuyVol: {total_buy_vol}, SellVol: {total_sell_vol}. "
                               f"Potential Range [{potential_short_pos}, {potential_long_pos}]. ALL ORDERS CANCELLED."
                           )
                          limit_breached = True
                          # According to rules, skip all orders for this product this tick
                          # We will achieve this by simply not proceeding with matching if breached
                 else:
                      # If limits ignored or product not in limits dict, proceed
                      limit_breached = False


                 if not limit_breached: # Proceed only if limit not breached (or ignored)
                    for order in product_orders:
                        order_copy = copy.deepcopy(order) # Work on a copy

                        # 1. Match against explicit order book
                        trades_explicit, sandbox_log_output = self._match_order_book(
                             timestamp, order_copy, order_depths_for_matching, sandbox_log_output
                         )
                        executed_own_trades_dict[product].extend(trades_explicit)

                        # 2. Match remaining volume against bot reactions (if applicable)
                        if abs(order_copy.quantity) > 0 and self.bot_behavior != "none":
                             trades_bot, sandbox_log_output = self._match_bot_reactions(
                                 timestamp, order_copy, bot_reaction_trades_at_t, sandbox_log_output
                             )
                             executed_own_trades_dict[product].extend(trades_bot)
                 # --- Else (limit breached): All orders for this product are effectively cancelled ---
            # <<< END POSITION LIMIT CHECK AND EXECUTION >>>


            self.sandbox_logs_capture.append({"sandboxLog": sandbox_log_output, "lambdaLog": log_stream.getvalue(), "timestamp": timestamp})

            for prod, trades in executed_own_trades_dict.items():
                for trade in trades:
                    self.all_trades_log_output.append(self._trade_to_dict(trade))

            for _, market_trade_row in bot_reaction_trades_at_t.iterrows():
                mtrade = Trade(
                    market_trade_row['symbol'], int(market_trade_row['price']), int(market_trade_row['quantity']),
                    market_trade_row['buyer'], market_trade_row['seller'], int(market_trade_row.name)
                )
                self.all_trades_log_output.append(self._trade_to_dict(mtrade))

            self._calculate_and_log_pnl(timestamp)
            self.last_processed_timestamp = timestamp

            if (i + 1) % 100 == 0:
                print(f"Processed timestamp {timestamp} ({(i + 1)} / {len(timestamps)})")

            perf_timer_end = time.time()
            print(f"Simulation finished in {perf_timer_end - perf_timer_start:.2f} seconds.")
            self._finalize_activity_log()
            self._generate_output()

    # --- Helper functions like _trade_to_dict, _calculate_and_log_pnl, _finalize_activity_log, _generate_output ---
    # --- need to remain, slightly adjusted _match_ functions below ---

    def _match_order_book(self, timestamp: int, order: Order, order_depths: Dict[Symbol, OrderDepth], log: str) -> Tuple[List[Trade], str]:
           """Matches an order against the explicit order book. NO LIMIT CHECKS HERE - done before."""
           trades_made = []
           product = order.symbol
           book = order_depths.get(product)
           if not book:
               log += f"\nWarning: No order book found for {product} at {timestamp} during explicit matching."
               return trades_made, log

           # NOTE: Per-fill limit check is removed. Assumed the aggregate check is sufficient.

           if order.quantity > 0:  # --- Buy Order ---
                sell_prices = sorted(book.sell_orders.keys())
                for price in sell_prices:
                    if order.quantity == 0: break # Filled
                    if price > order.price: break # Price too high

                    book_vol = abs(book.sell_orders[price])
                    vol_to_trade = min(order.quantity, book_vol)

                    # Execute the trade
                    trade = Trade(product, price, vol_to_trade, "SUBMISSION", "", timestamp)
                    trades_made.append(trade)
                    self.current_position[product] += vol_to_trade
                    self.cash[product] -= price * vol_to_trade
                    order.quantity -= vol_to_trade
                    book.sell_orders[price] += vol_to_trade # Reduce available volume
                    if book.sell_orders[price] == 0:
                        del book.sell_orders[price]

           else:  # --- Sell Order ---
               buy_prices = sorted(book.buy_orders.keys(), reverse=True)
               for price in buy_prices:
                    if order.quantity == 0: break # Filled
                    if price < order.price: break # Price too low

                    book_vol = abs(book.buy_orders[price])
                    vol_to_trade = min(abs(order.quantity), book_vol)

                    # Execute the trade
                    trade = Trade(product, price, vol_to_trade, "", "SUBMISSION", timestamp)
                    trades_made.append(trade)
                    self.current_position[product] -= vol_to_trade
                    self.cash[product] += price * vol_to_trade
                    order.quantity += vol_to_trade
                    book.buy_orders[price] -= vol_to_trade
                    if book.buy_orders[price] == 0:
                         del book.buy_orders[price]

           return trades_made, log

    def _match_bot_reactions(self, timestamp: int, order: Order, bot_trades_df: pd.DataFrame, log: str) -> Tuple[List[Trade], str]:
            """Matches remaining order quantity against potential bot reactions from log history. NO LIMIT CHECKS HERE."""
            trades_made = []
            product = order.symbol

            if bot_trades_df.empty or product not in bot_trades_df['symbol'].values:
                return trades_made, log

            product_bot_trades = bot_trades_df[bot_trades_df['symbol'] == product].copy()
            product_bot_trades.sort_index(inplace=True)

            # NOTE: Per-fill limit check is removed. Assumed the aggregate check is sufficient.

            if order.quantity > 0: # --- Player Buy Order vs Bot Sells ---
                 potential_matches = product_bot_trades[
                     (product_bot_trades['seller'].fillna("") != "") &
                     (product_bot_trades['buyer'].fillna("") == "")
                 ].sort_values('price')

                 for _, bot_trade in potential_matches.iterrows():
                    if order.quantity == 0: break
                    bot_price = int(bot_trade['price'])
                    bot_vol = abs(int(bot_trade['quantity']))

                    # Match based on bot_behavior (using price relative to player order)
                    should_match = self.bot_matching_func(order.price, bot_price)

                    if should_match:
                        vol_to_trade = min(order.quantity, bot_vol)
                        # Execute "trade"
                        trade = Trade(product, bot_price, vol_to_trade, "SUBMISSION", "BOT", timestamp)
                        trades_made.append(trade)
                        self.current_position[product] += vol_to_trade
                        self.cash[product] -= bot_price * vol_to_trade
                        order.quantity -= vol_to_trade

            else: # --- Player Sell Order vs Bot Buys ---
                potential_matches = product_bot_trades[
                    (product_bot_trades['buyer'].fillna("") != "") &
                    (product_bot_trades['seller'].fillna("") == "")
                ].sort_values('price', ascending=False)

                for _, bot_trade in potential_matches.iterrows():
                    if order.quantity == 0: break
                    bot_price = int(bot_trade['price'])
                    bot_vol = abs(int(bot_trade['quantity']))

                    # Match based on bot_behavior (using price relative to player order)
                    should_match = self.bot_matching_func(bot_price, order.price) # Order flipped for sell logic check

                    if should_match:
                        vol_to_trade = min(abs(order.quantity), bot_vol)
                        # Execute "trade"
                        trade = Trade(product, bot_price, vol_to_trade, "BOT", "SUBMISSION", timestamp)
                        trades_made.append(trade)
                        self.current_position[product] -= vol_to_trade
                        self.cash[product] += bot_price * vol_to_trade
                        order.quantity += vol_to_trade

            return trades_made, log
# Inside the Backtester class in backtester/backtester.py

    def _load_and_prepare_data(self):
        """Loads data from log files, merges trade history, precomputes."""
        all_market_data = []
        all_trade_history = []
        all_products = set()

        # Assume the first file is the primary source for market data structure/timestamps
        if not self.data_paths:
             raise ValueError("No data files provided to the backtester.")

        print(f"Using '{os.path.basename(self.data_paths[0])}' as primary market data source.")
        try:
            with open(self.data_paths[0], 'r', encoding='utf-8') as f: # Added encoding
                _, primary_market_df, primary_trades_df = util._parse_data(f)
        except FileNotFoundError:
             raise FileNotFoundError(f"Primary data file not found: {self.data_paths[0]}")
        except Exception as e:
             raise RuntimeError(f"Error parsing primary data file {self.data_paths[0]}: {e}")

        if primary_market_df.empty:
            raise ValueError(f"Primary data file '{os.path.basename(self.data_paths[0])}' contains no market data.")

        self.market_data = primary_market_df[
            (primary_market_df.index >= self.start_time) & (primary_market_df.index <= self.end_time)
        ]
        if self.market_data.empty:
            raise ValueError(f"No market data found within the specified time range {self.start_time}-{self.end_time} in primary file.")

        # Get initial product list from market data
        all_products.update(self.market_data['product'].unique())

        # Cache order depths ONLY from the primary market data source
        self._precompute_order_depths(self.market_data) # Cache depths from primary source

        # --- Load TRADE HISTORY from ALL files ---
        for data_path in self.data_paths:
            print(f"Loading trade history from '{os.path.basename(data_path)}'")
            try:
                with open(data_path, 'r', encoding='utf-8') as f: # Added encoding
                    _, _, trades_df = util._parse_data(f)
            except FileNotFoundError:
                print(f"Warning: Data file not found during trade history load: {data_path}. Skipping.")
                continue
            except Exception as e:
                 print(f"Warning: Error parsing trade history from {data_path}: {e}. Skipping.")
                 continue

            if not trades_df.empty:
                trades_df_filtered = trades_df[
                    (trades_df.index >= self.start_time) & (trades_df.index <= self.end_time)
                ]
                if not trades_df_filtered.empty:
                    # --- Get potential BOT reaction trades ---
                    # Ensure necessary columns exist before filtering
                    required_cols = ['symbol', 'price', 'quantity', 'buyer', 'seller']
                    if not all(col in trades_df_filtered.columns for col in required_cols):
                         print(f"Warning: Trade history in {data_path} missing required columns. Skipping bot trade extraction.")
                         continue

                    # Coerce types safely AFTER checking columns exist
                    trades_df_filtered['quantity'] = pd.to_numeric(trades_df_filtered['quantity'], errors='coerce')
                    trades_df_filtered['price'] = pd.to_numeric(trades_df_filtered['price'], errors='coerce')
                    trades_df_filtered.dropna(subset=['quantity', 'price'], inplace=True) # Drop if coercion failed
                    trades_df_filtered['quantity'] = trades_df_filtered['quantity'].astype(int)
                    trades_df_filtered['price'] = trades_df_filtered['price'].astype(int)
                    # Ensure buyer/seller are strings, handling potential float NaNs first
                    trades_df_filtered['buyer'] = trades_df_filtered['buyer'].fillna('').astype(str)
                    trades_df_filtered['seller'] = trades_df_filtered['seller'].fillna('').astype(str)


                    bot_trades = trades_df_filtered[
                        (trades_df_filtered['buyer'] != "SUBMISSION") &
                        (trades_df_filtered['seller'] != "SUBMISSION")
                    ].copy()

                    all_trade_history.append(bot_trades)
                    all_products.update(trades_df_filtered['symbol'].unique()) # Update products from trades too

        if not all_trade_history:
            print("Warning: No valid trade history found across all log files for bot reaction processing.")
            self.raw_log_trades = pd.DataFrame()
        else:
            self.raw_log_trades = pd.concat(all_trade_history).sort_index()
            # Remove duplicates - keep timestamp index! Check subset columns
            if not self.raw_log_trades.empty:
                key_cols = ['symbol', 'price', 'quantity', 'buyer', 'seller']
                if all(col in self.raw_log_trades.columns for col in key_cols):
                     self.raw_log_trades = self.raw_log_trades[~self.raw_log_trades.index.duplicated(keep='first') | ~self.raw_log_trades.duplicated(subset=key_cols, keep='first')]

                # Handle potential MultiIndex after concat if index wasn't unique, reset and set again
                if isinstance(self.raw_log_trades.index, pd.MultiIndex):
                    print("Warning: Duplicate timestamps detected in trade history, resetting index.")
                    self.raw_log_trades.reset_index(inplace=True)
                    self.raw_log_trades.set_index('timestamp', inplace=True)
                    # You might need more robust duplicate handling here depending on data
                    self.raw_log_trades = self.raw_log_trades[~self.raw_log_trades.index.duplicated(keep='first')]

                print(f"Total unique potential bot reaction trades loaded: {len(self.raw_log_trades)}")

        # --- Finalize listings and ensure constants have all products ---
        products_in_market = set(self.market_data['product'].unique()) if not self.market_data.empty else set()
        products_in_trades = set(self.raw_log_trades['symbol'].unique()) if not self.raw_log_trades.empty else set()
        all_products.update(products_in_market)
        all_products.update(products_in_trades)


        for prod in all_products:
             if prod not in self.listings:
                 self.listings[prod] = Listing(prod, prod, "SEASHELLS")
             # Ensure products exist in constants, adding defaults if missing
             if prod not in self.position_limit:
                  print(f"Warning: Product '{prod}' not found in POSITION_LIMITS, adding default limit of 20.")
                  self.position_limit[prod] = 20 # Add a default limit
             if prod not in self.fair_price_calc:
                  print(f"Warning: Product '{prod}' not found in FAIR_MKT_VALUE, adding default calc (mid_price).")
                  self.fair_price_calc[prod] = constants.mid_price

        print(f"Listings finalized: {list(self.listings.keys())}")
    # --- Keep the other helper functions (_trade_to_dict, _calculate_and_log_pnl, _finalize_activity_log, _generate_output) ---
    # Ensure _calculate_and_log_pnl and _finalize_activity_log use the updated attribute names correctly.
    def _precompute_order_depths(self, market_df: pd.DataFrame):
         """Precomputes and caches order depths for each timestamp."""
         print("Precomputing order depths...")
         self.order_depths_cache = {}
         grouped = market_df.groupby(market_df.index) # Group by timestamp index
         for timestamp, group in grouped:
             ts_order_depths = {}
             products_in_group = group['product'].unique()
             for product in products_in_group:
                 order_depth = OrderDepth()
                 product_row = group[group['product'] == product].iloc[0] # Get the row for this product
                 # Populate buy orders
                 for i in range(1, 4): # Assuming max 3 levels shown, adjust if needed
                     bid_price_col = f'bid_price_{i}'
                     bid_vol_col = f'bid_volume_{i}'
                     if bid_price_col in product_row and bid_vol_col in product_row and \
                        pd.notna(product_row[bid_price_col]) and pd.notna(product_row[bid_vol_col]):
                        order_depth.buy_orders[int(product_row[bid_price_col])] = int(product_row[bid_vol_col])
                 # Populate sell orders
                 for i in range(1, 4):
                     ask_price_col = f'ask_price_{i}'
                     ask_vol_col = f'ask_volume_{i}'
                     if ask_price_col in product_row and ask_vol_col in product_row and \
                        pd.notna(product_row[ask_price_col]) and pd.notna(product_row[ask_vol_col]):
                        order_depth.sell_orders[int(product_row[ask_price_col])] = -int(product_row[ask_vol_col]) # Note the negative sign
                 ts_order_depths[product] = order_depth
             self.order_depths_cache[timestamp] = ts_order_depths
         print("Order depths precomputed.")