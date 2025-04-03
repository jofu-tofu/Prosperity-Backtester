# backtester/backtester.py
# (Imports remain the same)
import copy
import io
import json
import os
import pickle
import time
import numpy as np
from collections import defaultdict, Counter
from contextlib import redirect_stdout
import traceback
from typing import Any, Dict, List, Literal, TextIO, Tuple

import pandas as pd

from . import constants
from . import util
from .datamodel import Order, Listing, Observation, OrderDepth, Trade, TradingState
Symbol = str

# (Paths remain the same)
script_dir_backtester = os.path.dirname(os.path.abspath(__file__))
BACKTESTER_BASE_DIR = os.path.dirname(script_dir_backtester)
TRADER_DIR_ABS = os.path.join(BACKTESTER_BASE_DIR, "backtester", "traders")
DATA_DIR_ABS = os.path.join(BACKTESTER_BASE_DIR, "data")

class Backtester:

    def __init__(
        self,
        trader_fname: str,
        data_fnames: List[str],
        timerange: tuple[int, int] = (0, float('inf')),
        bot_behavior: Literal["none", "eq", "lt", "lte"] = "lte", # Re-add bot_behavior explicitly
        ignore_limits: bool = False,
        # **kwargs # Remove kwargs if bot_behavior is explicit
    ):
        """Initializes the backtester."""
        self.trader_path = os.path.join(TRADER_DIR_ABS, trader_fname)
        self.data_paths = [os.path.join(DATA_DIR_ABS, fname) for fname in data_fnames]

        print(f"Initializing backtester with trader '{self.trader_path}' and data {self.data_paths}")
        # --- Store bot_behavior ---
        self.bot_behavior = bot_behavior
        print(f"Time range: {timerange}, Bot Behavior: {self.bot_behavior}, Ignore Limits: {ignore_limits}")
        # --- Store bot_behavior ---

        self.trader = util.get_trader(self.trader_path)
        self.start_time, self.end_time = timerange
        self.ignore_limits = ignore_limits

        self.listings = {}
        self.products = []
        self.market_data = pd.DataFrame()
        self.explicit_order_depths_cache = {}
        self.inferred_bot_liquidity_cache = {}
        self.raw_all_trades = pd.DataFrame()

        self._load_and_prepare_data()

        if not hasattr(self, 'products') or not self.products: raise ValueError("Products could not be determined.")
        print(f"DEBUG __init__: Products determined: {self.products}")

        self.current_position = {p: 0 for p in self.products}
        self.pnl = {p: 0.0 for p in self.products}; self.cash = {p: 0.0 for p in self.products}
        self.pnl_over_time = {p: [] for p in self.products}; self.total_pnl_over_time = []
        self.sandbox_logs_capture = []; self.all_trades_log_output = []
        self.last_processed_timestamp = -1

        if not hasattr(self, 'market_data') or self.market_data.empty: raise ValueError("Market data not loaded.")
        self.activity_log_output = self.market_data.copy()
        cols_to_drop = [col for col in self.activity_log_output.columns if col.startswith('pnl_')]
        if cols_to_drop: self.activity_log_output.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        if 'profit_and_loss' not in self.activity_log_output.columns: self.activity_log_output['profit_and_loss'] = np.nan
        else: self.activity_log_output['profit_and_loss'] = np.nan

        self.position_limit = {p: constants.POSITION_LIMITS.get(p, 20) for p in self.products}
        self.fair_price_calc = {p: constants.FAIR_MKT_VALUE.get(p, constants.mid_price) for p in self.products}
        print("Initialization complete.")

    # --- _load_and_prepare_data remains the same (uses _calculate_excess_volume_per_log) ---
    # --- _calculate_excess_volume_per_log remains the same ---
    # --- _precompute_explicit_order_depths remains the same ---
    def _load_and_prepare_data(self):
        # ... (logic as in previous correct version) ...
        print("--- Loading and Preparing Data (Per-Log Inference v2) ---")
        all_products = set(); all_trades_list = []
        agg_max_bot_buys = defaultdict(lambda: defaultdict(Counter)); agg_max_bot_sells = defaultdict(lambda: defaultdict(Counter))
        if not self.data_paths: raise ValueError("No data files provided.")
        print(f"Using '{os.path.basename(self.data_paths[0])}' as primary market data source.")
        try:
            with open(self.data_paths[0], 'r', encoding='utf-8') as f: _, primary_market_df, _ = util._parse_data(f)
        except Exception as e: raise RuntimeError(f"Error parsing primary data file {self.data_paths[0]}: {e}")
        if primary_market_df.empty: raise ValueError("Primary market activity data is empty.")
        self.market_data = primary_market_df[(primary_market_df.index >= self.start_time) & (primary_market_df.index <= self.end_time)].copy()
        print(f"DEBUG load: Market data shape after time filter: {self.market_data.shape}")
        if not self.market_data.empty:
            if 'product' in self.market_data.columns: all_products.update(p for p in self.market_data['product'].unique() if isinstance(p, str) and p)
            self._precompute_explicit_order_depths(self.market_data)
        else: print("Warning: No market data in time range."); self.explicit_order_depths_cache = {}
        for data_path in self.data_paths:
            print(f"-- Processing log for excess: '{os.path.basename(data_path)}'")
            try:
                with open(data_path, 'r', encoding='utf-8') as f: _, _, trades_df = util._parse_data(f)
            except Exception as e: print(f"Warning: Error parsing {data_path}: {e}. Skipping."); continue
            if not trades_df.empty:
                trades_df_filtered = trades_df[(trades_df.index >= self.start_time) & (trades_df.index <= self.end_time)].copy()
                if not trades_df_filtered.empty:
                    required_cols = ['symbol','price','quantity','buyer','seller']; # etc
                    if not all(col in trades_df_filtered.columns for col in required_cols): continue
                    for col in ['quantity','price']: trades_df_filtered[col] = pd.to_numeric(trades_df_filtered[col], errors='coerce')
                    trades_df_filtered.dropna(subset=['quantity','price'], inplace=True)
                    trades_df_filtered['quantity']=trades_df_filtered['quantity'].astype(int); trades_df_filtered['price']=trades_df_filtered['price'].astype(int)
                    trades_df_filtered['buyer']=trades_df_filtered['buyer'].fillna('').astype(str); trades_df_filtered['seller']=trades_df_filtered['seller'].fillna('').astype(str)
                    all_trades_list.append(trades_df_filtered)
                    submission_trades_this_log = trades_df_filtered[(trades_df_filtered['buyer'] == "SUBMISSION") | (trades_df_filtered['seller'] == "SUBMISSION")].copy()
                    if not submission_trades_this_log.empty:
                        # print(f"   Found {len(submission_trades_this_log)} submission trades.") # Reduce noise
                        log_excess_buys, log_excess_sells = self._calculate_excess_volume_per_log(submission_trades_this_log, self.explicit_order_depths_cache)
                        for ts, products in log_excess_buys.items():
                            for prod, prices in products.items():
                                for price, vol in prices.items(): agg_max_bot_buys[ts][prod][price] = max(agg_max_bot_buys[ts][prod][price], vol)
                        for ts, products in log_excess_sells.items():
                            for prod, prices in products.items():
                                for price, vol in prices.items(): agg_max_bot_sells[ts][prod][price] = max(agg_max_bot_sells[ts][prod][price], vol)
                    if 'symbol' in trades_df_filtered.columns: all_products.update(p for p in trades_df_filtered['symbol'].unique() if isinstance(p, str) and p)
        if not all_trades_list: print("Warning: No trade history found."); self.raw_all_trades = pd.DataFrame()
        else:
             self.raw_all_trades = pd.concat(all_trades_list);
             if not self.raw_all_trades.index.is_monotonic_increasing: self.raw_all_trades.sort_index(inplace=True)
             key_cols = ['symbol','price','quantity','buyer','seller']; initial_count = len(self.raw_all_trades)
             self.raw_all_trades = self.raw_all_trades[~self.raw_all_trades.index.duplicated(keep='first') | ~self.raw_all_trades.duplicated(subset=key_cols, keep='first')]
             print(f"DEBUG load: Consolidated self.raw_all_trades shape: {self.raw_all_trades.shape} (removed {initial_count - len(self.raw_all_trades)} duplicates)")
        self.products = sorted(list(all_products));
        if not self.products: raise ValueError("No products determined.")
        print(f"DEBUG load: Final product list determined: {self.products}")
        for prod in self.products: self.listings[prod] = Listing(prod, prod, "SEASHELLS")
        print("Building final inferred bot liquidity cache..."); self.inferred_bot_liquidity_cache = defaultdict(lambda: defaultdict(OrderDepth))
        timestamps_with_inferred = set(agg_max_bot_buys.keys()).union(agg_max_bot_sells.keys())
        for ts in timestamps_with_inferred:
            for prod in self.products:
                 has_buys=prod in agg_max_bot_buys[ts] and agg_max_bot_buys[ts][prod]; has_sells=prod in agg_max_bot_sells[ts] and agg_max_bot_sells[ts][prod]
                 if has_buys or has_sells:
                      bot_depth = OrderDepth();
                      if has_buys: bot_depth.buy_orders = dict(agg_max_bot_buys[ts][prod])
                      if has_sells: bot_depth.sell_orders = {p: -v for p, v in agg_max_bot_sells[ts][prod].items()}
                      self.inferred_bot_liquidity_cache[ts][prod] = bot_depth
        print(f"Final inferred bot liquidity cache created for {len(self.inferred_bot_liquidity_cache)} timestamps.")

    def _calculate_excess_volume_per_log(self, submission_trades_df, explicit_book_cache):
        # ... (logic remains the same) ...
        log_excess_bot_buys = defaultdict(lambda: defaultdict(Counter)); log_excess_bot_sells = defaultdict(lambda: defaultdict(Counter))
        grouped_trades = submission_trades_df.groupby([submission_trades_df.index, 'symbol'])
        for (timestamp, product), trades in grouped_trades:
            initial_explicit_book = explicit_book_cache.get(timestamp, {}).get(product);
            if not initial_explicit_book: continue
            sub_buys_at_price = Counter(); sub_sells_at_price = Counter()
            for _, row in trades.iterrows():
                price = int(row['price']); quantity = int(row['quantity'])
                if row['buyer'] == "SUBMISSION": sub_buys_at_price[price] += quantity
                elif row['seller'] == "SUBMISSION": sub_sells_at_price[price] += quantity
            if sub_buys_at_price: # Simulate Buys
                explicit_sells_copy = copy.deepcopy(initial_explicit_book.sell_orders); fills = Counter()
                for sub_p in sorted(sub_buys_at_price.keys()):
                    vol_match = sub_buys_at_price[sub_p]
                    for book_p in sorted(explicit_sells_copy.keys()):
                        if vol_match<=0 or book_p>sub_p: break
                        book_v=abs(explicit_sells_copy[book_p]); filled=min(vol_match,book_v)
                        if filled>0: 
                            fills[sub_p]+=filled 
                            explicit_sells_copy[book_p]+=filled
                            vol_match-=filled
                            if explicit_sells_copy[book_p]==0: del explicit_sells_copy[book_p]
                    excess = sub_buys_at_price[sub_p] - fills[sub_p]
                    if excess > 0: log_excess_bot_sells[timestamp][product][sub_p] = excess
            if sub_sells_at_price: # Simulate Sells
                explicit_buys_copy = copy.deepcopy(initial_explicit_book.buy_orders); fills = Counter()
                for sub_p in sorted(sub_sells_at_price.keys(), reverse=True):
                    vol_match = sub_sells_at_price[sub_p]
                    for book_p in sorted(explicit_buys_copy.keys(), reverse=True):
                        if vol_match<=0 or book_p<sub_p: break
                        book_v=abs(explicit_buys_copy[book_p]); filled=min(vol_match,book_v)
                        if filled>0: 
                            fills[sub_p]+=filled
                            explicit_buys_copy[book_p]-=filled
                            vol_match-=filled
                            if explicit_buys_copy[book_p]==0: 
                                del explicit_buys_copy[book_p]
                    excess = sub_sells_at_price[sub_p] - fills[sub_p]
                    if excess > 0: log_excess_bot_buys[timestamp][product][sub_p] = excess
        return log_excess_bot_buys, log_excess_bot_sells

    def _precompute_explicit_order_depths(self, market_df: pd.DataFrame):
        # ... (logic remains the same) ...
        print("Precomputing explicit order depths...")
        self.explicit_order_depths_cache = {}
        if market_df.empty or not pd.api.types.is_numeric_dtype(market_df.index): return
        grouped = market_df.groupby(market_df.index)
        for timestamp, group in grouped:
            ts_depths = {}
            products_in_group = group['product'].unique() if 'product' in group.columns else []
            for product in products_in_group:
                order_depth = OrderDepth(); product_rows = group.loc[group['product'] == product]
                if product_rows.empty: 
                    continue
                product_row = product_rows.iloc[0]
                for i in range(1, 4): 
                    bp, bv = f'bid_price_{i}', f'bid_volume_{i}'
                    if bp in product_row and bv in product_row and pd.notna(product_row[bp]) and pd.notna(product_row[bv]) and product_row[bv] > 0: 
                        try: order_depth.buy_orders[int(product_row[bp])] = int(product_row[bv])
                        except ValueError: pass
                for i in range(1, 4): 
                    ap, av = f'ask_price_{i}', f'ask_volume_{i}'
                    if ap in product_row and av in product_row and pd.notna(product_row[ap]) and pd.notna(product_row[av]) and product_row[av] > 0: 
                        try: order_depth.sell_orders[int(product_row[ap])] = -int(product_row[av])
                        except ValueError: pass
                ts_depths[product] = order_depth
            self.explicit_order_depths_cache[timestamp] = ts_depths
        print(f"Explicit order depths precomputed for {len(self.explicit_order_depths_cache)} timestamps.")

    # --- run method: Pass self.bot_behavior to second match call ---
    def run(self):
        # ... (Setup: traderData, last_processed_timestamp, timers, timestamps etc.) ...
        print("Starting simulation run...")
        traderData = ""; self.last_processed_timestamp = -1; perf_timer_start = time.time()
        if self.activity_log_output.empty: raise ValueError("activity_log_output empty before run.")
        timestamps = sorted(self.activity_log_output.index.unique());
        if not timestamps: raise ValueError("No timestamps to simulate.")
        print(f"Simulating {len(timestamps)} unique timestamps...")

        for i, timestamp in enumerate(timestamps):
            if timestamp < self.start_time or timestamp > self.end_time: continue

            # --- Get market_trades for state ---
            current_step_market_trades_dict = defaultdict(list)
            start_range, end_range = self.last_processed_timestamp + 1, timestamp
            try:
                if hasattr(self, 'raw_all_trades') and not self.raw_all_trades.empty: # Changed from raw_all_trades_for_state
                    if not self.raw_all_trades.index.is_monotonic_increasing: self.raw_all_trades.sort_index(inplace=True)
                    try:
                        trades_in_interval = self.raw_all_trades.loc[start_range:end_range]
                        for idx, row in trades_in_interval.iterrows():
                             symbol=row.get('symbol'); price=row.get('price'); quantity=row.get('quantity'); buyer=row.get('buyer'); seller=row.get('seller')
                             if all(v is not None for v in [symbol, price, quantity]):
                                 trade = Trade(symbol, int(price), int(quantity), buyer, seller, int(idx)); current_step_market_trades_dict[symbol].append(trade)
                    except KeyError: pass # Range not in index is fine
            except Exception as e: print(f"Error slicing self.raw_all_trades @{timestamp}: {e}")

            # --- Prepare TradingState ---
            current_explicit_depths = copy.deepcopy(self.explicit_order_depths_cache.get(timestamp, {}))
            for prod in self.products:
                if prod not in current_explicit_depths: current_explicit_depths[prod] = OrderDepth()
            state = TradingState(traderData, timestamp, self.listings, current_explicit_depths, {}, dict(current_step_market_trades_dict), self.current_position.copy(), Observation({}, {}),)

            # --- Run Trader ---
            log_stream=io.StringIO(); sandbox_log_output=""; trader_orders_dict={}; conversions=0
            start_run_time = time.time()
            try:
                with redirect_stdout(log_stream): trader_orders_dict, conversions, traderData = self.trader.run(state)
            except Exception as e: print(f"\nTRADER CRASH @{timestamp}: {e}"); traceback.print_exc(); sandbox_log_output += f"\n!!! TRADER CRASH !!!"; trader_orders_dict = {}
            run_time = time.time() - start_run_time
            if run_time * 1000 > 900: sandbox_log_output += f"\n>> WARNING: Trader exec > 900ms ({run_time*1000:.2f}ms) <<"
            captured_lambda_log = log_stream.getvalue(); log_stream.close()

            # --- Simulate Execution ---
            executed_own_trades_this_step = defaultdict(list)
            explicit_depths_for_matching = copy.deepcopy(current_explicit_depths)
            bot_liquidity_books_at_t = self.inferred_bot_liquidity_cache.get(timestamp, {})

            if trader_orders_dict:
                all_submitted_orders=[]; grouped_player_orders = defaultdict(list)
                # ... (Order validation/grouping) ...
                for product, orders in trader_orders_dict.items():
                   if product in self.listings and isinstance(orders, list):
                        for order in orders:
                            if isinstance(order, Order): all_submitted_orders.append(order)
                for order in all_submitted_orders:
                     if isinstance(order.price, int) and isinstance(order.quantity, int): grouped_player_orders[order.symbol].append(order)

                for product, product_orders in grouped_player_orders.items(): # Limit Check & Exec
                    limit_breached = False # ... (Limit check logic) ...
                    if not self.ignore_limits and product in self.position_limit:
                        total_buy=sum(o.quantity for o in product_orders if o.quantity>0); total_sell=sum(abs(o.quantity) for o in product_orders if o.quantity<0)
                        curr=self.current_position.get(product,0); limit=self.position_limit.get(product,0)
                        if (curr + total_buy > limit) or (curr - total_sell < -limit): limit_breached=True; sandbox_log_output += f"\nLIMIT BREACH {product}@{timestamp}."

                    if not limit_breached:
                        for order in product_orders:
                            order_copy = copy.deepcopy(order)
                            # 1. Match Explicit Book (bot_behavior doesn't apply)
                            trades_explicit, sandbox_log_output = self._match_against_book(
                                timestamp, order_copy, explicit_depths_for_matching, sandbox_log_output, "ExplicitBook", "any" # Use 'any' behavior for explicit book
                            )
                            executed_own_trades_this_step[product].extend(trades_explicit)

                            # 2. Match Inferred Bot Liquidity (apply self.bot_behavior)
                            if abs(order_copy.quantity) > 0 and product in bot_liquidity_books_at_t:
                                 current_product_inferred_book = {product: bot_liquidity_books_at_t[product]}
                                 # --- Pass self.bot_behavior ---
                                 trades_bot, sandbox_log_output = self._match_against_book(
                                     timestamp, order_copy, current_product_inferred_book, sandbox_log_output, "InferredBot", self.bot_behavior
                                 )
                                 # --- Pass self.bot_behavior ---
                                 executed_own_trades_this_step[product].extend(trades_bot)

            # --- Log results ---
            self.sandbox_logs_capture.append({"sandboxLog": sandbox_log_output, "lambdaLog": captured_lambda_log, "timestamp": timestamp})
            for trades in executed_own_trades_this_step.values(): # Log executed player trades
                for trade in trades: self.all_trades_log_output.append(self._trade_to_dict(trade))
            # Optionally log non-player trades (using self.raw_all_trades)
            if hasattr(self, 'raw_all_trades') and not self.raw_all_trades.empty:
                 non_player_trades_at_t = self.raw_all_trades[ (self.raw_all_trades.index == timestamp) & (self.raw_all_trades['buyer'] != "SUBMISSION") & (self.raw_all_trades['seller'] != "SUBMISSION") ]
                 for _, row in non_player_trades_at_t.iterrows():
                     mtrade = Trade(row['symbol'], int(row['price']), int(row['quantity']), row['buyer'], row['seller'], timestamp); self.all_trades_log_output.append(self._trade_to_dict(mtrade))

            self._calculate_and_log_pnl(timestamp)
            self.last_processed_timestamp = timestamp
        # --- End loop ---
        perf_timer_end = time.time(); print(f"\nSimulation loop finished in {perf_timer_end - perf_timer_start:.2f} seconds.")
        if hasattr(self, 'activity_log_output') and not self.activity_log_output.empty: print(f"Activity Log Tail:\n{self.activity_log_output.tail(10)}")
        self._finalize_activity_log()
        self._generate_output()


    # --- _match_against_book: Add bot_behavior parameter and logic ---
    def _match_against_book(self, timestamp: int, order: Order, order_books: Dict[Symbol, OrderDepth], log: str, book_type: str, bot_behavior: Literal["none", "eq", "lt", "lte", "any"]) -> Tuple[List[Trade], str]:
        """
        Generic matching function. Applies bot_behavior rules ONLY if book_type is 'InferredBot'.
        'any' behavior means match any price satisfying order limit.
        """
        trades_made = []; product = order.symbol
        book = order_books.get(product);
        if not book: return trades_made, log
        # Modify the original dict directly - ensures consumed liquidity is tracked across calls within same timestamp
        # book_copy = copy.deepcopy(book) # Don't copy here if we want state maintained

        if order.quantity > 0:  # Buy Order matches against Sell side
            sell_prices = sorted([int(p) for p in book.sell_orders.keys()])
            for price in sell_prices:
                if order.quantity <= 0: break # Order filled

                # 1. Basic order price limit check
                if price > order.price: continue # Book price too high for buy order limit

                # 2. Apply bot_behavior rules ONLY for InferredBot book
                if book_type == "InferredBot" and bot_behavior != "any":
                    allowed_by_behavior = False
                    if bot_behavior == "eq": allowed_by_behavior = (price == order.price)
                    elif bot_behavior == "lt": allowed_by_behavior = (price < order.price) # Bot sells CHEAPER than limit
                    elif bot_behavior == "lte": allowed_by_behavior = (price <= order.price) # Bot sells CHEAPER OR EQUAL to limit
                    elif bot_behavior == "none": allowed_by_behavior = False # Explicitly disallow matching inferred book

                    if not allowed_by_behavior:
                         # print(f"DEBUG SKIP Bot Match Buy: OrderP={order.price}, BookP={price}, Behavior='{bot_behavior}'") # Optional Debug
                         continue # Skip this price level based on behavior rule

                # --- Price checks passed, proceed with volume matching ---
                if price not in book.sell_orders: continue # Check if price still exists after potential modification
                book_vol = abs(book.sell_orders[price]);
                vol_to_trade = min(order.quantity, book_vol)
                if vol_to_trade <= 0: continue

                # --- Execute Trade ---
                trade = Trade(product, price, vol_to_trade, "SUBMISSION", book_type, timestamp); trades_made.append(trade)
                self.current_position[product] = self.current_position.get(product, 0) + vol_to_trade
                self.cash[product] = self.cash.get(product, 0.0) - (price * vol_to_trade)
                order.quantity -= vol_to_trade
                book.sell_orders[price] += vol_to_trade # Consume volume from the book
                if book.sell_orders[price] == 0: del book.sell_orders[price]

        elif order.quantity < 0:  # Sell Order matches against Buy side
            buy_prices = sorted([int(p) for p in book.buy_orders.keys()], reverse=True)
            for price in buy_prices:
                if order.quantity >= 0: break # Order filled

                # 1. Basic order price limit check
                if price < order.price: continue # Book price too low for sell order limit

                # 2. Apply bot_behavior rules ONLY for InferredBot book
                if book_type == "InferredBot" and bot_behavior != "any":
                     allowed_by_behavior = False
                     # Note: Price comparison is flipped for selling
                     if bot_behavior == "eq": allowed_by_behavior = (price == order.price)
                     elif bot_behavior == "lt": allowed_by_behavior = (price > order.price) # Bot buys HIGHER than limit
                     elif bot_behavior == "lte": allowed_by_behavior = (price >= order.price) # Bot buys HIGHER OR EQUAL to limit
                     elif bot_behavior == "none": allowed_by_behavior = False

                     if not allowed_by_behavior:
                          # print(f"DEBUG SKIP Bot Match Sell: OrderP={order.price}, BookP={price}, Behavior='{bot_behavior}'") # Optional Debug
                          continue

                # --- Price checks passed, proceed with volume matching ---
                if price not in book.buy_orders: continue # Check if price still exists
                book_vol = abs(book.buy_orders[price]);
                vol_to_trade = min(abs(order.quantity), book_vol)
                if vol_to_trade <= 0: continue

                # --- Execute Trade ---
                trade = Trade(product, price, vol_to_trade, book_type, "SUBMISSION", timestamp); trades_made.append(trade)
                self.current_position[product] = self.current_position.get(product, 0) - vol_to_trade
                self.cash[product] = self.cash.get(product, 0.0) + (price * vol_to_trade)
                order.quantity += vol_to_trade
                book.buy_orders[price] -= vol_to_trade # Consume volume from the book
                if book.buy_orders[price] == 0: del book.buy_orders[price]

        return trades_made, log

    # --- _calculate_and_log_pnl, _finalize_activity_log, _generate_output, _trade_to_dict ---
    # --- Paste the LATEST working versions here (logging product-specific PNL) ---
    def _calculate_and_log_pnl(self, timestamp: int):
        # ... (logic as provided previously - logs product-specific PNL to profit_and_loss) ...
         total_pnl_at_t = 0.0
         if hasattr(self.activity_log_output, 'index') and not self.activity_log_output.index.is_monotonic_increasing: self.activity_log_output.sort_index(inplace=True)
         for product in self.products:
              calculated_pnl = self.pnl.get(product, 0.0)
              if product in self.fair_price_calc and product in self.current_position:
                  order_depth = self.explicit_order_depths_cache.get(timestamp, {}).get(product, OrderDepth())
                  fair_value = self.fair_price_calc[product](order_depth)
                  if pd.isna(fair_value): # ffill logic
                      try:
                           product_activity=self.activity_log_output.loc[(self.activity_log_output['product']==product) & (self.activity_log_output.index<=timestamp)]
                           if not product_activity.empty and 'mid_price' in product_activity.columns: 
                               ffilled=product_activity['mid_price'].ffill()
                               if not ffilled.empty and pd.notna(ffilled.iloc[-1]): fair_value=ffilled.iloc[-1]
                      except Exception: pass
                  current_cash = self.cash.get(product, 0.0); current_pos = self.current_position.get(product, 0)
                  if not pd.isna(fair_value): calculated_pnl = current_cash + (current_pos * fair_value)
                  self.pnl[product] = calculated_pnl; self.pnl_over_time[product].append((timestamp, calculated_pnl))
              row_mask_product = (self.activity_log_output.index == timestamp) & (self.activity_log_output['product'] == product)
              if row_mask_product.any():
                  if 'profit_and_loss' in self.activity_log_output.columns: self.activity_log_output.loc[row_mask_product, 'profit_and_loss'] = calculated_pnl
              total_pnl_at_t += calculated_pnl
         self.total_pnl_over_time.append((timestamp, total_pnl_at_t))

    def _finalize_activity_log(self):
        # ... (logic as provided previously - handles product-specific PNL in profit_and_loss) ...
        print("\n--- Starting _finalize_activity_log (Product-Specific PNL) ---")
        if not hasattr(self, 'activity_log_output') or self.activity_log_output.empty: print("ERROR finalize: activity_log_output missing/empty."); self.final_activity_log = pd.DataFrame(); return
        self.final_activity_log = self.activity_log_output.copy(); pnl_col = 'profit_and_loss'
        unexpected_pnl_cols = [col for col in self.final_activity_log.columns if col.startswith('pnl_')];
        if unexpected_pnl_cols: self.final_activity_log.drop(columns=unexpected_pnl_cols, inplace=True, errors='ignore')
        if pnl_col in self.final_activity_log.columns:
            if not pd.api.types.is_numeric_dtype(self.final_activity_log[pnl_col]): self.final_activity_log[pnl_col] = pd.to_numeric(self.final_activity_log[pnl_col], errors='coerce')
            self.final_activity_log[pnl_col] = self.final_activity_log.groupby('product')[pnl_col].ffill(); self.final_activity_log[pnl_col] = self.final_activity_log[pnl_col].fillna(0)
        else: self.final_activity_log[pnl_col] = 0.0
        if 'day' not in self.final_activity_log.columns: self.final_activity_log.insert(0, 'day', -1)
        standard_cols_order = ['day','product','bid_price_1','bid_volume_1','bid_price_2','bid_volume_2','bid_price_3','bid_volume_3','ask_price_1','ask_volume_1','ask_price_2','ask_volume_2','ask_price_3','ask_volume_3','mid_price','profit_and_loss']
        final_standard_cols = [col for col in standard_cols_order if col in self.final_activity_log.columns]; processed_cols = set(final_standard_cols + ['day'])
        other_cols = sorted([col for col in self.final_activity_log.columns if col not in processed_cols and not col.startswith('pnl_')]); final_col_order = []
        if 'day' in self.final_activity_log.columns: final_col_order.append('day'); final_col_order.extend([col for col in final_standard_cols if col != 'day']); final_col_order.extend(other_cols)
        self.final_activity_log = self.final_activity_log[[col for col in final_col_order if col in self.final_activity_log.columns]]
        print(f"Columns AFTER finalization & reorder: {self.final_activity_log.columns.tolist()}")
        print("--- Finished _finalize_activity_log ---"); self.final_activity_log.index.name = 'timestamp'

    def _generate_output(self):
        # ... (logic remains the same) ...
        output_stream = io.StringIO(); print("\n--- Generating final output string ---"); output_stream.write("Sandbox logs:\n");
        for log_entry in self.sandbox_logs_capture:
            try: json.dump(log_entry, output_stream); output_stream.write("\n")
            except Exception as e: print(f"Error dumping sandbox log: {e}"); output_stream.write('{"error": "log dump failed"}\n')
        output_stream.write("\n\nActivities log:\n");
        if hasattr(self, 'final_activity_log') and not self.final_activity_log.empty:
            try: self.final_activity_log.to_csv(output_stream, sep=';', index=True, float_format='%.6f')
            except Exception as e: print(f"ERROR generate: Failed write activity log: {e}"); output_stream.write("ERROR writing activity log\n")
        else: print("Warning generate: Final activity log empty."); header = "day;timestamp;product;bid_price_1;bid_volume_1;ask_price_1;ask_volume_1;mid_price;profit_and_loss\n"; output_stream.write(header)
        output_stream.write("\n\n\nTrade History:\n");
        try: json.dump(sorted(self.all_trades_log_output, key=lambda x: x.get('timestamp', 0)), output_stream, indent=2)
        except Exception as e: print(f"Error dumping trade history: {e}"); output_stream.write('[{"error": "trade dump failed"}]\n')
        self.output = output_stream.getvalue(); print("--- Final output string generated. ---"); output_stream.close()

    def _trade_to_dict(self, trade: Trade) -> Dict:
        # ... (logic remains the same) ...
        return {"timestamp": trade.timestamp, "buyer": trade.buyer or "", "seller": trade.seller or "", "symbol": trade.symbol, "currency": "SEASHELLS", "price": trade.price, "quantity": trade.quantity}