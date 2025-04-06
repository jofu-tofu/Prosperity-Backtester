# --- START OF FILE backtester.py ---
# backtester/backtester.py
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
from typing import Any, Dict, List, Literal, TextIO, Tuple, Optional

import pandas as pd

# Assuming datamodel.py is in the same directory or accessible
try:
    from .datamodel import Order, Listing, Observation, OrderDepth, Trade, TradingState
    from . import constants
    from . import util
except ImportError:
    # Fallback for running script directly
    from datamodel import Order, Listing, Observation, OrderDepth, Trade, TradingState
    import constants
    import util

Symbol = str

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
        bot_behavior: Literal["none", "eq", "lt", "lte"] = "lte",
        ignore_limits: bool = False,
    ):
        self.trader_path = os.path.join(TRADER_DIR_ABS, trader_fname)
        self.data_paths = [os.path.join(DATA_DIR_ABS, fname) for fname in data_fnames]
        print(f"Initializing backtester with trader '{self.trader_path}' and data {self.data_paths}")
        print(f"Time range: {timerange}, Bot Behavior (Inference/Matching): {bot_behavior}, Ignore Limits: {ignore_limits}")

        self.trader = util.get_trader(self.trader_path)
        self.start_time, self.end_time = timerange
        self.ignore_limits = ignore_limits
        self.bot_behavior = bot_behavior

        self.listings: Dict[Symbol, Listing] = {}
        self.products: List[Symbol] = []
        self.market_data = pd.DataFrame() # Stores the raw structure from the primary log
        self.explicit_order_depths_cache: Dict[int, Dict[Symbol, OrderDepth]] = {}
        self.inferred_bot_liquidity_cache: Dict[int, Dict[Symbol, OrderDepth]] = defaultdict(lambda: defaultdict(OrderDepth))
        self.raw_all_trades = pd.DataFrame() # Stores all trades from all logs

        # Load raw data structures first
        self._load_and_prepare_data()

        if not hasattr(self, 'products') or not self.products:
            raise ValueError("Products not determined after data loading.")
        print(f"DEBUG __init__: Products determined: {self.products}")

        # Initialize state variables AFTER products are known
        self.current_position: Dict[Symbol, int] = {p: 0 for p in self.products}
        self.pnl: Dict[Symbol, float] = {p: 0.0 for p in self.products}
        self.cash: Dict[Symbol, float] = {p: 0.0 for p in self.products}
        self.pnl_over_time: Dict[Symbol, List[Tuple[int, float]]] = {p: [] for p in self.products}
        self.total_pnl_over_time: List[Tuple[int, float]] = []
        self.sandbox_logs_capture: List[Dict[str, Any]] = []
        self.all_trades_log_output: List[Dict[str, Any]] = []
        self.last_processed_timestamp: int = -1

        # Activity log structure for output - initialized in run()
        self.activity_log_output = pd.DataFrame()
        self.final_activity_log = pd.DataFrame() # For finalized output

        self.position_limit: Dict[Symbol, int] = {p: constants.POSITION_LIMITS.get(p, 20) for p in self.products}
        self.fair_price_calc: Dict[Symbol, Any] = {p: constants.FAIR_MKT_VALUE.get(p, constants.mid_price) for p in self.products}

        if not hasattr(self, 'market_data') or self.market_data.empty:
            print("Warning: Market data structure is empty after loading.")
            # Consider raising an error if market data is essential for structure

        print("Initialization complete.")

    def _load_and_prepare_data(self):
        print("--- Loading and Preparing Data (Per-Log Inference v2) ---")
        all_products = set()
        all_trades_list = []
        agg_max_bot_buys = defaultdict(lambda: defaultdict(Counter))
        agg_max_bot_sells = defaultdict(lambda: defaultdict(Counter))

        if not self.data_paths:
            raise ValueError("No data files provided.")

        print(f"Using '{os.path.basename(self.data_paths[0])}' as primary market data source.")
        try:
            with open(self.data_paths[0], 'r', encoding='utf-8') as f:
                _, primary_market_df, _ = util._parse_data(f)
        except Exception as e:
            raise RuntimeError(f"Error parsing primary data file {self.data_paths[0]}: {e}")

        # Store the raw structure but filter for the run time range immediately
        self.market_data = primary_market_df[
            (primary_market_df.index >= self.start_time) & (primary_market_df.index <= self.end_time)
        ].copy()
        print(f"DEBUG load: Primary market data shape after time filter: {self.market_data.shape}")

        if not self.market_data.empty:
            if 'product' in self.market_data.columns:
                # Use only products present within the time range of the primary log
                valid_products = self.market_data['product'].unique()
                all_products.update(p for p in valid_products if isinstance(p, str) and p)
            self._precompute_explicit_order_depths(self.market_data) # Use time-filtered data
        else:
            print("Warning: No market data in the specified time range from the primary file.")
            self.explicit_order_depths_cache = {}

        # Process all logs for trades and bot inference
        for data_path in self.data_paths:
            print(f"-- Processing log for excess liquidity: '{os.path.basename(data_path)}'")
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    _, _, trades_df = util._parse_data(f)
            except Exception as e:
                print(f"Warning: Error parsing {data_path}: {e}. Skipping for excess calculation.")
                continue

            if not trades_df.empty:
                # Filter trades by the global time range
                trades_df_filtered = trades_df[
                    (trades_df.index >= self.start_time) & (trades_df.index <= self.end_time)
                ].copy()
                if not trades_df_filtered.empty:
                    required_cols = ['symbol', 'price', 'quantity', 'buyer', 'seller']
                    if not all(col in trades_df_filtered.columns for col in required_cols):
                        print(f"Warning: Skipping {data_path} for trades/inference - missing columns.")
                        continue

                    # Ensure correct types
                    for col in ['quantity', 'price']:
                        trades_df_filtered[col] = pd.to_numeric(trades_df_filtered[col], errors='coerce')
                    trades_df_filtered.dropna(subset=['quantity', 'price'], inplace=True)
                    trades_df_filtered['quantity'] = trades_df_filtered['quantity'].astype(int)
                    trades_df_filtered['price'] = trades_df_filtered['price'].astype(int)
                    trades_df_filtered['buyer'] = trades_df_filtered['buyer'].fillna('').astype(str)
                    trades_df_filtered['seller'] = trades_df_filtered['seller'].fillna('').astype(str)

                    # Add all valid trades within the time range to the raw list
                    all_trades_list.append(trades_df_filtered)

                    # Perform inference only on submission trades within the time range
                    submission_trades_this_log = trades_df_filtered[
                        (trades_df_filtered['buyer'] == "SUBMISSION") | (trades_df_filtered['seller'] == "SUBMISSION")
                    ].copy()

                    if not submission_trades_this_log.empty:
                        # Use the precomputed explicit depths (already time-filtered)
                        log_excess_buys, log_excess_sells = self._calculate_excess_volume_per_log(
                            submission_trades_this_log, self.explicit_order_depths_cache
                        )
                        # Aggregate maxes across logs
                        for ts, products_data in log_excess_buys.items():
                            for prod, prices_data in products_data.items():
                                for price, vol in prices_data.items():
                                    agg_max_bot_buys[ts][prod][price] = max(agg_max_bot_buys[ts][prod].get(price, 0), vol)
                        for ts, products_data in log_excess_sells.items():
                            for prod, prices_data in products_data.items():
                                for price, vol in prices_data.items():
                                    agg_max_bot_sells[ts][prod][price] = max(agg_max_bot_sells[ts][prod].get(price, 0), vol)

                    # Update products based on trades found in *any* log within the time range
                    if 'symbol' in trades_df_filtered.columns:
                         symbols_in_trades = trades_df_filtered['symbol'].unique()
                         all_products.update(p for p in symbols_in_trades if isinstance(p, str) and p)

        # Consolidate all raw trades
        if not all_trades_list:
            print("Warning: No trade history found across all logs in the time range.")
            self.raw_all_trades = pd.DataFrame(columns=['symbol', 'price', 'quantity', 'buyer', 'seller']) # Ensure columns exist
            self.raw_all_trades.index.name = 'timestamp'
        else:
            self.raw_all_trades = pd.concat(all_trades_list)
            if not self.raw_all_trades.index.is_monotonic_increasing:
                self.raw_all_trades.sort_index(inplace=True)
            # Robust duplicate removal considering index and essential columns
            key_cols = ['symbol', 'price', 'quantity', 'buyer', 'seller']
            initial_count = len(self.raw_all_trades)
            # Identify duplicates based on index AND key columns
            is_duplicate = self.raw_all_trades.index.duplicated(keep='first') & self.raw_all_trades.duplicated(subset=key_cols, keep=False)
            # Keep the first occurrence if the index is duplicated OR if the row content is duplicated
            self.raw_all_trades = self.raw_all_trades[~(is_duplicate)] # Keep non-duplicates
            # A more aggressive approach might be needed if logs truly have overlapping identical trades at same timestamp
            # self.raw_all_trades = self.raw_all_trades.loc[~self.raw_all_trades.index.duplicated(keep='first')]
            print(f"DEBUG load: Consolidated self.raw_all_trades shape: {self.raw_all_trades.shape} (removed {initial_count - len(self.raw_all_trades)} duplicates)")


        # Finalize product list
        self.products = sorted(list(all_products))
        if not self.products:
             # If no products found in primary log or any trades, try getting from constants as last resort
             print("Warning: No products determined from data. Trying constants...")
             self.products = sorted(list(constants.POSITION_LIMITS.keys()))
             if not self.products:
                  raise ValueError("Could not determine products from data or constants.")
        print(f"DEBUG load: Final product list determined: {self.products}")

        # Create listings based on the final product list
        self.listings = {prod: Listing(prod, prod, "SEASHELLS") for prod in self.products}

         # --- REVISED AGAIN: Build MARGINAL inferred bot liquidity cache ---
        print("Building MARGINAL inferred bot liquidity cache...")
        self.inferred_bot_liquidity_cache = defaultdict(lambda: defaultdict(OrderDepth))
        timestamps_with_inferred = set(agg_max_bot_buys.keys()).union(agg_max_bot_sells.keys())

        for ts in sorted(list(timestamps_with_inferred)): # Process timestamps in order
            prod_max_buy_vol_at_price = agg_max_bot_buys.get(ts, {}) # {prod: {price: max_vol}}
            prod_max_sell_vol_at_price = agg_max_bot_sells.get(ts, {}) # {prod: {price: max_vol}}

            for prod in self.products:
                max_buy_vol = prod_max_buy_vol_at_price.get(prod, {})
                max_sell_vol = prod_max_sell_vol_at_price.get(prod, {})

                if not max_buy_vol and not max_sell_vol:
                    continue # Skip if no inferred liquidity for this product/ts

                marginal_depth = OrderDepth()

                # Process BUY liquidity (Bots buying = Our Sell Fills)
                # Convert max vol at price P (meaning they buy X @ P or better)
                # to marginal vol *only* at price P.
                if max_buy_vol:
                    sorted_buy_prices = sorted(max_buy_vol.keys(), reverse=True) # Best price first (highest)
                    accounted_for_volume = 0
                    for i, price in enumerate(sorted_buy_prices):
                        total_volume_at_or_better = max_buy_vol[price]
                        # Marginal volume at this specific price level
                        marginal_volume = max(0, total_volume_at_or_better - accounted_for_volume)
                        if marginal_volume > 0:
                            marginal_depth.buy_orders[price] = marginal_volume
                        accounted_for_volume = total_volume_at_or_better # Update volume accounted for by better prices

                # Process SELL liquidity (Bots selling = Our Buy Fills)
                # Convert max vol at price P (meaning they sell X @ P or better)
                # to marginal vol *only* at price P.
                if max_sell_vol:
                    sorted_sell_prices = sorted(max_sell_vol.keys()) # Best price first (lowest)
                    accounted_for_volume = 0
                    for i, price in enumerate(sorted_sell_prices):
                        total_volume_at_or_better = max_sell_vol[price]
                        # Marginal volume at this specific price level
                        marginal_volume = max(0, total_volume_at_or_better - accounted_for_volume)
                        if marginal_volume > 0:
                             # Store sell orders with NEGATIVE volume
                            marginal_depth.sell_orders[price] = -marginal_volume
                        accounted_for_volume = total_volume_at_or_better # Update volume accounted for

                # Assign the calculated marginal depth
                if marginal_depth.buy_orders or marginal_depth.sell_orders:
                    self.inferred_bot_liquidity_cache[ts][prod] = marginal_depth
                # else: # Debug if needed
                #     if max_buy_vol or max_sell_vol:
                #         print(f"WARN Cache Build: No marginal volume derived for {prod}@{ts} despite inferred data.")
                #         print(f"  Max Buy Vols: {max_buy_vol}")
                #         print(f"  Max Sell Vols: {max_sell_vol}")


        print(f"Final MARGINAL inferred bot liquidity cache created for {len(self.inferred_bot_liquidity_cache)} timestamps.")
        # --- END REVISED CACHE BUILDING ---


    def _calculate_excess_volume_per_log(self, submission_trades_df: pd.DataFrame, explicit_book_cache: Dict[int, Dict[Symbol, OrderDepth]]):
        """Calculates excess bot liquidity implied by submission trades against explicit books for a single log."""
        log_excess_bot_buys = defaultdict(lambda: defaultdict(Counter)) # What bots must have been BUYING (to fill our sells)
        log_excess_bot_sells = defaultdict(lambda: defaultdict(Counter)) # What bots must have been SELLING (to fill our buys)
        grouped_trades = submission_trades_df.groupby([submission_trades_df.index, 'symbol'])

        for (timestamp, product), trades in grouped_trades:
            # Get the explicit book state *at that timestamp*
            # Use .get() for safety, default to empty dict/OrderDepth
            initial_explicit_book = explicit_book_cache.get(timestamp, {}).get(product)
            if not initial_explicit_book:
                # print(f"Debug Excess: No explicit book for {product}@{timestamp}, skipping inference.")
                continue # Cannot infer without knowing the explicit state

            # Aggregate submission buys/sells at this specific timestamp/product
            sub_buys_at_price = Counter()
            sub_sells_at_price = Counter()
            for _, row in trades.iterrows():
                price = int(row['price'])
                quantity = int(row['quantity'])
                if row['buyer'] == "SUBMISSION":
                    sub_buys_at_price[price] += quantity
                elif row['seller'] == "SUBMISSION":
                    sub_sells_at_price[price] += quantity

            # Simulate OUR BUYS against the explicit SELL side
            if sub_buys_at_price:
                explicit_sells_copy = copy.deepcopy(initial_explicit_book.sell_orders) # Operate on a copy
                our_buy_fills_vs_explicit = Counter()
                # Iterate through our buy prices (ascending)
                for sub_p in sorted(sub_buys_at_price.keys()):
                    our_buy_vol_at_p = sub_buys_at_price[sub_p]
                    vol_remaining_to_fill = our_buy_vol_at_p
                    # Iterate through available explicit sell prices (ascending)
                    for book_p in sorted(explicit_sells_copy.keys()):
                        if vol_remaining_to_fill <= 0: break
                        if book_p > sub_p: continue # Explicit ask is higher than our buy price, no match

                        book_v = abs(explicit_sells_copy[book_p])
                        filled_on_explicit = min(vol_remaining_to_fill, book_v)

                        if filled_on_explicit > 0:
                            our_buy_fills_vs_explicit[sub_p] += filled_on_explicit
                            explicit_sells_copy[book_p] += filled_on_explicit # Consume volume (+ makes it closer to 0)
                            vol_remaining_to_fill -= filled_on_explicit
                            if explicit_sells_copy[book_p] == 0:
                                del explicit_sells_copy[book_p]

                    # Calculate excess: Our total buy volume at this price MINUS what the explicit book filled
                    excess_needed_from_bots = our_buy_vol_at_p - our_buy_fills_vs_explicit[sub_p]
                    if excess_needed_from_bots > 0:
                        # This excess must have been filled by BOT SELL orders at or below our buy price (sub_p)
                        # We record the maximum *volume* the bots sold at exactly *our* price (sub_p)
                        log_excess_bot_sells[timestamp][product][sub_p] = excess_needed_from_bots

            # Simulate OUR SELLS against the explicit BUY side
            if sub_sells_at_price:
                explicit_buys_copy = copy.deepcopy(initial_explicit_book.buy_orders) # Operate on a copy
                our_sell_fills_vs_explicit = Counter()
                 # Iterate through our sell prices (descending)
                for sub_p in sorted(sub_sells_at_price.keys(), reverse=True):
                    our_sell_vol_at_p = sub_sells_at_price[sub_p]
                    vol_remaining_to_fill = our_sell_vol_at_p
                     # Iterate through available explicit buy prices (descending)
                    for book_p in sorted(explicit_buys_copy.keys(), reverse=True):
                        if vol_remaining_to_fill <= 0: break
                        if book_p < sub_p: continue # Explicit bid is lower than our sell price, no match

                        book_v = abs(explicit_buys_copy[book_p])
                        filled_on_explicit = min(vol_remaining_to_fill, book_v)

                        if filled_on_explicit > 0:
                            our_sell_fills_vs_explicit[sub_p] += filled_on_explicit
                            explicit_buys_copy[book_p] -= filled_on_explicit # Consume volume
                            vol_remaining_to_fill -= filled_on_explicit
                            if explicit_buys_copy[book_p] == 0:
                                del explicit_buys_copy[book_p]

                    # Calculate excess: Our total sell volume at this price MINUS what the explicit book filled
                    excess_needed_from_bots = our_sell_vol_at_p - our_sell_fills_vs_explicit[sub_p]
                    if excess_needed_from_bots > 0:
                        # This excess must have been filled by BOT BUY orders at or above our sell price (sub_p)
                         # We record the maximum *volume* the bots bought at exactly *our* price (sub_p)
                        log_excess_bot_buys[timestamp][product][sub_p] = excess_needed_from_bots

        return log_excess_bot_buys, log_excess_bot_sells


    def _precompute_explicit_order_depths(self, market_df: pd.DataFrame):
        """Builds a cache of OrderDepth objects from the market data feed (L1-3)."""
        print("Precomputing explicit order depths from primary market data...")
        self.explicit_order_depths_cache = {}
        if market_df.empty:
            print("Warning: Market data empty, cannot precompute explicit depths.")
            return
        if not pd.api.types.is_numeric_dtype(market_df.index):
             print(f"Warning: Market data index type is {market_df.index.dtype}, expected numeric. Attempting conversion.")
             try:
                 market_df.index = pd.to_numeric(market_df.index)
             except Exception as e:
                 print(f"ERROR: Cannot convert market data index to numeric: {e}. Cannot group by timestamp.")
                 return

        # Ensure required columns exist and handle potential missing data gracefully
        required_cols = ['product'] + [f'{side}_price_{i}' for side in ['bid','ask'] for i in [1,2,3]] \
                                    + [f'{side}_volume_{i}' for side in ['bid','ask'] for i in [1,2,3]]
        missing_cols = [col for col in required_cols if col not in market_df.columns]
        if missing_cols:
            print(f"Warning: Missing columns in market data needed for explicit depth: {missing_cols}. Depths might be incomplete.")

        # Convert price/volume columns to numeric, coercing errors
        for col in required_cols:
            if col in market_df.columns and col != 'product':
                if not pd.api.types.is_numeric_dtype(market_df[col]):
                     market_df[col] = pd.to_numeric(market_df[col], errors='coerce')


        try:
            # Group by timestamp index
            grouped = market_df.groupby(market_df.index)
        except Exception as e:
             print(f"ERROR: Failed to group market_df by index for explicit depths: {e}")
             return

        for timestamp, group in grouped:
            ts_depths: Dict[Symbol, OrderDepth] = {}
            # Iterate through unique products within this timestamp group
            products_in_group = group['product'].unique() if 'product' in group.columns else []

            for product in products_in_group:
                if not isinstance(product, str) or not product: continue # Skip invalid product names

                order_depth = OrderDepth()
                # Get the row(s) for this specific product at this timestamp
                # Usually only one row, but handle potential duplicates by taking the first
                product_rows = group[group['product'] == product]
                if product_rows.empty: continue
                product_row = product_rows.iloc[0] # Use the first row for this product/timestamp

                # Extract Buy Orders (Levels 1-3)
                for i in range(1, 4):
                    bp_col, bv_col = f'bid_price_{i}', f'bid_volume_{i}'
                    if bp_col in product_row.index and bv_col in product_row.index:
                        bid_price = product_row[bp_col]
                        bid_volume = product_row[bv_col]
                        # Check for valid numeric data and positive volume
                        if pd.notna(bid_price) and pd.notna(bid_volume) and bid_volume > 0:
                            try:
                                order_depth.buy_orders[int(bid_price)] = order_depth.buy_orders.get(int(bid_price), 0) + int(bid_volume)
                            except (ValueError, TypeError): pass # Ignore conversion errors

                # Extract Sell Orders (Levels 1-3)
                for i in range(1, 4):
                    ap_col, av_col = f'ask_price_{i}', f'ask_volume_{i}'
                    if ap_col in product_row.index and av_col in product_row.index:
                        ask_price = product_row[ap_col]
                        ask_volume = product_row[av_col]
                         # Check for valid numeric data and positive volume
                        if pd.notna(ask_price) and pd.notna(ask_volume) and ask_volume > 0:
                            try:
                                # Store sell orders with negative volume internally
                                order_depth.sell_orders[int(ask_price)] = order_depth.sell_orders.get(int(ask_price), 0) - int(ask_volume)
                            except (ValueError, TypeError): pass # Ignore conversion errors

                # Only add depth if it contains orders
                if order_depth.buy_orders or order_depth.sell_orders:
                    ts_depths[product] = order_depth

            # Only add timestamp to cache if there was depth for at least one product
            if ts_depths:
                self.explicit_order_depths_cache[timestamp] = ts_depths

        print(f"Explicit order depths precomputed for {len(self.explicit_order_depths_cache)} timestamps.")


    def run(
        self,
        explicit_book_override: Optional[Dict[int, Dict[Symbol, OrderDepth]]] = None,
        inferred_book_override: Optional[Dict[int, Dict[Symbol, OrderDepth]]] = None
        ):
        # Determine run mode based on explicit override presence
        run_mode = "OVERRIDE_EXPLICIT" if explicit_book_override is not None else "NORMAL"
        # Note: inferred_book_override might be present even in NORMAL mode if passed from app.py permutation
        print(f"\n--- ENTERING Backtester.run (Mode: {run_mode}) ---")
        print(f"Explicit Override Provided: {explicit_book_override is not None}")
        print(f"Inferred Override Provided: {inferred_book_override is not None}")


        traderData = "" # Initialize traderData for the run
        print("Resetting internal state for run...")
        self.current_position = {p: 0 for p in self.products}
        self.pnl = {p: 0.0 for p in self.products}
        self.cash = {p: 0.0 for p in self.products}
        self.pnl_over_time = {p: [] for p in self.products}
        self.total_pnl_over_time = []
        self.sandbox_logs_capture = []
        self.all_trades_log_output = []
        self.last_processed_timestamp = -1

        # --- Initialize activity_log_output for this run ---
        # Start with the index and product column from the original market data (time-filtered)
        if not hasattr(self, 'market_data') or self.market_data.empty:
            print("Error: Cannot initialize activity log - original market data structure is missing or empty.")
            # Create a minimal empty frame to avoid crashing later, but output will be minimal
            self.activity_log_output = pd.DataFrame(index=pd.Index([], dtype='int64', name='timestamp'),
                                                     columns=['product', 'profit_and_loss']) # Add essential cols
        else:
            # Select essential non-price columns if they exist, otherwise just product
            cols_to_keep = ['product']
            if 'day' in self.market_data.columns: cols_to_keep.append('day')
            # Add other non-price/vol columns you want to preserve here...

            # Ensure we only try to keep columns that actually exist
            cols_to_keep = [col for col in cols_to_keep if col in self.market_data.columns]
            if 'product' not in cols_to_keep: # Ensure product is always present
                 print("Warning: 'product' column missing from market_data, activity log may be incomplete.")
                 self.activity_log_output = self.market_data[[]].copy() # Keep only index
                 self.activity_log_output['product'] = np.nan # Add product column as NaN
            else:
                 self.activity_log_output = self.market_data[cols_to_keep].copy()

            # Define all potential price/volume/pnl columns to be reset/added
            price_vol_cols = [
                f'{side}_{ptype}_{level}' # Corrected order: side_ptype_level
                for side in ['bid', 'ask']
                for ptype in ['price', 'volume']
                for level in [1, 2, 3]
            ]
            other_cols_to_reset = ['mid_price', 'profit_and_loss']
            cols_to_initialize = price_vol_cols + other_cols_to_reset

            # Add/Reset these columns with NaN
            for col in cols_to_initialize:
                self.activity_log_output[col] = np.nan

            # Ensure numeric type for PnL and mid_price after initialization
            for col in ['profit_and_loss', 'mid_price']:
                 if col in self.activity_log_output.columns:
                     self.activity_log_output[col] = pd.to_numeric(self.activity_log_output[col], errors='coerce')

        print(f"Internal state reset complete. Activity log initialized with columns: {self.activity_log_output.columns.tolist()}")
        # --- End Activity Log Init ---


        perf_timer_start = time.time()

        # --- Determine Active Caches and Timestamps ---
        active_explicit_book_cache: Dict[int, Dict[Symbol, OrderDepth]]
        active_inferred_cache: Dict[int, Dict[Symbol, OrderDepth]]
        active_fv_book_cache: Dict[int, Dict[Symbol, OrderDepth]] # Book used for Fair Value / PnL

        if run_mode == "OVERRIDE_EXPLICIT":
            active_explicit_book_cache = explicit_book_override
            # In override mode, inferred matching is typically disabled
            # UNLESS an inferred override is *also* explicitly provided
            if inferred_book_override is not None:
                 active_inferred_cache = inferred_book_override
                 use_inferred_book = True # Use the provided inferred override
                 print("RUN MODE: OVERRIDE_EXPLICIT (with Inferred Override) - Using overrides for both. Matching inferred.")
            else:
                 active_inferred_cache = {} # No inferred override, no matching
                 use_inferred_book = False
                 print("RUN MODE: OVERRIDE_EXPLICIT (No Inferred Override) - Using explicit override. Inferred matching disabled.")
            active_fv_book_cache = active_explicit_book_cache # PnL uses the explicit override book
        else: # NORMAL mode
            active_explicit_book_cache = self.explicit_order_depths_cache
            # Check if an inferred override was passed (e.g., from permutation)
            if inferred_book_override is not None:
                 active_inferred_cache = inferred_book_override
                 use_inferred_book = True # Use the provided override
                 print("RUN MODE: NORMAL (with Inferred Override) - Using explicit cache. Using inferred override for matching.")
            else: # Standard normal run, use the originally calculated inferred cache
                 active_inferred_cache = self.inferred_bot_liquidity_cache
                 use_inferred_book = True # Use the original inferred cache
                 print("RUN MODE: NORMAL (Standard) - Using explicit cache. Using original inferred cache for matching.")
            # PnL calculation always uses the explicit book in normal mode
            active_fv_book_cache = self.explicit_order_depths_cache
        # Determine timestamps to iterate over based on the *explicit* book (original or override)
        # This defines the "ticks" of the simulation
        timestamps_to_iterate = sorted([
            ts for ts in active_explicit_book_cache.keys()
            if self.start_time <= ts <= self.end_time
        ])

        if not timestamps_to_iterate:
            print(f"Warning: No timestamps found in active explicit book cache within time range ({self.start_time}-{self.end_time}). No simulation steps will run.")
            self._finalize_activity_log() # Finalize the (likely empty) log
            self._generate_output()
            return
        print(f"Simulating {len(timestamps_to_iterate)} timestamps from active explicit book cache...")
        # --- End Determine ---

        # --- Main Simulation Loop ---
        for i, timestamp in enumerate(timestamps_to_iterate):

            # --- Get Market Trades for this interval ---
            current_step_market_trades_dict = defaultdict(list)
            start_range, end_range = self.last_processed_timestamp + 1, timestamp
            try:
                if hasattr(self, 'raw_all_trades') and not self.raw_all_trades.empty:
                    # Ensure index is sorted for slicing
                    if not self.raw_all_trades.index.is_monotonic_increasing:
                        self.raw_all_trades.sort_index(inplace=True)
                    # Slice using loc for timestamp range (inclusive)
                    trades_in_interval = self.raw_all_trades.loc[start_range:end_range]
                    if not trades_in_interval.empty:
                        for idx, row in trades_in_interval.iterrows():
                            # Check essential fields and convert
                            symbol=row.get('symbol')
                            price=row.get('price')
                            quantity=row.get('quantity')
                            buyer=row.get('buyer', '') # Default to empty string
                            seller=row.get('seller', '') # Default to empty string
                            if all(v is not None for v in [symbol, price, quantity]) and symbol in self.products:
                                try:
                                    trade = Trade(symbol, int(price), int(quantity), buyer, seller, int(idx))
                                    current_step_market_trades_dict[symbol].append(trade)
                                except (ValueError, TypeError) as conversion_err:
                                     print(f"Warn: Skipping trade due to conversion error @{idx}: {conversion_err}, Data: {row.to_dict()}")
            except KeyError:
                 pass # Range not found in index, common for first timestamp
            except Exception as e:
                print(f"Error slicing self.raw_all_trades between {start_range}-{end_range} @{timestamp}: {e}")
                traceback.print_exc()
            # --- End Market Trades ---


            # --- Prepare State for Trader ---
            # Trader always sees the state based on the *active explicit book* (original or override)
            current_depths_for_state = copy.deepcopy(active_explicit_book_cache.get(timestamp, {}))
            # Ensure all products exist in the state, even if empty
            for prod in self.products:
                if prod not in current_depths_for_state:
                    current_depths_for_state[prod] = OrderDepth()

            state = TradingState(
                traderData,
                timestamp,
                self.listings,
                current_depths_for_state, # Depths trader sees
                {}, # own_trades - Deprecated? (Filled by backtester matching) - Should be empty dict based on docs
                dict(current_step_market_trades_dict), # market_trades since last call
                self.current_position.copy(), # Current positions
                Observation({}, {}) # observations - TODO: Populate if needed
            )
            # --- End Prepare State ---

            # --- Call Trader ---
            log_stream = io.StringIO()
            sandbox_log_output = ""
            trader_orders_dict: Dict[Symbol, List[Order]] = {}
            conversions = 0 # Placeholder for conversion logic if implemented
            start_run_time = time.time()
            try:
                # Redirect stdout to capture prints from trader
                with redirect_stdout(log_stream):
                    trader_orders_dict, conversions, traderData = self.trader.run(state)
                    # Basic validation of trader output
                    if not isinstance(trader_orders_dict, dict):
                        sandbox_log_output += f"\nError: Trader returned orders of type {type(trader_orders_dict)}, expected dict."
                        trader_orders_dict = {} # Reset to empty dict
                    if not isinstance(traderData, str):
                         sandbox_log_output += f"\nWarning: Trader returned traderData of type {type(traderData)}, expected str. Coercing."
                         traderData = str(traderData) # Attempt coercion

            except Exception as e:
                print(f"\nTRADER CRASH @{timestamp}: {e}")
                traceback.print_exc()
                sandbox_log_output += f"\n!!! TRADER CRASH: {e} !!!"
                trader_orders_dict = {} # Clear orders on crash
            run_time = time.time() - start_run_time
            captured_lambda_log = log_stream.getvalue()
            log_stream.close()
            if run_time * 1000 > 900: # Check runtime limit (adjust as needed)
                sandbox_log_output += f"\n>> WARNING: Trader execution time > 900ms ({run_time*1000:.2f}ms) <<"
            # --- End Call Trader ---


            # --- Process Orders and Match ---
            executed_own_trades_this_step = defaultdict(list)
            explicit_depths_for_matching = copy.deepcopy(active_explicit_book_cache.get(timestamp, {}))
            # Use the determined active_inferred_cache for matching
            inferred_depths_for_matching = copy.deepcopy(active_inferred_cache.get(timestamp, {})) if use_inferred_book else {}
            if trader_orders_dict:
                all_submitted_orders: List[Order] = []
                # Validate and flatten orders
                for product, orders in trader_orders_dict.items():
                   if product not in self.listings:
                       sandbox_log_output += f"\nWarning: Trader sent orders for unknown product '{product}', skipping."
                       continue
                   if not isinstance(orders, list):
                        sandbox_log_output += f"\nWarning: Trader sent orders for '{product}' not as a list (type: {type(orders)}), skipping."
                        continue

                   for order in orders:
                       if isinstance(order, Order) and \
                          isinstance(order.symbol, str) and order.symbol == product and \
                          isinstance(order.price, int) and \
                          isinstance(order.quantity, int) and \
                          order.quantity != 0:
                            all_submitted_orders.append(order)
                       else:
                            sandbox_log_output += f"\nWarning: Invalid order object received for {product}: {order}, skipping."

                # Group validated orders by product
                grouped_player_orders = defaultdict(list)
                for order in all_submitted_orders:
                     grouped_player_orders[order.symbol].append(order)

                # Process orders per product
                for product, product_orders in grouped_player_orders.items():
                    limit = self.position_limit.get(product, 0)
                    current_pos = self.current_position.get(product, 0)
                    potential_buy_qty = sum(o.quantity for o in product_orders if o.quantity > 0)
                    potential_sell_qty = sum(abs(o.quantity) for o in product_orders if o.quantity < 0)

                    # Check position limits before matching
                    limit_breached = False
                    if not self.ignore_limits:
                        if (current_pos + potential_buy_qty > limit):
                            limit_breached = True
                            sandbox_log_output += f"\nLIMIT BREACH (Buy): {product}@{timestamp}. Pos:{current_pos}, Limit:{limit}, Trying to buy:{potential_buy_qty}. Orders Cancelled."
                        elif (current_pos - potential_sell_qty < -limit):
                             limit_breached = True
                             sandbox_log_output += f"\nLIMIT BREACH (Sell): {product}@{timestamp}. Pos:{current_pos}, Limit:{-limit}, Trying to sell:{potential_sell_qty}. Orders Cancelled."

                    if limit_breached:
                        # Do not process orders for this product at this timestamp
                        continue

                    # Match valid orders
                    # Sort orders for potentially more deterministic matching (optional)
                    # product_orders.sort(key=lambda o: o.price, reverse=(o.quantity < 0)) # Example sort

                    for order in product_orders:
                          order_copy = copy.deepcopy(order)

                          # 1. Match Explicit Book (Original or Override)
                          trades_explicit, sandbox_log_output = self._match_against_book(
                              timestamp, order_copy, explicit_depths_for_matching, sandbox_log_output,
                              "ExplicitBook", "any"
                          )
                          executed_own_trades_this_step[product].extend(trades_explicit)

                          # 2. Match Inferred Book (Original or Override, if enabled)
                          if use_inferred_book and abs(order_copy.quantity) > 0 and product in inferred_depths_for_matching:
                               # Pass only the specific product's inferred book for matching
                               current_prod_inferred = {product: inferred_depths_for_matching[product]}
                               trades_bot, sandbox_log_output = self._match_against_book(
                                   timestamp, order_copy, current_prod_inferred, sandbox_log_output,
                                   "InferredBot", self.bot_behavior # Use configured bot matching rule
                               )
                               executed_own_trades_this_step[product].extend(trades_bot)


            # --- Log Results for the Step ---
            # Store sandbox/lambda logs
            self.sandbox_logs_capture.append({
                "sandboxLog": sandbox_log_output,
                "lambdaLog": captured_lambda_log,
                "timestamp": timestamp
            })

            # Add executed player trades to the final log output
            for trades in executed_own_trades_this_step.values():
                for trade in trades:
                    self.all_trades_log_output.append(self._trade_to_dict(trade))

            # Add non-player market trades from the interval to the final log output for context
            if not current_step_market_trades_dict: # Check if dict is empty
                pass
            else:
                for product_trades in current_step_market_trades_dict.values():
                    for trade in product_trades:
                         # Add only if not involving SUBMISSION (already logged above)
                        if trade.buyer != "SUBMISSION" and trade.seller != "SUBMISSION":
                             self.all_trades_log_output.append(self._trade_to_dict(trade))


            # Calculate and log PnL based on the Fair Value book cache
            self._calculate_and_log_pnl(timestamp, active_fv_book_cache)
            # --- End Log Results ---

            self.last_processed_timestamp = timestamp
        # --- End Main Simulation Loop ---

        perf_timer_end = time.time()
        print(f"\nSim loop finished ({run_mode}) in {perf_timer_end - perf_timer_start:.2f}s.")

        # Print tail of activity log for quick check
        if hasattr(self, 'activity_log_output') and not self.activity_log_output.empty:
            print("\nActivity Log Tail (End of Run):")
            print(self.activity_log_output.tail())

        self._finalize_activity_log() # Prepare the activity log for output
        self._generate_output()      # Create the final output string


    def _match_against_book(
        self,
        timestamp: int,
        order: Order, # The order to be matched (will be modified in place)
        order_books: Dict[Symbol, OrderDepth], # The books to match against (will be modified in place)
        log: str, # Current sandbox log string
        book_type: str, # Identifier for logging ("ExplicitBook", "InferredBot")
        match_behavior: Literal["none", "eq", "lt", "lte", "any"] # Rule for matching inferred bots
        ) -> Tuple[List[Trade], str]:
        """Matches a single order against a provided set of order books."""

        trades_made: List[Trade] = []
        product = order.symbol

        # Check if the product exists in the books for matching
        book = order_books.get(product)
        if not book:
            return trades_made, log # No book for this product

        original_order_quantity = order.quantity # Store for comparison later

        if order.quantity > 0:  # Player wants to BUY
            # Match against SELL orders in the book
            if not book.sell_orders: return trades_made, log # No sell orders to match against

            # Iterate through sell prices in ascending order (best price first)
            # Sort keys during iteration to handle potential modifications
            sell_prices = sorted(book.sell_orders.keys())
            for price in sell_prices:
                if order.quantity <= 0: break # Order fully filled

                # Basic check: Can we afford this price?
                if price > order.price: continue # Book price is higher than our max buy price

                # --- Behavior check ONLY for inferred book ---
                # 'any' behavior for explicit book means we just check price affordability (done above)
                if book_type == "InferredBot" and match_behavior != "any":
                    allowed = False
                    # Note: Prices are book prices (sell side)
                    if match_behavior == "eq": allowed = (price == order.price)
                    elif match_behavior == "lt": allowed = (price < order.price) # Bot sells strictly below our limit
                    elif match_behavior == "lte": allowed = (price <= order.price) # Bot sells at or below our limit
                    elif match_behavior == "none": allowed = False # Explicitly disable matching this book type
                    if not allowed:
                        # log += f"\nDebug: Bot behavior '{match_behavior}' disallowed match: Buy@{order.price} vs BotSell@{price}"
                        continue # Skip this price level based on behavior rule
                # --- End behavior check ---

                # Check if price still exists in the original dict (could be removed by prior fills)
                if price not in book.sell_orders: continue

                book_vol = abs(book.sell_orders[price]) # Available volume at this price
                vol_to_trade = min(order.quantity, book_vol) # How much can we trade?

                if vol_to_trade <= 0: continue # Should not happen, but safety check

                # Create trade record
                trade = Trade(product, price, vol_to_trade, "SUBMISSION", book_type, timestamp)
                trades_made.append(trade)

                # Update state: Position and Cash
                self.current_position[product] = self.current_position.get(product, 0) + vol_to_trade
                self.cash[product] = self.cash.get(product, 0.0) - (price * vol_to_trade)

                # Update remaining order quantity
                order.quantity -= vol_to_trade

                # Update book volume (consume liquidity)
                book.sell_orders[price] += vol_to_trade # Add positive volume to negative sell volume
                if book.sell_orders[price] == 0:
                    del book.sell_orders[price] # Remove price level if fully consumed

        elif order.quantity < 0:  # Player wants to SELL
             # Match against BUY orders in the book
            if not book.buy_orders: return trades_made, log # No buy orders to match against

            # Iterate through buy prices in descending order (best price first)
            buy_prices = sorted(book.buy_orders.keys(), reverse=True)
            for price in buy_prices:
                if order.quantity >= 0: break # Order fully filled (quantity becomes 0)

                # Basic check: Is the price acceptable?
                if price < order.price: continue # Book price is lower than our min sell price

                # --- Behavior check ONLY for inferred book ---
                if book_type == "InferredBot" and match_behavior != "any":
                    allowed = False
                     # Note: Prices are book prices (buy side)
                    if match_behavior == "eq": allowed = (price == order.price)
                    # We are selling, so we want bot buy price >= our sell price
                    elif match_behavior == "lt": allowed = (price > order.price) # Bot buys strictly above our limit (better for us)
                    elif match_behavior == "lte": allowed = (price >= order.price) # Bot buys at or above our limit
                    elif match_behavior == "none": allowed = False
                    if not allowed:
                         # log += f"\nDebug: Bot behavior '{match_behavior}' disallowed match: Sell@{order.price} vs BotBuy@{price}"
                         continue
                # --- End behavior check ---

                if price not in book.buy_orders: continue

                book_vol = abs(book.buy_orders[price])
                # vol_to_trade is positive quantity
                vol_to_trade = min(abs(order.quantity), book_vol)

                if vol_to_trade <= 0: continue

                # Create trade record
                trade = Trade(product, price, vol_to_trade, book_type, "SUBMISSION", timestamp)
                trades_made.append(trade)

                 # Update state: Position and Cash
                self.current_position[product] = self.current_position.get(product, 0) - vol_to_trade
                self.cash[product] = self.cash.get(product, 0.0) + (price * vol_to_trade)

                # Update remaining order quantity (add positive volume to negative sell quantity)
                order.quantity += vol_to_trade

                # Update book volume (consume liquidity)
                book.buy_orders[price] -= vol_to_trade
                if book.buy_orders[price] == 0:
                    del book.buy_orders[price]

        # Log fills
        # if trades_made:
        #     log += f"\nMatched {book_type}: {product} Qty:{abs(original_order_quantity - order.quantity)} Trades:{len(trades_made)}"

        return trades_made, log

    def _calculate_and_log_pnl(self, timestamp: int, book_cache_for_fv: Dict[int, Dict[Symbol, OrderDepth]]):
        """Calculates PnL based on current positions, cash, and fair value derived from book_cache_for_fv.
           Logs PnL, mid_price, and L1-3 book state into self.activity_log_output.
        """
        total_pnl_at_t = 0.0
        # Ensure DataFrame index is sorted before potential ffill lookups
        if hasattr(self.activity_log_output, 'index') and not self.activity_log_output.index.is_monotonic_increasing:
             try:
                  self.activity_log_output.sort_index(inplace=True)
             except Exception as sort_err:
                  print(f"Error sorting activity_log_output index: {sort_err}")


        for product in self.products:
             # Get last known PnL state (before this timestamp's calculation)
             last_known_pnl = self.pnl.get(product, 0.0)
             calculated_pnl = last_known_pnl # Default to last known if calculation fails
             current_cash = self.cash.get(product, 0.0)
             current_pos = self.current_position.get(product, 0)

             # --- Initialize placeholders for book levels and calculated mid-price ---
             logged_bid_p: Dict[int, float] = {1: np.nan, 2: np.nan, 3: np.nan}
             logged_bid_v: Dict[int, float] = {1: np.nan, 2: np.nan, 3: np.nan}
             logged_ask_p: Dict[int, float] = {1: np.nan, 2: np.nan, 3: np.nan}
             logged_ask_v: Dict[int, float] = {1: np.nan, 2: np.nan, 3: np.nan}
             calculated_mid_price: float = np.nan # Reset for this product/timestamp
             fair_value: float = np.nan # Initialize fair value for PnL calc

             # Check if we should calculate PnL for this product
             if product in self.fair_price_calc and product in self.current_position:
                 # Get the order depth from the specified cache for Fair Value calculation
                 order_depth = book_cache_for_fv.get(timestamp, {}).get(product, OrderDepth())

                 # --- Extract Top Levels from the Fair Value Book Cache ---
                 try:
                      if order_depth.buy_orders:
                           sorted_bids = sorted(order_depth.buy_orders.keys(), reverse=True)
                           for i, price in enumerate(sorted_bids[:3]): # Get top 3 bids
                               level = i + 1
                               logged_bid_p[level] = float(price)
                               logged_bid_v[level] = float(order_depth.buy_orders[price])
                      if order_depth.sell_orders:
                           sorted_asks = sorted(order_depth.sell_orders.keys())
                           for i, price in enumerate(sorted_asks[:3]): # Get top 3 asks
                               level = i + 1
                               logged_ask_p[level] = float(price)
                               logged_ask_v[level] = float(abs(order_depth.sell_orders[price])) # Store positive volume

                      # --- Calculate Mid-Price from Logged Top Levels ---
                      best_bid = logged_bid_p[1]
                      best_ask = logged_ask_p[1]
                      if pd.notna(best_bid) and pd.notna(best_ask):
                           calculated_mid_price = (best_bid + best_ask) / 2.0
                      # else: mid_price remains NaN
                 except Exception as bp_err:
                      print(f"Warn: Error getting book levels for FV {product}@{timestamp}: {bp_err}")
                 # --- End Level Extraction ---

                 # --- Calculate Fair Value using the function defined for the product ---
                 try:
                     # Ensure the fair value function handles empty OrderDepth objects
                     fair_value = self.fair_price_calc[product](order_depth)
                     if not isinstance(fair_value, (int, float)): # Basic type check
                          # Attempt conversion if possible, otherwise set NaN
                          try: fair_value = float(fair_value)
                          except (ValueError, TypeError): fair_value = np.nan
                 except Exception as fv_err:
                     print(f"Warn: Error calculating fair value {product}@{timestamp}: {fv_err}")
                     fair_value = np.nan # Ensure it's NaN if calc fails

                 # --- Ffill Fair Value if Calculation Failed (Based on Past Mid-Price) ---
                 # This is a fallback, ideally the fair_value function is robust
                 if pd.isna(fair_value):
                      try:
                           # Look back in the *currently updating* activity log for this product's mid_price
                           if timestamp in self.activity_log_output.index:
                                product_activity = self.activity_log_output.loc[
                                    (self.activity_log_output['product'] == product) &
                                    (self.activity_log_output.index < timestamp) # Look *before* current timestamp
                                ]
                                if not product_activity.empty and 'mid_price' in product_activity.columns:
                                     # Forward fill the mid_price column for this product up to the previous timestamp
                                     ffilled_mid = product_activity['mid_price'].ffill()
                                     if not ffilled_mid.empty and pd.notna(ffilled_mid.iloc[-1]):
                                         fair_value = ffilled_mid.iloc[-1]
                                         # print(f"Debug: Ffilled FV for {product}@{timestamp} to {fair_value} from previous mid_price") # Optional debug
                      except Exception as ffill_err:
                           print(f"Warn: Error ffilling FV for {product}@{timestamp}: {ffill_err}")
                           pass # Keep fair_value as NaN if ffill fails
                 # --- End Ffill ---

                 # --- Calculate PNL based on Fair Value ---
                 # Only update PnL if fair value is valid *and* position is non-zero or cash exists
                 if pd.notna(fair_value) and (current_pos != 0 or current_cash != 0):
                     calculated_pnl = current_cash + (current_pos * fair_value)
                     self.pnl[product] = calculated_pnl # Update ongoing PnL cache for next step
                 # else: calculated_pnl remains as last_known_pnl

             # --- Log results to the activity log DataFrame ---
             try:
                # Check if the timestamp exists in the index (it should, as we iterate over them)
                if timestamp in self.activity_log_output.index:
                     # Create a boolean mask for the specific row(s)
                     row_mask_product = (self.activity_log_output.index == timestamp) & (self.activity_log_output['product'] == product)
                     # Check if any rows match (should be exactly one)
                     if row_mask_product.any():
                         # Log PNL (could be newly calculated or the carried-forward value)
                         self.activity_log_output.loc[row_mask_product, 'profit_and_loss'] = calculated_pnl
                         # Log MidPrice derived from the FV book cache
                         self.activity_log_output.loc[row_mask_product, 'mid_price'] = calculated_mid_price

                         # Log all available levels derived from the FV book cache
                         for level in [1, 2, 3]:
                             bp_col, bv_col = f'bid_price_{level}', f'bid_volume_{level}'
                             ap_col, av_col = f'ask_price_{level}', f'ask_volume_{level}'
                             if bp_col in self.activity_log_output.columns:
                                 self.activity_log_output.loc[row_mask_product, bp_col] = logged_bid_p[level]
                             if bv_col in self.activity_log_output.columns:
                                 self.activity_log_output.loc[row_mask_product, bv_col] = logged_bid_v[level]
                             if ap_col in self.activity_log_output.columns:
                                 self.activity_log_output.loc[row_mask_product, ap_col] = logged_ask_p[level]
                             if av_col in self.activity_log_output.columns:
                                 self.activity_log_output.loc[row_mask_product, av_col] = logged_ask_v[level]
                     # else:
                         # print(f"Warn: No matching row found in activity log for {product}@{timestamp}")
                 # else:
                      # print(f"Warn: Timestamp {timestamp} not found in activity log index during PnL logging.")

             except Exception as log_err:
                 print(f"Error logging PNL/Prices {product}@{timestamp}: {log_err}")
                 traceback.print_exc()

             # Append to PnL time series for potential later analysis (using the PNL value determined for this step)
             self.pnl_over_time[product].append((timestamp, calculated_pnl))

             # Accumulate total PNL for this timestamp using the calculated PNL for the product
             if pd.notna(calculated_pnl): # Only add valid PNLs
                total_pnl_at_t += calculated_pnl
             # If PNL was NaN (e.g., FV failed and no prior PNL), it won't be added to total

        self.total_pnl_over_time.append((timestamp, total_pnl_at_t))


    def _finalize_activity_log(self):
        """Prepares the activity log for output by filling NaNs and ordering columns."""
        print("\n--- Starting _finalize_activity_log ---")
        if not hasattr(self, 'activity_log_output') or not isinstance(self.activity_log_output, pd.DataFrame) or self.activity_log_output.empty:
            print("ERROR finalize: activity_log_output missing, not DataFrame, or empty. Final log will be empty.")
            self.final_activity_log = pd.DataFrame()
            return

        print(f"Columns BEFORE finalization: {self.activity_log_output.columns.tolist()}")
        # Work on a copy to store the final version
        self.final_activity_log = self.activity_log_output.copy()
        pnl_col = 'profit_and_loss'
        mid_col = 'mid_price'
        product_col = 'product'

        # --- Fill NaNs ---
        if product_col not in self.final_activity_log.columns:
            print(f"ERROR finalize: '{product_col}' missing. Cannot finalize correctly.")
            self.final_activity_log.index.name = 'timestamp'
            return

        # Fill PnL forward *per product* and then fill remaining NaNs with 0
        if pnl_col in self.final_activity_log.columns:
            print(f"DEBUG finalize: Processing '{pnl_col}'...")
            if not pd.api.types.is_numeric_dtype(self.final_activity_log[pnl_col]):
                self.final_activity_log[pnl_col] = pd.to_numeric(self.final_activity_log[pnl_col], errors='coerce')

            # Assign ffill result directly back to the column
            self.final_activity_log[pnl_col] = self.final_activity_log.groupby(product_col, group_keys=False)[pnl_col].ffill()
            # Assign fillna result directly back to the column (avoiding inplace on a potential view)
            self.final_activity_log[pnl_col] = self.final_activity_log[pnl_col].fillna(0)

            print(f"DEBUG finalize: '{pnl_col}' processing complete. Final NaN count: {self.final_activity_log[pnl_col].isna().sum()}")
        else:
            print(f"Warning finalize: '{pnl_col}' column not found! Adding column with 0s.")
            self.final_activity_log[pnl_col] = 0.0

        # Fill mid_price forward *per product*
        if mid_col in self.final_activity_log.columns:
            if not pd.api.types.is_numeric_dtype(self.final_activity_log[mid_col]):
                 self.final_activity_log[mid_col] = pd.to_numeric(self.final_activity_log[mid_col], errors='coerce')

            # Assign ffill result directly back to the column
            self.final_activity_log[mid_col] = self.final_activity_log.groupby(product_col, group_keys=False)[mid_col].ffill()
            # No fillna(0) for mid_price, leave NaNs

            print(f"DEBUG finalize: '{mid_col}' ffilled. Final NaN count: {self.final_activity_log[mid_col].isna().sum()}")
        # --- Reorder Columns ---
        # Define the desired standard column order
        standard_cols_order = [
            'day', 'timestamp', 'product',
            'bid_price_1', 'bid_volume_1', 'bid_price_2', 'bid_volume_2', 'bid_price_3', 'bid_volume_3',
            'ask_price_1', 'ask_volume_1', 'ask_price_2', 'ask_volume_2', 'ask_price_3', 'ask_volume_3',
            'mid_price', 'profit_and_loss'
        ]
        # Add timestamp to the DataFrame from index if it's not already a column
        if 'timestamp' not in self.final_activity_log.columns:
             self.final_activity_log.reset_index(inplace=True) # Move index to 'timestamp' column

        # Get columns present in the DataFrame
        present_cols = self.final_activity_log.columns.tolist()
        # Filter standard order to only include present columns
        final_ordered_cols = [col for col in standard_cols_order if col in present_cols]
        # Add any other columns present that aren't in the standard list (sorted alphabetically)
        other_cols = sorted([col for col in present_cols if col not in final_ordered_cols])
        final_ordered_cols.extend(other_cols)

        try:
             self.final_activity_log = self.final_activity_log[final_ordered_cols]
        except KeyError as e:
             print(f"ERROR finalize: KeyError during column reordering: {e}. Using available columns.")
             # Fallback: use whatever columns are present
             self.final_activity_log = self.final_activity_log[[col for col in present_cols]]

        # Set the index back to timestamp if it was moved
        if 'timestamp' in self.final_activity_log.columns:
             self.final_activity_log.set_index('timestamp', inplace=True)
        elif self.final_activity_log.index.name != 'timestamp':
             self.final_activity_log.index.name = 'timestamp' # Ensure index has name


        print(f"Columns AFTER finalization & reorder: {self.final_activity_log.columns.tolist()}")
        print("--- Finished _finalize_activity_log ---")


    def _generate_output(self):
        """Generates the final combined output string."""
        output_stream = io.StringIO()
        print("\n--- Generating final output string ---")

        # --- Sandbox Logs ---
        output_stream.write("Sandbox logs:\n")
        # Use safe json dumping for potentially complex log structures
        for i, log_entry in enumerate(self.sandbox_logs_capture):
            try:
                # Ensure basic types are serializable
                safe_entry = {
                    "sandboxLog": str(log_entry.get("sandboxLog", "")),
                    "lambdaLog": str(log_entry.get("lambdaLog", "")),
                    "timestamp": int(log_entry.get("timestamp", -1))
                }
                json.dump(safe_entry, output_stream)
                output_stream.write("\n")
            except Exception as e:
                print(f"Error dumping sandbox log entry {i} (ts={log_entry.get('timestamp', 'N/A')}): {e}")
                output_stream.write(f'{{"error": "sandbox log dump failed", "index": {i}, "timestamp": {log_entry.get("timestamp", -1)}}}\n')
        output_stream.write("\n\n") # Add blank lines for separation

        # --- Activities Log ---
        output_stream.write("Activities log:\n")
        # Define the header expected by the platform parser
        activity_header = "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss\n"
        output_stream.write(activity_header) # Write header first

        if hasattr(self, 'final_activity_log') and isinstance(self.final_activity_log, pd.DataFrame) and not self.final_activity_log.empty:
            try:
                # Prepare DataFrame for CSV output
                output_df = self.final_activity_log.copy()
                # Reset index to get timestamp as a column if it's the index
                if output_df.index.name == 'timestamp':
                     output_df.reset_index(inplace=True)

                # Select only the columns present in the header (in that order)
                header_cols = [col for col in activity_header.strip().split(';') if col]
                # Create missing columns with default value (e.g., NaN or empty string) if needed
                for col in header_cols:
                    if col not in output_df.columns:
                        # Determine appropriate default based on column name
                        if 'price' in col or 'profit' in col: default_val = np.nan
                        elif 'volume' in col: default_val = np.nan # Or perhaps 0? Check platform requirements
                        elif col == 'day': default_val = -1 # Or np.nan?
                        else: default_val = ''
                        output_df[col] = default_val
                        print(f"Debug generate: Added missing activity column '{col}' with default.")

                # Select and reorder columns according to the header
                output_df = output_df[header_cols]

                # Convert to CSV, ensuring proper formatting
                # Use float_format for controlled precision, handle NaNs appropriately
                output_df.to_csv(output_stream, sep=';', index=False, header=False, float_format='%.2f', na_rep='') # Use empty string for NaN

            except Exception as e:
                print(f"ERROR generate: Failed to write final activity log to output stream: {e}")
                traceback.print_exc()
                # Write error marker to output if CSV fails
                output_stream.write(f"ERROR writing activity log data;{e};\n")
        else:
             print("Warning generate: Final activity log missing or empty. Only header written.")
             # Header already written above

        output_stream.write("\n\n\n") # Add more blank lines

        # --- Trade History ---
        output_stream.write("Trade History:\n")
        try:
            safe_trades = []
            # Sort trades by timestamp primarily, then maybe price/symbol as secondary for determinism
            sorted_trades = sorted(self.all_trades_log_output, key=lambda x: x.get('timestamp', 0))

            for i, trade_dict in enumerate(sorted_trades):
                 try:
                      # Ensure correct types and format for JSON
                      safe_trade = {
                          "timestamp": int(trade_dict.get('timestamp', 0)),
                          "buyer": str(trade_dict.get('buyer', '') or ""), # Ensure string, handle None
                          "seller": str(trade_dict.get('seller', '') or ""),# Ensure string, handle None
                          "symbol": str(trade_dict.get('symbol', '')),
                          "currency": str(trade_dict.get('currency', 'SEASHELLS')),
                          "price": float(trade_dict.get('price', 0.0)),
                          "quantity": int(trade_dict.get('quantity', 0))
                      }
                      safe_trades.append(safe_trade)
                 except Exception as dict_e:
                      print(f"Error converting trade dict {i} (ts={trade_dict.get('timestamp', 'N/A')}): {dict_e}")
                      safe_trades.append({"error": "trade dict conversion failed", "index": i, "original_data": str(trade_dict)}) # Include original if possible

            # Dump the list of trade dictionaries as a JSON array
            json.dump(safe_trades, output_stream, indent=None, separators=(',', ':')) # Compact JSON format

        except Exception as e:
            print(f"Error dumping trade history: {e}")
            traceback.print_exc()
            output_stream.write('{"error": "trade dump failed"}\n') # Write JSON error object


        # --- Finalize Output ---
        self.output = output_stream.getvalue()
        print("--- Final output string generated. ---")
        output_stream.close()


    def _trade_to_dict(self, trade: Trade) -> Dict[str, Any]:
        """Converts a Trade object to a dictionary suitable for logging."""
        return {
            "timestamp": trade.timestamp,
            "buyer": trade.buyer or "", # Replace None with empty string
            "seller": trade.seller or "", # Replace None with empty string
            "symbol": trade.symbol,
            "currency": "SEASHELLS", # Assuming constant currency
            "price": trade.price,
            "quantity": trade.quantity
        }

# --- END OF FILE backtester.py ---