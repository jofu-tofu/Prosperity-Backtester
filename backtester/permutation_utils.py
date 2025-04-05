# permutation_utils.py (New File or add to util.py)
import pandas as pd
import numpy as np
import copy
from collections import defaultdict
from .datamodel import OrderDepth # Adjust import if needed

def calculate_relative_order_books(market_data_df, vwap_series):
    """
    Calculates order book offsets relative to rounded VWAP.
    Returns a dictionary: {timestamp: {product: {'buys': {offset: vol}, 'sells': {offset: vol}}}}
    """
    print("Calculating relative order books...")
    relative_books = defaultdict(dict)
    # Merge VWAP for easier access, ensuring alignment
    data = market_data_df.copy()
    data['vwap'] = vwap_series
    data.dropna(subset=['vwap'], inplace=True) # Only process where VWAP exists

    for timestamp, group in data.groupby(data.index):
        ts_relative_books = {}
        for _, row in group.iterrows():
            product = row.get('product')
            vwap = row.get('vwap')
            if not product or pd.isna(vwap): continue

            rounded_vwap = round(vwap)
            rel_buys = {}
            rel_sells = {}

            # Bids
            for i in range(1, 4):
                bp, bv = row.get(f'bid_price_{i}'), row.get(f'bid_volume_{i}')
                if pd.notna(bp) and pd.notna(bv) and bv > 0:
                    try: rel_buys[int(bp) - rounded_vwap] = int(bv)
                    except ValueError: pass
            # Asks
            for i in range(1, 4):
                ap, av = row.get(f'ask_price_{i}'), row.get(f'ask_volume_{i}')
                if pd.notna(ap) and pd.notna(av) and av > 0:
                    try: rel_sells[int(ap) - rounded_vwap] = int(av) # Store positive volume
                    except ValueError: pass

            if rel_buys or rel_sells:
                 ts_relative_books[product] = {'buys': rel_buys, 'sells': rel_sells}
        if ts_relative_books:
            relative_books[timestamp] = ts_relative_books
    print(f"Calculated relative books for {len(relative_books)} timestamps.")
    return dict(relative_books)

def generate_permuted_order_book_cache(
    original_timestamps: np.ndarray,
    permuted_vwap_map: dict, # {timestamp: permuted_vwap}
    relative_books_cache: dict # {timestamp: {product: {'buys':{offs:vol}, 'sells':{offs:vol}}}}
    ) -> dict:
    """
    Reconstructs order books based on permuted VWAP and original relative structures.
    Returns a cache: {timestamp: {product: OrderDepth}}
    """
    permuted_book_cache = defaultdict(lambda: defaultdict(OrderDepth))
    for ts in original_timestamps:
        permuted_vwap = permuted_vwap_map.get(ts)
        original_relative_ts_book = relative_books_cache.get(ts)

        if pd.isna(permuted_vwap) or not original_relative_ts_book:
            continue # Cannot reconstruct if VWAP or relative structure is missing

        rounded_permuted_vwap = round(permuted_vwap)

        for product, rel_book in original_relative_ts_book.items():
            new_depth = OrderDepth()
            # Reconstruct bids
            for offset, vol in rel_book.get('buys', {}).items():
                new_depth.buy_orders[rounded_permuted_vwap + offset] = vol
            # Reconstruct asks
            for offset, vol in rel_book.get('sells', {}).items():
                new_depth.sell_orders[rounded_permuted_vwap + offset] = -vol # Store as negative

            if new_depth.buy_orders or new_depth.sell_orders:
                permuted_book_cache[ts][product] = new_depth

    return dict(permuted_book_cache)

def get_log_vwap_changes(vwap_series):
    """Calculate log changes in VWAP, handling NaNs and zeros."""
    log_vwap = np.log(vwap_series.clip(lower=1e-9)) # Clip to avoid log(0)
    log_changes = log_vwap.diff()
    # Fill first NaN (due to diff) with 0? Or handle differently? Let's use 0 for now.
    log_changes = log_changes.fillna(0)
    return log_changes

def block_permutation(series, block_size):
    """Performs block permutation on a pandas Series."""
    n = len(series)
    num_blocks = n // block_size
    if num_blocks == 0: return series # Not enough data for a full block

    # Extract blocks (as numpy array for easier indexing)
    series_values = series.values[:num_blocks * block_size] # Drop trailing partial block
    blocks = np.split(series_values, num_blocks)

    # Permute block indices
    permuted_indices = np.random.permutation(num_blocks)
    permuted_blocks = [blocks[i] for i in permuted_indices]

    # Reconstruct series
    permuted_values = np.concatenate(permuted_blocks)
    # Create a new series with the original index (cropped)
    permuted_series = pd.Series(permuted_values, index=series.index[:num_blocks * block_size])

    # Handle the trailing partial block if needed - simplest is to drop it
    # Or could append it unshuffled, or handle more complexly. Dropping is standard.
    return permuted_series

def reconstruct_log_vwap(initial_log_vwap, permuted_log_changes):
    """Reconstructs log VWAP series from initial value and permuted changes."""
    # Ensure alignment - use the index from permuted_log_changes
    log_vwap = pd.Series(index=permuted_log_changes.index, dtype=float)
    log_vwap.iloc[0] = initial_log_vwap # Start with the initial value
    # Add the permuted changes cumulatively (skip first value as it's the diff from before start)
    # This assumes permuted_log_changes[0] is the change from t(-1) to t(0)
    log_vwap = initial_log_vwap + permuted_log_changes.cumsum()
    return log_vwap

# --- Make sure these functions are importable in app.py ---
# Example: from permutation_utils import ... OR from . import permutation_utils