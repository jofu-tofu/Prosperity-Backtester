# permutation_utils.py
import pandas as pd
import numpy as np
import copy
from collections import defaultdict, Counter
from typing import Dict, Tuple, Any

# Assuming datamodel.py is in the same directory or accessible
try:
    from .datamodel import OrderDepth
except ImportError:
    try:
        from datamodel import OrderDepth # Fallback
    except ImportError:
        print("Warning: permutation_utils could not import OrderDepth.")
        OrderDepth = Any


MAX_REASONABLE_OFFSET = 15 # Limit offset applied from relative books

# --- Per-Product Relative Book Calculation ---
def calculate_relative_product_books(
    product_market_data: pd.DataFrame,
    product_vwap_series: pd.Series
    ) -> Dict[int, Dict[str, Dict[int, int]]]:
    """
    Calculates relative order books (offsets from VWAP) for a SINGLE product.

    Args:
        product_market_data: DataFrame containing market data ONLY for the target product.
        product_vwap_series: Series containing the VWAP for the target product, indexed by timestamp.

    Returns:
        Dictionary {timestamp: {'buys': {offset: vol}, 'sells': {offset: vol}}}
    """
    print(f"Calculating relative books for product...") # Removed product name for brevity
    relative_books = defaultdict(dict)
    # Combine market data with its VWAP series
    data = product_market_data.copy()
    data['vwap'] = product_vwap_series
    # Drop rows where VWAP is missing for this product
    data.dropna(subset=['vwap'], inplace=True)

    # Group by timestamp (already filtered for one product)
    for timestamp, row in data.iterrows():
        vwap = row.get('vwap')
        if pd.isna(vwap): continue

        rounded_vwap = round(vwap)
        rel_buys: Dict[int, int] = {}
        rel_sells: Dict[int, int] = {}

        # Extract relative bids
        for i in range(1, 4):
            bp, bv = row.get(f'bid_price_{i}'), row.get(f'bid_volume_{i}')
            if pd.notna(bp) and pd.notna(bv) and bv > 0:
                try:
                    offset = int(bp) - rounded_vwap
                    rel_buys[offset] = rel_buys.get(offset, 0) + int(bv) # Sum volumes at same offset
                except (ValueError, TypeError): pass

        # Extract relative asks
        for i in range(1, 4):
            ap, av = row.get(f'ask_price_{i}'), row.get(f'ask_volume_{i}')
            if pd.notna(ap) and pd.notna(av) and av > 0:
                try:
                    offset = int(ap) - rounded_vwap
                    rel_sells[offset] = rel_sells.get(offset, 0) + int(av) # Sum volumes at same offset
                except (ValueError, TypeError): pass

        if rel_buys or rel_sells:
            relative_books[timestamp] = {'buys': rel_buys, 'sells': rel_sells}

    print(f"Calculated relative books for {len(relative_books)} timestamps for this product.")
    return dict(relative_books)


# --- Absolute VWAP Changes (Remains the same) ---
def get_absolute_vwap_changes(vwap_series: pd.Series) -> pd.Series:
    """Calculate absolute dollar changes in VWAP for a single series."""
    if vwap_series.empty:
        print("Warning: get_absolute_vwap_changes received empty series.")
        return pd.Series(dtype=float)
    abs_changes = vwap_series.diff()
    abs_changes = abs_changes.fillna(0) # Fill first NaN with 0 change
    # print(f"DEBUG get_absolute_vwap_changes: Calculated {len(abs_changes)} changes.")
    return abs_changes

# --- Block Permutation (Remains the same) ---
def block_permutation(series: pd.Series, block_size: int) -> Tuple[pd.Series, pd.Series]:
    """
    Performs block permutation on a pandas Series.

    Returns:
        Tuple: (permuted_series, original_indices_map)
               original_indices_map maps the index of the permuted_series
               back to the index of the original series where the block came from.
    """
    n = len(series)
    if block_size <= 0 or block_size > n:
         print(f"Warning block_perm: Invalid block_size {block_size} for series length {n}. Returning original.")
         original_indices_map = pd.Series(series.index, index=series.index)
         return series.copy(), original_indices_map

    num_blocks = n // block_size
    if num_blocks == 0:
        print(f"Warn block_perm: No full blocks (len={n}, size={block_size}). Returning original.")
        original_indices_map = pd.Series(series.index, index=series.index)
        return series.copy(), original_indices_map

    # Work only with full blocks
    effective_length = num_blocks * block_size
    series_to_permute = series.iloc[:effective_length]
    original_index = series_to_permute.index
    series_values = series_to_permute.values

    # Split values and indices into blocks
    value_blocks = np.split(series_values, num_blocks)
    index_blocks = np.split(original_index, num_blocks)

    # Permute the order of block indices
    block_indices = np.arange(num_blocks)
    # print(f"DEBUG block_perm: Original block order: {block_indices}")
    permuted_block_indices_order = np.random.permutation(block_indices)
    # print(f"DEBUG block_perm: Permuted block order: {permuted_block_indices_order}")


    # Reconstruct permuted values and the corresponding original indices map
    permuted_values_list = []
    original_indices_in_permuted_order = [] # List to build the map's values

    for original_block_index in permuted_block_indices_order:
        permuted_values_list.append(value_blocks[original_block_index])
        original_indices_in_permuted_order.extend(index_blocks[original_block_index])

    permuted_values = np.concatenate(permuted_values_list)
    new_index = original_index # Use the original index for the permuted series

    permuted_series = pd.Series(permuted_values, index=new_index, name=series.name)

    # Create the map: Index = index of permuted series, Value = original index
    original_indices_map = pd.Series(original_indices_in_permuted_order, index=new_index)

    # Handle tail if any (append non-permuted part)
    if effective_length < n:
        tail_series = series.iloc[effective_length:]
        permuted_series = pd.concat([permuted_series, tail_series])
        # Map tail indices to themselves
        tail_map = pd.Series(tail_series.index, index=tail_series.index)
        original_indices_map = pd.concat([original_indices_map, tail_map])


    # print(f"DEBUG block_perm: Permuted series len={len(permuted_series)}, Map len={len(original_indices_map)}")
    return permuted_series, original_indices_map


# --- Reconstruct Absolute VWAP (Remains the same) ---
def reconstruct_absolute_vwap(initial_vwap: float, permuted_abs_changes: pd.Series) -> pd.Series:
    """Reconstructs VWAP series from initial value and permuted absolute changes."""
    if permuted_abs_changes.empty:
        print("Warning: reconstruct_absolute_vwap received empty changes series.")
        return pd.Series(dtype=float)
    if pd.isna(initial_vwap):
         print("Warning: reconstruct_absolute_vwap received NaN initial_vwap.")
         # Decide handling: return empty, raise error, or start from 0? Let's return empty for now.
         return pd.Series(dtype=float)

    # The value at time t is initial_vwap + sum_of_absolute_changes_up_to_t
    # Ensure the index is sorted before cumsum if it might not be
    if not permuted_abs_changes.index.is_monotonic_increasing:
         permuted_abs_changes = permuted_abs_changes.sort_index()

    reconstructed = initial_vwap + permuted_abs_changes.cumsum()
    reconstructed.name = 'reconstructed_vwap' # Give the series a name
    return reconstructed


# --- MODIFIED: Generate Permuted Books (Independent Products) ---
def generate_permuted_order_book_cache_independent(
    products: list,
    all_timestamps: np.ndarray, # Unified list of all unique timestamps across all product permutations
    all_permuted_vwaps: Dict[str, pd.Series], # {product: Series(permuted_vwap, index=perm_timestamp)}
    all_original_indices_maps: Dict[str, pd.Series], # {product: Series(original_ts, index=perm_timestamp)}
    all_relative_books: Dict[str, Dict[int, Dict[str, Dict[int, int]]]] # {product: {original_ts: {'buys':..., 'sells':...}}}
    ) -> Dict[int, Dict[str, OrderDepth]]:
    """
    Generates a permuted order book cache where each product's price path
    and structure application are handled independently.

    Args:
        products: List of product symbols to generate books for.
        all_timestamps: Sorted array of unique timestamps present in any permuted VWAP series.
        all_permuted_vwaps: Dict mapping product to its independently permuted VWAP Series.
        all_original_indices_maps: Dict mapping product to its index map (perm_ts -> orig_ts).
        all_relative_books: Dict mapping product to its own relative book cache.

    Returns:
        Dictionary {permuted_timestamp: {product: OrderDepth}}
    """
    print(f"Generating permuted books independently for {len(products)} products across {len(all_timestamps)} timestamps (MAX_OFFSET={MAX_REASONABLE_OFFSET})")
    permuted_book_cache: Dict[int, Dict[str, OrderDepth]] = defaultdict(lambda: defaultdict(OrderDepth))

    for ts_perm in all_timestamps:
        # Create books for all products at this permuted timestamp
        for product in products:
            # Get this product's specific permuted VWAP and index map
            product_vwap_series = all_permuted_vwaps.get(product)
            product_orig_map = all_original_indices_maps.get(product)
            product_rel_books = all_relative_books.get(product)

            if product_vwap_series is None or product_orig_map is None or product_rel_books is None:
                # print(f"Warn BookGen: Missing data for product {product} @ perm_ts {ts_perm}. Skipping book generation.")
                continue # Skip if essential data for this product is missing

            # Get the permuted VWAP for *this product* at *this timestamp*
            permuted_vwap = product_vwap_series.get(ts_perm)
            # Get the original timestamp corresponding to this permuted timestamp *for this product*
            original_ts_to_use = product_orig_map.get(ts_perm)

            # Check if we have valid data for this step
            if pd.isna(permuted_vwap) or original_ts_to_use is None:
                # print(f"Warn BookGen: No valid VWAP ({permuted_vwap}) or original_ts ({original_ts_to_use}) for {product} @ perm_ts {ts_perm}.")
                continue

            # Get the relative book structure for *this product* from the *original timestamp*
            original_relative_ts_book = product_rel_books.get(original_ts_to_use)
            if not original_relative_ts_book:
                # print(f"Warn BookGen: No relative book structure found for {product} @ original_ts {original_ts_to_use} (mapped from perm_ts {ts_perm}).")
                continue # Skip if no structure to apply

            # --- Apply relative structure to the product's permuted VWAP ---
            rounded_permuted_vwap = round(permuted_vwap)
            new_depth = OrderDepth()

            # Apply buy offsets
            for offset, vol in original_relative_ts_book.get('buys', {}).items():
                # Apply offset limit if desired
                if abs(offset) <= MAX_REASONABLE_OFFSET:
                    try:
                         price = int(rounded_permuted_vwap + offset)
                         new_depth.buy_orders[price] = new_depth.buy_orders.get(price, 0) + vol
                    except (ValueError, TypeError): pass # Ignore conversion errors

            # Apply sell offsets (remember stored vol is positive, output needs negative)
            for offset, vol in original_relative_ts_book.get('sells', {}).items():
                 if abs(offset) <= MAX_REASONABLE_OFFSET:
                    try:
                         price = int(rounded_permuted_vwap + offset)
                         # Sell orders are stored with negative quantity
                         new_depth.sell_orders[price] = new_depth.sell_orders.get(price, 0) - vol
                    except (ValueError, TypeError): pass

            # Only add if the generated depth has orders
            if new_depth.buy_orders or new_depth.sell_orders:
                permuted_book_cache[ts_perm][product] = new_depth
            # --- End applying structure ---

    num_generated_ts = len(permuted_book_cache)
    print(f"Generated permuted order book cache for {num_generated_ts} timestamps.")
    if num_generated_ts == 0 and len(all_timestamps) > 0:
         print("Warning: No books were generated despite having input timestamps. Check data validity and logic.")

    return dict(permuted_book_cache)
# permutation_utils.py
# ... (other functions remain the same) ...
# In permutation_utils.py

# In permutation_utils.py

# --- REVERTED to simple offset calculation ---
def calculate_relative_inferred_books(
    product_inferred_book_cache: Dict[int, OrderDepth], # Input is MARGINAL inferred book
    product_vwap_series: pd.Series
    ) -> Dict[int, Dict[str, Dict[int, int]]]:
    """
    Calculates relative offsets for a SINGLE product's MARGINAL inferred liquidity cache.
    (Logic is now the same as for explicit books).
    """
    print(f"Calculating relative MARGINAL INFERRED books for product...")
    relative_books = defaultdict(dict)
    common_timestamps = sorted(list(set(product_inferred_book_cache.keys()) & set(product_vwap_series.index)))

    for timestamp in common_timestamps:
        inferred_depth = product_inferred_book_cache.get(timestamp)
        vwap = product_vwap_series.get(timestamp)
        if inferred_depth is None or pd.isna(vwap): continue

        rounded_vwap = round(vwap)
        rel_buys: Dict[int, int] = {}
        rel_sells: Dict[int, int] = {}
        ts_rel_structure = {}

        # Calculate buy offsets directly from marginal book
        if inferred_depth.buy_orders:
            for price, marginal_volume in inferred_depth.buy_orders.items():
                if marginal_volume > 0:
                    try:
                        offset = int(price) - rounded_vwap
                        rel_buys[offset] = rel_buys.get(offset, 0) + int(marginal_volume)
                    except (ValueError, TypeError): pass

        # Calculate sell offsets directly from marginal book
        if inferred_depth.sell_orders:
            for price, marginal_volume in inferred_depth.sell_orders.items():
                 if marginal_volume < 0: # Sells are negative
                    try:
                        offset = int(price) - rounded_vwap
                        # Store positive volume in relative structure
                        rel_sells[offset] = rel_sells.get(offset, 0) + abs(int(marginal_volume))
                    except (ValueError, TypeError): pass

        if rel_buys: ts_rel_structure['buys'] = rel_buys
        if rel_sells: ts_rel_structure['sells'] = rel_sells
        if ts_rel_structure: relative_books[timestamp] = ts_rel_structure

    print(f"Calculated relative MARGINAL INFERRED books for {len(relative_books)} timestamps.")
    return dict(relative_books)
# --- END REVERTED function ---


# This function is almost identical to the explicit one, just operates on different relative books
def generate_permuted_inferred_book_cache_independent(
    products: list,
    all_timestamps: np.ndarray,
    all_permuted_vwaps: Dict[str, pd.Series],
    all_original_indices_maps: Dict[str, pd.Series],
    all_relative_inferred_books: Dict[str, Dict[int, Dict[str, Dict[int, int]]]] # Takes INFERRED relative books
    ) -> Dict[int, Dict[str, OrderDepth]]:
    """
    Generates a permuted INFERRED order book cache where each product's structure
    application is handled independently based on permuted VWAP.
    """
    print(f"Generating permuted INFERRED books independently for {len(products)} products across {len(all_timestamps)} timestamps (MAX_OFFSET={MAX_REASONABLE_OFFSET})")
    permuted_book_cache: Dict[int, Dict[str, OrderDepth]] = defaultdict(lambda: defaultdict(OrderDepth))

    for ts_perm in all_timestamps:
        for product in products:
            # Get this product's specific data
            product_vwap_series = all_permuted_vwaps.get(product)
            product_orig_map = all_original_indices_maps.get(product)
            product_rel_books = all_relative_inferred_books.get(product) # Use INFERRED relative books

            if product_vwap_series is None or product_orig_map is None or product_rel_books is None: continue

            permuted_vwap = product_vwap_series.get(ts_perm)
            original_ts_to_use = product_orig_map.get(ts_perm)
            if pd.isna(permuted_vwap) or original_ts_to_use is None: continue

            original_relative_ts_book = product_rel_books.get(original_ts_to_use)
            if not original_relative_ts_book: continue

            rounded_permuted_vwap = round(permuted_vwap)
            new_depth = OrderDepth()

            # Apply relative inferred bot BUY structure
            for offset, vol in original_relative_ts_book.get('buys', {}).items():
                if abs(offset) <= MAX_REASONABLE_OFFSET:
                    try:
                         price = int(rounded_permuted_vwap + offset)
                         new_depth.buy_orders[price] = new_depth.buy_orders.get(price, 0) + vol
                    except (ValueError, TypeError): pass

            # Apply relative inferred bot SELL structure (stored pos vol, output needs neg vol)
            for offset, vol in original_relative_ts_book.get('sells', {}).items():
                 if abs(offset) <= MAX_REASONABLE_OFFSET:
                    try:
                         price = int(rounded_permuted_vwap + offset)
                         new_depth.sell_orders[price] = new_depth.sell_orders.get(price, 0) - vol # Output negative
                    except (ValueError, TypeError): pass

            if new_depth.buy_orders or new_depth.sell_orders:
                permuted_book_cache[ts_perm][product] = new_depth

    num_generated_ts = len(permuted_book_cache)
    print(f"Generated permuted INFERRED order book cache for {num_generated_ts} timestamps.")
    return dict(permuted_book_cache)