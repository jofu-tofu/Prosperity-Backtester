# --- START OF FILE app.py ---
import io
import os
import time
import traceback
import hashlib # For seeding per product

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

# Import backtester components (assuming previous robust imports)
try:
    from backtester import backtester as bt_module
    from backtester import util, constants
except ImportError:
    st.error("Fatal Error: Could not import backtester modules. Check file structure and imports.")
    st.stop()

try:
    from backtester import permutation_utils as perm_utils
except ImportError:
    try:
        import permutation_utils as perm_utils
    except ImportError:
        perm_utils = None # Permutations disabled

from typing import List, Dict, Tuple, Optional, Any

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Prosperity Backtester")

# --- Constants & Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
TRADER_DIR_ABS = os.path.join(script_dir, "backtester", "traders")
DATA_DIR_ABS = os.path.join(script_dir, "data")
OUTPUT_DIR_ABS = os.path.join(script_dir, "output")
os.makedirs(TRADER_DIR_ABS, exist_ok=True)
os.makedirs(DATA_DIR_ABS, exist_ok=True)
os.makedirs(OUTPUT_DIR_ABS, exist_ok=True)

# --- Helper Functions ---
@st.cache_data(max_entries=5)
def parse_backtest_output(output_string: str) -> Tuple[List[Dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Parses the full backtester output string using the util function."""
    if not output_string:
        return [], pd.DataFrame(), pd.DataFrame()
    print(f"Parsing backtest output ({len(output_string)} bytes)...")
    sb_logs, market_df, trades_df = [], pd.DataFrame(), pd.DataFrame() # Initialize
    try:
        output_io = io.StringIO(output_string)
        # Use the robust parser from util
        sb_logs, market_df, trades_df = util._parse_data(output_io)

        print(f"DEBUG PARSE: util._parse_data results -> market_df type: {type(market_df)}, empty: {market_df.empty if market_df is not None else 'N/A'}") # Add print

        # --- Post-Parsing Type/Index Checks ---
        if market_df is not None and not market_df.empty:
            print("DEBUG PARSE: Processing non-empty market_df...") # Add print
            # Ensure timestamp index is integer
            if not pd.api.types.is_integer_dtype(market_df.index):
                 market_df.index = pd.to_numeric(market_df.index, errors='coerce').dropna().astype(int)
                 print(f"DEBUG PARSE: Market DF index converted. Type: {market_df.index.dtype}") # Add print
            # Ensure numeric columns are numeric
            for col in market_df.columns:
                if ('price' in col or 'volume' in col or 'profit' in col):
                     if not pd.api.types.is_numeric_dtype(market_df[col]):
                        market_df[col] = pd.to_numeric(market_df[col], errors='coerce')
            # Ensure product is string
            if 'product' in market_df.columns:
                 market_df['product'] = market_df['product'].astype(str)
            print("DEBUG PARSE: Market DF type conversions done.") # Add print

        if trades_df is not None and not trades_df.empty:
             print("DEBUG PARSE: Processing non-empty trades_df...") # Add print
             if not pd.api.types.is_integer_dtype(trades_df.index):
                  trades_df.index = pd.to_numeric(trades_df.index, errors='coerce').dropna().astype(int)
             for col in ['price', 'quantity']:
                 if col in trades_df.columns:
                     if not pd.api.types.is_numeric_dtype(trades_df[col]):
                         trades_df[col] = pd.to_numeric(trades_df[col], errors='coerce')
             for col in ['symbol', 'buyer', 'seller']:
                 if col in trades_df.columns:
                     trades_df[col] = trades_df[col].fillna('').astype(str)
             print("DEBUG PARSE: Trades DF type conversions done.") # Add print

        print("Parsing complete (inside try block).")
        # --- Add final check before returning ---
        final_market_df = market_df if market_df is not None else pd.DataFrame()
        final_trades_df = trades_df if trades_df is not None else pd.DataFrame()
        final_sb_logs = sb_logs if sb_logs is not None else []
        print(f"DEBUG PARSE: Returning market_df -> empty: {final_market_df.empty}")
        # --- End final check ---
        return final_sb_logs, final_market_df, final_trades_df

    except Exception as e:
        st.error(f"Error during parsing or type conversion: {e}") # Make error more specific
        print(f"--- Error During Parsing/Conversion ---")
        print(f"Output length: {len(output_string)}")
        traceback.print_exc()
        print(f"--- End Error During Parsing/Conversion ---")
        # Return empty frames on error
        return [], pd.DataFrame(), pd.DataFrame()

@st.cache_data
def load_file_content(filepath: str) -> str:
    """Loads the raw content of any text file."""
    print(f"Loading file content: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"File not found: {filepath}")
        return f"Error: File not found at {filepath}."
    except Exception as e:
        st.error(f"Error reading file {filepath}: {e}")
        return f"Error loading file content from {filepath}: {e}"

# --- MODIFIED run_backtest wrapper signature ---
def run_backtest(
    trader_filename: str,
    data_filenames: List[str],
    time_range: Tuple[int, int],
    bot_behavior: str,
    ignore_limits: bool,
    explicit_book_override_arg: Optional[Dict[int, Dict[str, Any]]] = None, # Renamed arg
    inferred_book_override_arg: Optional[Dict[int, Dict[str, Any]]] = None, # Renamed arg
    ) -> Optional[bt_module.Backtester]:
    """Creates and runs the backtester instance, returning the instance."""
    run_type = "OVERRIDE" if explicit_book_override_arg is not None else "NORMAL"
    print(f"\n--- Running Backtest ({run_type}) ---")
    print(f"Trader: {trader_filename}, Logs: {data_filenames}, Time: {time_range}, BotBehavior: {bot_behavior}, IgnoreLimits: {ignore_limits}")
    print(f"Explicit Override Provided: {explicit_book_override_arg is not None}, Inferred Override Provided: {inferred_book_override_arg is not None}") # Log override status

    full_trader_path = os.path.join(TRADER_DIR_ABS, trader_filename)
    full_data_paths = [os.path.join(DATA_DIR_ABS, fname) for fname in data_filenames]
    if not os.path.exists(full_trader_path):
        st.error(f"Trader file not found: {full_trader_path}")
        print(f"Error: Trader file does not exist at {full_trader_path}")
        return None
    if not full_data_paths:
         st.error("No data log files selected.")
         print("Error: No data filenames provided.")
         return None
    # Check if primary data file exists
    if not os.path.exists(full_data_paths[0]):
         st.error(f"Primary data file not found: {full_data_paths[0]}")
         print(f"Error: Primary data file does not exist at {full_data_paths[0]}")
         return None

    try:
        print("Creating Backtester instance...")
        t_start_init = time.time()
        backtester_instance = bt_module.Backtester(trader_fname=trader_filename, data_fnames=data_filenames, timerange=time_range, bot_behavior=bot_behavior, ignore_limits=ignore_limits)
        print(f"Backtester instance created in {time.time()-t_start_init:.2f}s.")

        print(f"Starting simulation run ({run_type})...")
        t_start_run = time.time()
        # --- CORRECTED CALL to backtester_instance.run ---
        backtester_instance.run(
            explicit_book_override=explicit_book_override_arg, # Use new arg name matching method signature
            inferred_book_override=inferred_book_override_arg  # Use new arg name matching method signature
        )
        # --- END CORRECTED CALL ---
        print(f"Simulation run completed in {time.time()-t_start_run:.2f}s.")
        return backtester_instance
    except Exception as e:
        st.error(f"Error during backtest execution: {e}")
        st.code(traceback.format_exc())
        print(f"--- Error During Backtest Execution ---")
        traceback.print_exc()
        print(f"--- End Backtest Execution Error ---")
        return None

def calculate_row_vwap(row: pd.Series) -> float:
    """Calculates VWAP for a single row (designed for pd.DataFrame.apply)."""
    n = 0.0
    d = 0.0
    try:
        for i in range(1, 4):
            bp, bv = row.get(f'bid_price_{i}', np.nan), row.get(f'bid_volume_{i}', np.nan)
            ap, av = row.get(f'ask_price_{i}', np.nan), row.get(f'ask_volume_{i}', np.nan)
            if pd.notna(bp) and pd.notna(bv) and bv > 0:
                n += bp * bv
                d += bv
            if pd.notna(ap) and pd.notna(av) and av > 0:
                n += ap * av
                d += av
        if d > 0:
             return n / d
        else:
             return np.nan
    except Exception:
        return np.nan

# --- Charting Functions ---
def display_pnl_chart(market_df: pd.DataFrame, title_prefix: str):
    st.markdown("##### Profit and Loss (PNL)")
    required_cols = ['product', 'profit_and_loss']
    if market_df.empty:
        st.info(f"PNL Chart: No data ({title_prefix}).")
        return
    pnl_plot_data = market_df.copy()
    if pnl_plot_data.index.name == 'timestamp':
        pnl_plot_data.reset_index(inplace=True)
    elif 'timestamp' not in pnl_plot_data.columns:
        st.warning(f"PNL Chart: 'timestamp' not found ({title_prefix}).")
        return
    if not all(col in pnl_plot_data.columns for col in required_cols):
        st.warning(f"PNL Chart: Missing cols ({title_prefix}). Found: {pnl_plot_data.columns.tolist()}")
        return
    if not pd.api.types.is_numeric_dtype(pnl_plot_data['timestamp']):
        pnl_plot_data['timestamp'] = pd.to_numeric(pnl_plot_data['timestamp'], errors='coerce')
    if not pd.api.types.is_numeric_dtype(pnl_plot_data['profit_and_loss']):
        pnl_plot_data['profit_and_loss'] = pd.to_numeric(pnl_plot_data['profit_and_loss'], errors='coerce')
    pnl_plot_data.dropna(subset=['profit_and_loss', 'product', 'timestamp'], inplace=True)
    if pnl_plot_data.empty:
        st.info(f"PNL Chart: No valid data points ({title_prefix}).")
        return
    try:
        fig_pnl = px.line(pnl_plot_data, x='timestamp', y='profit_and_loss', color='product', title=f"{title_prefix} - PNL")
        fig_pnl.update_layout(hovermode="x unified", margin=dict(l=20,r=20,t=50,b=20), xaxis_title="Timestamp", yaxis_title="PNL", legend_title="Product", height=350)
        fig_pnl.update_traces(hovertemplate="<b>%{fullData.name}</b><br>TS: %{x}<br>PNL: %{y:.2f}<extra></extra>")
        total_pnl = pnl_plot_data.groupby('timestamp')['profit_and_loss'].sum().reset_index()
        if not total_pnl.empty:
            fig_pnl.add_trace(go.Scatter(x=total_pnl['timestamp'], y=total_pnl['profit_and_loss'], mode='lines', name='Overall Total', line=dict(color='rgba(0,0,0,0.6)', width=4, dash='dot'), hovertemplate="<b>Overall</b><br>TS: %{x}<br>Total PNL: %{y:.2f}<extra></extra>"))
        st.plotly_chart(fig_pnl, use_container_width=True)
    except Exception as e:
        st.error(f"PNL Chart Error ({title_prefix}): {e}")
        traceback.print_exc()

def display_position_chart(trades_df: pd.DataFrame, title_prefix: str):
    st.markdown("##### Positions ('SUBMISSION')")
    required_cols = ['symbol', 'price', 'quantity', 'buyer', 'seller']
    if trades_df.empty:
        st.info(f"Position Chart: No data ({title_prefix}).")
        return
    player_trades_base = trades_df.copy()
    if player_trades_base.index.name == 'timestamp':
        player_trades_base.reset_index(inplace=True)
    elif 'timestamp' not in player_trades_base.columns:
        st.warning(f"Position Chart: 'timestamp' not found ({title_prefix}).")
        return
    if not all(col in player_trades_base.columns for col in required_cols):
        st.warning(f"Position Chart: Missing cols ({title_prefix}). Found: {player_trades_base.columns.tolist()}")
        return
    player_trades = player_trades_base[(player_trades_base['buyer'] == "SUBMISSION") | (player_trades_base['seller'] == "SUBMISSION")].copy()
    if player_trades.empty:
        st.info(f"Position Chart: No 'SUBMISSION' trades ({title_prefix}).")
        return
    try:
        player_trades['quantity'] = pd.to_numeric(player_trades['quantity'], errors='coerce').fillna(0).astype(int)
        player_trades['timestamp'] = pd.to_numeric(player_trades['timestamp'], errors='coerce')
        player_trades.dropna(subset=['timestamp'], inplace=True)
        player_trades['symbol'] = player_trades['symbol'].astype(str)
        player_trades['buyer'] = player_trades['buyer'].astype(str)
        player_trades['seller'] = player_trades['seller'].astype(str)
        player_trades['signed_quantity'] = player_trades.apply(lambda r: r['quantity'] if r['buyer']=='SUBMISSION' else -r['quantity'], axis=1)
        player_trades.sort_values(by='timestamp', inplace=True)
        player_trades['position'] = player_trades.groupby('symbol')['signed_quantity'].cumsum()
        if player_trades['position'].isna().all():
            st.warning(f"Position Chart: All NaNs ({title_prefix}).")
            return
        fig_pos = px.line(player_trades, x='timestamp', y='position', color='symbol', title=f"{title_prefix} - Positions")
        fig_pos.update_layout(hovermode="x unified", margin=dict(l=20,r=20,t=50,b=20), xaxis_title="Timestamp", yaxis_title="Position", legend_title="Product", height=300)
        fig_pos.update_traces(hovertemplate="<b>%{fullData.name}</b><br>TS: %{x}<br>Pos: %{y}<extra></extra>")
        st.plotly_chart(fig_pos, use_container_width=True)
    except Exception as e:
        st.error(f"Position Chart Error ({title_prefix}): {e}")
        traceback.print_exc()

def display_norm_price_chart(market_df: pd.DataFrame, title_prefix: str):
    """Displays the normalized price change chart (VWAP or Mid-Price)."""
    st.markdown("##### Market Prices (Normalized Change Since Start)")
    print(f"\n--- DEBUG Norm Chart START ({title_prefix}) ---")
    if market_df.empty:
        st.info(f"Norm Price Chart: No market data available ({title_prefix}).")
        print(f"--- DEBUG Norm Chart END ({title_prefix}) - Empty Input DF ---")
        return

    market_data_base = market_df.copy()
    if market_data_base.index.name == 'timestamp':
        market_data_base.reset_index(inplace=True)
    elif 'timestamp' not in market_data_base.columns:
         st.warning(f"Norm Chart: Ts missing ({title_prefix}).")
         print(f"--- DEBUG Norm Chart END ({title_prefix}) - No Ts ---")
         return
    market_data_base['timestamp'] = pd.to_numeric(market_data_base['timestamp'], errors='coerce')
    market_data_base.dropna(subset=['timestamp'], inplace=True)
    if 'product' not in market_data_base.columns:
        st.warning(f"Norm Chart: Product missing ({title_prefix}).")
        print(f"--- DEBUG Norm Chart END ({title_prefix}) - No Prod ---")
        return
    market_data_base['product'] = market_data_base['product'].astype(str)


    # --- Process Each Product Independently ---
    all_processed_product_data = []
    products_to_plot = market_data_base['product'].unique()
    print(f"  Products to process: {products_to_plot}")
    product_price_sources = {} # Store which price source was used for each product

    for product in products_to_plot:
        print(f"\n  Processing product: {product}")
        product_df = market_data_base[market_data_base['product'] == product].copy()
        if product_df.empty:
            print(f"  Skipping {product}: No data rows.")
            continue

        product_df.sort_values(by='timestamp', inplace=True) # Sort early

        price_col_to_use = None
        vwap_needs_recalc = True # Assume recalc needed unless valid exists

        # Check existing vwap
        if 'vwap' in product_df.columns:
            if pd.api.types.is_numeric_dtype(product_df['vwap']):
                if not product_df['vwap'].isna().all():
                    print(f"    Using existing VWAP for {product}.")
                    vwap_needs_recalc = False
                    price_col_to_use = 'vwap'

        # Attempt recalculation if needed
        if vwap_needs_recalc:
            print(f"    Attempting VWAP recalculation for {product}...")
            try:
                 price_vol_cols=[f'{s}_{t}_{l}' for s in ['bid','ask'] for t in ['price','volume'] for l in [1,2,3]]
                 for col in price_vol_cols:
                      if col in product_df.columns:
                           if not pd.api.types.is_numeric_dtype(product_df[col]):
                                print(f"      Converting {col} to numeric for {product}") # Log conversion
                                product_df.loc[:, col] = pd.to_numeric(product_df[col], errors='coerce') # Use .loc
                 # Apply calculation
                 product_df['vwap_recalculated'] = product_df.apply(calculate_row_vwap, axis=1)
                 # Fill within the product's isolated df
                 product_df['vwap_recalculated'] = product_df['vwap_recalculated'].ffill().bfill()

                 if not product_df['vwap_recalculated'].isna().all():
                      print(f"    Using recalculated VWAP for {product}.")
                      price_col_to_use = 'vwap_recalculated'
                 else:
                      print(f"    WARN: Recalculated VWAP is ALL NaN for {product}.")
            except Exception as e:
                 print(f"    ERROR recalculating VWAP for {product}: {e}")
                 traceback.print_exc()
                 if 'vwap_recalculated' not in product_df.columns: # Ensure column exists even if calc failed
                      product_df['vwap_recalculated'] = np.nan


        # Fallback to Mid-Price for THIS product
        if price_col_to_use is None:
             print(f"    VWAP unavailable for {product}. Falling back to 'mid_price'.")
             if 'mid_price' in product_df.columns:
                  if not pd.api.types.is_numeric_dtype(product_df['mid_price']):
                       product_df.loc[:, 'mid_price'] = pd.to_numeric(product_df['mid_price'], errors='coerce')
                  # Optional: Fill mid-price for the product
                  # product_df['mid_price'] = product_df['mid_price'].ffill().bfill()
                  if not product_df['mid_price'].isna().all():
                       print(f"    Using 'mid_price' for {product}.")
                       price_col_to_use = 'mid_price'
                  else:
                       print(f"    'mid_price' all NaN for {product}.")
             else:
                  print(f"    'mid_price' column not found for {product}.")
        # --- End Price Column Determination ---

        # Store the chosen source for this product
        product_price_sources[product] = price_col_to_use if price_col_to_use else "None"

        # --- Normalize THIS product ---
        if price_col_to_use:
            print(f"    Normalizing {product} using column '{price_col_to_use}'.")
            product_norm_data = product_df[['timestamp', 'product', price_col_to_use]].copy()
            product_norm_data.dropna(subset=[price_col_to_use], inplace=True)

            if not product_norm_data.empty:
                # Already sorted by timestamp
                first_value = product_norm_data[price_col_to_use].iloc[0]
                product_norm_data['price_change'] = product_norm_data[price_col_to_use] - first_value
                print(f"    Normalization complete for {product}. First value: {first_value}. Head of price_change:")
                print(product_norm_data[['timestamp', price_col_to_use, 'price_change']].head())
                # Append the processed data for this product
                all_processed_product_data.append(product_norm_data)
            else:
                print(f"    Skipping normalization for {product}: No valid data after dropping NaNs in '{price_col_to_use}'.")
        else:
            print(f"    Skipping normalization for {product}: No suitable price column found.")

    # --- Combine Processed Data & Plot ---
    if not all_processed_product_data:
        st.warning(f"Norm Price Chart: No products had valid data for normalization ({title_prefix}).")
        print(f"--- DEBUG Norm Chart END ({title_prefix}) - No Data After Processing All Products ---")
        return

    final_plot_data = pd.concat(all_processed_product_data, ignore_index=True)

    if 'price_change' not in final_plot_data.columns:
        st.warning(f"Norm Price Chart: 'price_change' column missing in final data ({title_prefix}).")
        print(f"--- DEBUG Norm Chart END ({title_prefix}) - Final Data Invalid (No price_change) ---")
        return

    if final_plot_data['price_change'].isna().all():
         st.warning(f"Norm Price Chart: Final combined data has no valid 'price_change' ({title_prefix}).")
         print(f"--- DEBUG Norm Chart END ({title_prefix}) - Final Data Invalid (All NaN) ---")
         return

    # Build Title and Plot
    source_summary = ", ".join([f"{prod}: {src.replace('_recalculated','')}" for prod, src in product_price_sources.items() if src != "None"])
    y_axis_title = f"{title_prefix} - Norm. Change ($)" # Simpler title
    print(f"  Final plot data shape: {final_plot_data.shape}. Plotting...")

    try:
        # Plot using the combined dataframe with 'price_change' column
        fig_price_norm = px.line(final_plot_data, x='timestamp', y='price_change', color='product')

        # Layout and hovertemplate
        min_y, max_y = final_plot_data['price_change'].min(), final_plot_data['price_change'].max()
        pad = (max_y - min_y) * 0.05 + 1 if pd.notna(min_y) and pd.notna(max_y) else 1
        ymin = (min_y - pad) if pd.notna(min_y) else -1
        ymax = (max_y + pad) if pd.notna(max_y) else 1
        fig_price_norm.update_layout(title=y_axis_title, hovermode="x unified", margin=dict(l=20,r=20,t=50,b=20), xaxis_title="Timestamp", yaxis_title="Price Change ($)", legend_title="Product", height=300, yaxis_zeroline=True, yaxis_zerolinecolor='Gray', yaxis_zerolinewidth=1, yaxis_range=[ymin, ymax])
        fig_price_norm.update_traces(hovertemplate=f"<b>%{{fullData.name}}</b><br>TS: %{{x}}<br>Change: %{{y:+.2f}}<extra></extra>") # Simplified hover

        st.plotly_chart(fig_price_norm, use_container_width=True)

        # --- Add Caption Back ---
        if "Permutation" in title_prefix:
             st.caption("ℹ️ Note: Prices permuted independently per product.")
        # Provide info about price sources used below the chart
        st.caption(f"Price source used for normalization: {source_summary}")
        # --- End Caption ---

        print(f"--- DEBUG Norm Chart END ({title_prefix}) - Plotted Successfully ---")

    except Exception as e:
         st.error(f"Error creating Norm Price chart ({title_prefix}): {e}")
         print(f"--- DEBUG Norm Chart END ({title_prefix}) - Plotting Error ---")
         traceback.print_exc()

# --- Adjusted display_fill_dist_chart to accept log viewer mode ---
def display_fill_dist_chart(market_df: pd.DataFrame, trades_df: pd.DataFrame, title_prefix: str, is_permutation_run: bool = False, is_log_viewer_mode: bool = False): # Add log viewer flag
    """Displays the fill distance from VWAP chart."""
    st.markdown("##### Fill Distance from VWAP")
    if market_df.empty or trades_df.empty:
        st.info(f"Fill Dist Chart: Data missing ({title_prefix}).")
        return
    market_data_processed = market_df.copy()
    trades_processed = trades_df.copy()
    if market_data_processed.index.name == 'timestamp':
        market_data_processed.reset_index(inplace=True)
    if trades_processed.index.name == 'timestamp':
        trades_processed.reset_index(inplace=True)
    if 'timestamp' not in market_data_processed.columns or 'timestamp' not in trades_processed.columns:
        st.warning(f"Fill Dist Chart: Timestamp missing ({title_prefix}).")
        return
    market_data_processed['timestamp'] = pd.to_numeric(market_data_processed['timestamp'], errors='coerce').dropna()
    trades_processed['timestamp'] = pd.to_numeric(trades_processed['timestamp'], errors='coerce').dropna()

    # --- Ensure/Calculate VWAP on Market Data ---
    vwap_available = False
    if 'vwap' in market_data_processed.columns:
        if pd.api.types.is_numeric_dtype(market_data_processed['vwap']):
            if not market_data_processed['vwap'].isna().all():
                vwap_available = True
    if not vwap_available:
        try: # Calculate VWAP
            price_vol_cols=[f'{s}_{t}_{l}' for s in ['bid','ask'] for t in ['price','volume'] for l in [1,2,3]]
            for col in price_vol_cols:
                 if col in market_data_processed.columns:
                      if not pd.api.types.is_numeric_dtype(market_data_processed[col]):
                          market_data_processed[col] = pd.to_numeric(market_data_processed[col], errors='coerce')
            market_data_processed['vwap'] = market_data_processed.apply(calculate_row_vwap, axis=1)
            if 'product' in market_data_processed.columns:
                market_data_processed['vwap'] = market_data_processed.groupby('product')['vwap'].ffill().bfill()
            if 'vwap' in market_data_processed.columns:
                if not market_data_processed['vwap'].isna().all():
                    vwap_available = True
        except Exception as e:
            print(f"Error calculating VWAP for Fill Dist ({title_prefix}): {e}")
    if not vwap_available:
        st.warning(f"Fill Dist Chart: VWAP unavailable ({title_prefix}).")
        return

    # --- Process Trades ---
    required_trade_cols = ['timestamp', 'symbol', 'price', 'quantity', 'buyer', 'seller']
    if not all(col in trades_processed.columns for col in required_trade_cols):
        st.warning(f"Fill Dist Chart: Trades DF missing cols ({title_prefix}).")
        return

    # Filter for player trades FIRST
    player_trades = trades_processed[(trades_processed['buyer'] == "SUBMISSION") | (trades_processed['seller'] == "SUBMISSION")].copy()
    if player_trades.empty:
        st.info(f"Fill Dist Chart: No SUBMISSION trades found ({title_prefix}).")
        return

    # Determine fill type (counterparty)
    def get_fill_type_log_viewer(row):
        is_buyer = row['buyer'] == 'SUBMISSION'
        counterparty = row.get('seller' if is_buyer else 'buyer', '') or 'Unknown' # Treat empty string as Unknown
        # Keep original counterparty unless it's empty, SUBMISSION, or specific known non-fills
        if counterparty in ["", "SUBMISSION", "INTERNAL", "CLEARING"] : # Add other known non-fills if needed
             return "Other/Unknown"
        else:
             return counterparty # Return actual counterparty name

    def get_fill_type_backtest(row):
         is_buyer = row['buyer'] == 'SUBMISSION'
         counterparty = row.get('seller' if is_buyer else 'buyer', '')
         # Only keep Explicit or Inferred labels generated by backtester
         return counterparty if counterparty in ['ExplicitBook', 'InferredBot'] else 'Other/Unknown'

    # Apply appropriate function based on mode
    if is_log_viewer_mode:
        print("Applying Log Viewer fill type logic")
        player_trades['fill_type'] = player_trades.apply(get_fill_type_log_viewer, axis=1)
        # Keep trades where fill_type isn't Other/Unknown
        player_trades = player_trades[player_trades['fill_type'] != 'Other/Unknown'].copy()
        print(f"Log Viewer Fill Types Found: {player_trades['fill_type'].unique()}")
    else: # Backtest or Permutation
        print("Applying Backtest fill type logic")
        player_trades['fill_type'] = player_trades.apply(get_fill_type_backtest, axis=1)
        expected_fills = ['ExplicitBook', 'InferredBot']
        player_trades = player_trades[player_trades['fill_type'].isin(expected_fills)].copy()


    if player_trades.empty:
        st.info(f"Fill Dist Chart: No relevant trades remain after filtering ({title_prefix}).")
        return

    # Ensure numeric types for price/quantity
    player_trades['price'] = pd.to_numeric(player_trades['price'], errors='coerce')
    player_trades['quantity'] = pd.to_numeric(player_trades['quantity'], errors='coerce')
    player_trades.dropna(subset=['price', 'quantity', 'timestamp', 'symbol'], inplace=True)
    if player_trades.empty:
        st.info(f"Fill Dist Chart: No valid player trades ({title_prefix}).")
        return

    # --- Merge Trades with Market VWAP ---
    vwap_data_for_merge = market_data_processed[['timestamp', 'product', 'vwap']].dropna().copy()
    if 'product' not in vwap_data_for_merge.columns:
         st.warning(f"Fill Dist Chart: 'product' missing in VWAP data ({title_prefix}).")
         return
    merged_trades = pd.merge(player_trades, vwap_data_for_merge, left_on=['timestamp', 'symbol'], right_on=['timestamp', 'product'], how='left')
    merged_trades.dropna(subset=['vwap', 'price'], inplace=True)
    if merged_trades.empty:
        st.info(f"Fill Dist Chart: No trades after VWAP merge ({title_prefix}).")
        return

    # --- Calculate Distance & Aggregate ---
    try:
        merged_trades['distance'] = (merged_trades['price'] - merged_trades['vwap']).abs()
        max_b = 5
        labels = {i: f"${i:.2f}-${i+1-0.01:.2f}" for i in range(max_b)}
        labels[max_b] = f"${max_b:.2f}+"
        merged_trades['dist_bucket'] = np.floor(merged_trades['distance']).astype(int).clip(upper=max_b)
        merged_trades['dist_label'] = merged_trades['dist_bucket'].map(labels)
        label_order = [labels[i] for i in range(max_b + 1)]
        agg_cols = ['dist_label', 'fill_type', 'symbol']
        volume_summary = merged_trades.groupby(agg_cols)['quantity'].sum().reset_index()
        volume_summary.rename(columns={'quantity':'total_volume', 'symbol': 'product'}, inplace=True)
        overall_summary = volume_summary.groupby(['dist_label', 'fill_type'])['total_volume'].sum().reset_index()
        overall_summary['product'] = 'Overall'
        combined_summary = pd.concat([overall_summary, volume_summary], ignore_index=True)
        prod_names = sorted(merged_trades['symbol'].unique())
        display_order = ['Overall'] + prod_names
        if combined_summary.empty:
            st.info(f"Fill Dist Chart: No volume data ({title_prefix}).")
            return

        # --- Create Chart ---
        unique_fill_types = combined_summary['fill_type'].unique()
        print(f"Fill types for plot: {unique_fill_types}") # Debug
        color_map = {ftype: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, ftype in enumerate(unique_fill_types)}
        # Standardize colors for known types
        if "ExplicitBook" in color_map: color_map["ExplicitBook"] = px.colors.qualitative.Plotly[0]
        if "InferredBot" in color_map: color_map["InferredBot"] = px.colors.qualitative.Plotly[1]
        if "Other/Unknown" in color_map: color_map["Other/Unknown"] = "grey"

        fig_fill = px.bar(combined_summary, x='dist_label', y='total_volume', color='fill_type', facet_col='product', barmode='group', title=f"{title_prefix} - Fill Volume by Distance from VWAP", labels={'dist_label':'Distance ($)', 'total_volume':'Volume', 'fill_type':'Source', 'product':'Group'}, category_orders={"dist_label": label_order, "product": display_order}, color_discrete_map=color_map)
        fig_fill.update_layout(margin=dict(l=20,r=20,t=50,b=20), height=400)
        fig_fill.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig_fill.update_yaxes(rangemode='tozero')
        st.plotly_chart(fig_fill, use_container_width=True)
    except Exception as e:
        st.error(f"Fill Dist Chart Error ({title_prefix}): {e}")
        traceback.print_exc()


# In app.py

def display_inferred_liquidity_chart(
    market_df: pd.DataFrame,
    inferred_cache: Dict[int, Dict[str, Any]],
    explicit_cache: Dict[int, Dict[str, Any]], # Add explicit cache argument
    title_prefix: str
    ):
    """Displays BOTH inferred and explicit liquidity volume and relative price vs VWAP."""
    # Update title to reflect both books
    st.markdown("##### Explicit & Inferred Liquidity Books (Relative to VWAP)")
    print(f"\n--- DEBUG Combined Book Chart START ({title_prefix}) ---")

    # Initial checks for market_df (needed for VWAP)
    if market_df.empty: st.info(f"Combined Book Chart: No market data ({title_prefix})."); print(f"--- END Combined: market_df empty ---"); return
    # Check if *at least one* cache has data
    if not inferred_cache and not explicit_cache: st.info(f"Combined Book Chart: Both inferred and explicit caches are empty ({title_prefix})."); print(f"--- END Combined: Both caches empty ---"); return

    # --- Prepare Data & Ensure/Calculate VWAP ---
    # (VWAP Preparation logic remains the same as before)
    market_data_processed = market_df.copy()
    # ... (Reset index, check timestamp/product cols) ...
    if market_data_processed.index.name == 'timestamp': market_data_processed.reset_index(inplace=True)
    elif 'timestamp' not in market_data_processed.columns: st.warning(f"Combined Chart: Ts missing ({title_prefix})."); print(f"--- END Combined: No Ts ---"); return
    market_data_processed['timestamp'] = pd.to_numeric(market_data_processed['timestamp'], errors='coerce'); market_data_processed.dropna(subset=['timestamp'], inplace=True)
    if 'product' not in market_data_processed.columns: st.warning(f"Combined Chart: Product missing ({title_prefix})."); print(f"--- END Combined: No Prod ---"); return
    market_data_processed['product'] = market_data_processed['product'].astype(str)
    vwap_available = False
    if 'vwap' in market_data_processed.columns and pd.api.types.is_numeric_dtype(market_data_processed['vwap']) and not market_data_processed['vwap'].isna().all(): vwap_available = True
    else:
        try: # Calculate VWAP
             price_vol_cols=[f'{s}_{t}_{l}' for s in ['bid','ask'] for t in ['price','volume'] for l in [1,2,3]]
             for col in price_vol_cols:
                  if col in market_data_processed.columns and not pd.api.types.is_numeric_dtype(market_data_processed[col]): market_data_processed[col] = pd.to_numeric(market_data_processed[col], errors='coerce')
             market_data_processed['vwap'] = market_data_processed.apply(calculate_row_vwap, axis=1).groupby(market_data_processed['product']).ffill().bfill() # Group by product during fill
             if 'vwap' in market_data_processed.columns and not market_data_processed['vwap'].isna().all(): vwap_available = True
        except Exception as e: print(f"Error calculating VWAP for Combined Chart ({title_prefix}): {e}")
    if not vwap_available: st.warning(f"Combined Chart: VWAP unavailable ({title_prefix})."); print(f"--- END Combined: VWAP unavailable ---"); return
    # --- End VWAP Prep ---

    processed_data = []
    processed_timestamps_inferred = 0
    processed_timestamps_explicit = 0
    skipped_timestamps_vwap = 0
    print('DEBUG: Explicit Cache:', explicit_cache)
    # Combine timestamps from both caches to iterate through all relevant times
    all_timestamps = sorted(list(set(inferred_cache.keys()) | set(explicit_cache.keys())))
    print(f"  Processing {len(all_timestamps)} unique timestamps from both caches.")

    for timestamp in all_timestamps:
        # Get corresponding VWAPs from market_df for this timestamp
        vwaps_at_ts = market_data_processed[market_data_processed['timestamp'] == timestamp].set_index('product')['vwap']
        if vwaps_at_ts.empty:
            skipped_timestamps_vwap +=1
            continue # Skip timestamp if no VWAP data found

        # --- Process INFERRED Data ---
        if timestamp in inferred_cache:
            processed_timestamps_inferred += 1
            product_depths_inferred = inferred_cache[timestamp]
            for product, inferred_depth in product_depths_inferred.items():
                vwap_price = vwaps_at_ts.get(product)
                if pd.isna(vwap_price): continue
                # Inferred Buys
                if inferred_depth.buy_orders:
                    for price, volume in inferred_depth.buy_orders.items():
                        if volume > 0: processed_data.append({'timestamp': timestamp,'product': product,'relative_price': price - vwap_price,'volume': volume,'type': 'Inferred Buy','original_price': price, 'book': 'Inferred'})
                # Inferred Sells
                if inferred_depth.sell_orders:
                    for price, volume in inferred_depth.sell_orders.items():
                         if volume < 0: processed_data.append({'timestamp': timestamp,'product': product,'relative_price': price - vwap_price,'volume': abs(volume),'type': 'Inferred Sell','original_price': price, 'book': 'Inferred'})

        # --- Process EXPLICIT Data ---
        if timestamp in explicit_cache:
            processed_timestamps_explicit += 1
            product_depths_explicit = explicit_cache[timestamp]
            for product, explicit_depth in product_depths_explicit.items():
                vwap_price = vwaps_at_ts.get(product)
                if pd.isna(vwap_price): continue
                # Explicit Bids
                if explicit_depth.buy_orders:
                    for price, volume in explicit_depth.buy_orders.items():
                         if volume > 0: processed_data.append({'timestamp': timestamp,'product': product,'relative_price': price - vwap_price,'volume': volume,'type': 'Explicit Bid','original_price': price, 'book': 'Explicit'})
                # Explicit Asks
                if explicit_depth.sell_orders:
                    for price, volume in explicit_depth.sell_orders.items():
                         if volume < 0: processed_data.append({'timestamp': timestamp,'product': product,'relative_price': price - vwap_price,'volume': abs(volume),'type': 'Explicit Ask','original_price': price, 'book': 'Explicit'})

    print(f"  Processed {processed_timestamps_explicit} explicit timestamps, {processed_timestamps_inferred} inferred timestamps. Skipped {skipped_timestamps_vwap} due to missing VWAP.")

    if not processed_data:
        st.info(f"Combined Book Chart: No processable liquidity data found ({title_prefix}).")
        print(f"--- DEBUG Combined Chart END ({title_prefix}) - No Data After Processing ---")
        return

    plot_df = pd.DataFrame(processed_data)
    # Ensure correct types for plotting
    plot_df['volume'] = pd.to_numeric(plot_df['volume'], errors='coerce')
    plot_df['relative_price'] = pd.to_numeric(plot_df['relative_price'], errors='coerce')
    plot_df.dropna(subset=['volume', 'relative_price'], inplace=True)

    print(f"  Final plot data shape: {plot_df.shape}")
    if plot_df.empty:
         st.info(f"Combined Book Chart: Data empty after final cleaning ({title_prefix}).")
         print(f"--- DEBUG Combined Chart END ({title_prefix}) - Data empty after clean ---"); return

    # --- Create Plot ---
    try:
        # Define colors & symbols
        color_map = {
            'Inferred Buy': px.colors.qualitative.Plotly[2], # Green
            'Inferred Sell': px.colors.qualitative.Plotly[3], # Red
            'Explicit Bid': px.colors.qualitative.Plotly[0], # Blue
            'Explicit Ask': px.colors.qualitative.Plotly[9]  # Orange/Brown
        }
        symbol_map = { # Use different symbols
            'Inferred Buy': 'circle',
            'Inferred Sell': 'circle',
            'Explicit Bid': 'diamond',
            'Explicit Ask': 'diamond',
        }

        fig = px.scatter(plot_df,
                         x='timestamp',
                         y='relative_price',
                         size='volume',
                         color='type', # Color by combined type
                         symbol='book', # Use different symbols for Explicit vs Inferred
                         # symbol_map=symbol_map, # Apply symbol map - Seems symbol map is not directly supported this way in px.scatter color/symbol combo? Let's use color only for now.
                         facet_col='product',
                         color_discrete_map=color_map,
                         title=f"{title_prefix} - Explicit & Inferred Books vs. VWAP", # Updated Title
                         labels={'relative_price': 'Price Relative to VWAP ($)',
                                 'volume': 'Volume',
                                 'type': 'Order Type', # Combined type label
                                 'product': 'Product',
                                 'book': 'Book Type'}, # Added label for symbol if used
                         hover_data={'timestamp': True, 'product': True, 'relative_price': ':.2f',
                                     'volume': True, 'type': True, 'original_price': True, 'book': True}
                        )

        fig.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=20)) # Increased height slightly
        fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black') # Highlight VWAP line (y=0)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])) # Clean facet titles
        # Make inferred points slightly transparent?
        # fig.for_each_trace(lambda t: t.update(opacity=0.7) if "Inferred" in t.name else ()) # Might need adjustment based on trace names

        st.plotly_chart(fig, use_container_width=True)
        print(f"--- DEBUG Combined Chart END ({title_prefix}) - Plotted Successfully ---")

    except Exception as e:
        st.error(f"Error creating Combined Book chart ({title_prefix}): {e}")
        print(f"--- DEBUG Combined Chart END ({title_prefix}) - Plotting Error ---")
        traceback.print_exc()
# --- Session State & Clear/Rerun Functions ---
# (Initialize, Clear functions remain the same)
def initialize_session_state():
    if 'log_viewer_mode' not in st.session_state: st.session_state.log_viewer_mode = False
    if 'parsed_log_data' not in st.session_state: st.session_state.parsed_log_data = ([], pd.DataFrame(), pd.DataFrame(), "", {}) # Added empty dict for cache
    if 'last_loaded_log_file' not in st.session_state: st.session_state.last_loaded_log_file = None
    if 'last_run_trader_file' not in st.session_state: st.session_state.last_run_trader_file = None
    if 'permutation_results' not in st.session_state: st.session_state.permutation_results = {'original_pnl': None, 'permuted_pnls': [], 'permuted_pnls_raw': [], 'p_value': None}
    if 'run_permutation' not in st.session_state: st.session_state.run_permutation = False
    if 'show_output_log_state' not in st.session_state: st.session_state.show_output_log_state = False
    if 'view_permutation_index' not in st.session_state: st.session_state.view_permutation_index = 0
    if 'viewed_permutation_data' not in st.session_state: st.session_state.viewed_permutation_data = ([], pd.DataFrame(), pd.DataFrame(), "")
    if 'permutation_rerun_args' not in st.session_state: st.session_state.permutation_rerun_args = {}
    if 'view_perm_input' not in st.session_state: st.session_state.view_perm_input = 1
initialize_session_state()

def clear_permutation_results():
    print("Clearing permutation results from session state.")
    st.session_state.permutation_results = {'original_pnl': None, 'permuted_pnls': [], 'permuted_pnls_raw': [], 'p_value': None}
    st.session_state.view_permutation_index = 0
    st.session_state.viewed_permutation_data = ([], pd.DataFrame(), pd.DataFrame(), "")
    st.session_state.permutation_rerun_args = {}
    st.session_state.view_perm_input = 1

def clear_parsed_data():
    print("Clearing parsed data and permutation results from session state.")
    st.session_state.parsed_log_data = ([], pd.DataFrame(), pd.DataFrame(), "", {}) # Clear with 5 elements
    st.session_state.last_loaded_log_file = None
    st.session_state.last_run_trader_file = None
    clear_permutation_results() # Also clear perm results

def rerun_and_view_permutation(perm_index_to_view: int, perm_args: dict):
    """Regenerates and reruns a specific independent permutation index using on_click callback."""
    print(f"\n--- DEBUG: Entering rerun_and_view_permutation for Index: {perm_index_to_view} ---")
    st.session_state.viewed_permutation_data = ([], pd.DataFrame(), pd.DataFrame(), "") # Clear previous view
    st.session_state.view_permutation_index = 0 # Reset active view index

    print("--- DEBUG RERUN: Inspecting received perm_args ---")
    if not perm_args: st.warning("Cannot view: Rerun args empty."); print("DEBUG RERUN: perm_args empty!"); return
    required_keys = ['product_data', 'product_relative_explicit_books', 'product_relative_inferred_books', 'block_size', 'trader_file', 'data_files', 'time_range', 'bot_behavior', 'ignore_limits', 'products']
    missing_keys = [k for k in required_keys if k not in perm_args]
    if missing_keys: st.error(f"Missing rerun args: {missing_keys}"); print(f"DEBUG RERUN: Missing keys: {missing_keys}"); return
    if not isinstance(perm_args.get('product_data'), dict): st.error("Invalid rerun 'product_data'"); print("DEBUG RERUN: Invalid product_data"); return
    max_perms_run = len(st.session_state.permutation_results.get('permuted_pnls_raw', []))
    if not (1 <= perm_index_to_view <= max_perms_run): st.warning(f"Index {perm_index_to_view} out of range (1-{max_perms_run})."); return
    if not perm_utils: st.error("Cannot view: Permutation utils unavailable."); return
    print("--- END DEBUG RERUN: Argument Inspection ---")

    perm_progress = st.progress(0.0, text=f"Regenerating permutation {perm_index_to_view} data...")
    try: # Main Rerun Logic
        products_to_permute = perm_args['products']; print(f"DEBUG RERUN: Products: {products_to_permute}")
        all_permuted_vwaps = {}; all_original_indices_maps = {}; all_timestamps_set = set(); base_seed = perm_index_to_view - 1
        # Regenerate Data Per Product
        for idx, product in enumerate(products_to_permute):
            perm_progress.progress(0.1 + 0.4*(idx/len(products_to_permute)), text=f"Permuting {product}...")
            prod_data = perm_args['product_data'].get(product)
            if not prod_data or not isinstance(prod_data.get('abs_changes'), pd.Series): print(f"Warn Rerun: Invalid data for {product}"); continue
            prod_seed = base_seed + int(hashlib.sha1(product.encode()).hexdigest(), 16); np.random.seed(prod_seed % (2**32 - 1))
            perm_chg, orig_map = perm_utils.block_permutation(prod_data['abs_changes'], perm_args['block_size'])
            recon_vwap = perm_utils.reconstruct_absolute_vwap(prod_data['initial_vwap'], perm_chg)
            if recon_vwap.empty: print(f"Warn Rerun: Recon failed {product}."); continue
            all_permuted_vwaps[product]=recon_vwap; all_original_indices_maps[product]=orig_map; all_timestamps_set.update(recon_vwap.index.unique())
        if not all_permuted_vwaps: raise ValueError("No VWAPs reconstructed.")
        # Generate BOTH Permuted Book Caches
        all_ts_sorted = np.sort(list(all_timestamps_set)); print(f"DEBUG RERUN: Generating books for {len(all_ts_sorted)} timestamps.")
        perm_progress.progress(0.6, text=f"Generating Explicit Books...")
        permuted_explicit_cache = perm_utils.generate_permuted_order_book_cache_independent(products_to_permute, all_ts_sorted, all_permuted_vwaps, all_original_indices_maps, perm_args['product_relative_explicit_books'])
        if not permuted_explicit_cache: raise ValueError("Explicit book cache empty")
        perm_progress.progress(0.7, text=f"Generating Inferred Books...")
        permuted_inferred_cache = perm_utils.generate_permuted_inferred_book_cache_independent(products_to_permute, all_ts_sorted, all_permuted_vwaps, all_original_indices_maps, perm_args['product_relative_inferred_books'])
        if not permuted_inferred_cache: raise ValueError("Inferred book cache empty")
        print(f"DEBUG RERUN: Explicit cache len: {len(permuted_explicit_cache)}, Inferred cache len: {len(permuted_inferred_cache)}")
        # Run Backtest
        perm_progress.progress(0.8, text=f"Running backtest...")
        print("DEBUG RERUN: Calling run_backtest with overrides...")
        view_backtester = run_backtest(
             perm_args['trader_file'], perm_args['data_files'][:1], perm_args['time_range'],
             perm_args['bot_behavior'], perm_args['ignore_limits'],
             explicit_book_override_arg=permuted_explicit_cache, # Pass explicit override
             inferred_book_override_arg=permuted_inferred_cache  # Pass inferred override
        )
        # Parse & Store
        perm_progress.progress(0.95, text=f"Parsing results...")
        if view_backtester and hasattr(view_backtester, 'output') and view_backtester.output:
            print("DEBUG RERUN: Backtest successful, parsing output...")
            s_logs, m_df, t_df = parse_backtest_output(view_backtester.output)
            st.session_state.viewed_permutation_data = (s_logs, m_df, t_df, view_backtester.output)
            st.session_state.view_permutation_index = perm_index_to_view # Set the index *after* success
            st.success(f"Loaded results for permutation {perm_index_to_view}.")
        else: print("DEBUG RERUN: Backtest failed or no output!"); st.error(f"Failed rerun for permutation {perm_index_to_view}.")
    except Exception as e: st.error(f"Error viewing permutation {perm_index_to_view}: {e}"); traceback.print_exc(); print(f"--- ERROR in rerun_and_view_permutation {perm_index_to_view} ---"); traceback.print_exc(); print("--- END ERROR ---")
    finally: perm_progress.empty()

# --- UI Layout (Sidebar) ---
# (Sidebar setup remains the same as previous version)
with st.sidebar:
    st.header("⚙️ Configuration")
    is_viewer_mode = st.toggle("Log Viewer Mode", key="log_viewer_mode", value=False, help="If enabled, parse and display a single log file instead of running a trader.", on_change=clear_parsed_data)
    st.caption("Run simulation with trader." if not is_viewer_mode else "Parse and display a single log file.")
    st.divider()

    selected_trader_fname = None
    selected_data_fnames = []
    selected_log_viewer_file = None
    perm_blocksize = 10
    perm_n_iterations = 99
    time_range_values = (0, constants.MAX_TIMESTAMP)
    bot_behavior = 'lte'
    ignore_limits_checkbox = False
    run_permutation_test = False

    if not is_viewer_mode: # Backtest Mode UI
        st.subheader("1. Select Trader")
        try:
            trader_files = sorted([f for f in os.listdir(TRADER_DIR_ABS) if f.endswith(".py") and f != "__init__.py"])
        except Exception as e:
            st.error(f"Trader dir err: {e}")
            trader_files = []
        if not trader_files:
            st.warning(f"No traders in {TRADER_DIR_ABS}.")
        else:
            selected_trader_fname = st.selectbox("Trader File:", trader_files, key="trader_select", index=0, label_visibility="collapsed")
            st.session_state.last_run_trader_file = selected_trader_fname
            if selected_trader_fname:
                trader_code = load_file_content(os.path.join(TRADER_DIR_ABS, selected_trader_fname))
                with st.expander("View Trader Code"):
                    st.code(trader_code, language="python")
        st.subheader("2. Select Data Log(s)")
        try:
            data_files = sorted([f for f in os.listdir(DATA_DIR_ABS) if f.endswith(".log")])
        except Exception as e:
            st.error(f"Data dir err: {e}")
            data_files = []
        if not data_files:
            st.warning(f"No logs in {DATA_DIR_ABS}.")
        else:
            default_selection = [data_files[0]] if data_files else None
            selected_data_fnames = st.multiselect("Log Files (first primary):", data_files, default=default_selection, key="data_select", label_visibility="collapsed")
        st.subheader("3. Settings")
        col_a, col_b = st.columns(2)
        with col_a:
            bot_behavior = st.selectbox("Bot Matching:", ["none", "lt", "lte"], index=2, key="bot_behavior", help="Rule vs inferred bots.")
        with col_b:
            ignore_limits_checkbox = st.checkbox("Ignore Limits", value=False, key="ignore_limits")
        min_time, max_time = 0, constants.MAX_TIMESTAMP
        if selected_data_fnames:
            try:
                 with open(os.path.join(DATA_DIR_ABS, selected_data_fnames[0]), 'r', encoding='utf-8') as f:
                     _, temp_mkt, _ = util._parse_data(f)
                 if temp_mkt is not None and not temp_mkt.empty and pd.api.types.is_numeric_dtype(temp_mkt.index):
                     min_time, max_time = int(temp_mkt.index.min()), int(temp_mkt.index.max())
            except Exception as e:
                print(f"Warn: Time range fail {selected_data_fnames[0]}: {e}")
        min_time = max(0, min_time)
        max_time = min(constants.MAX_TIMESTAMP, max_time) if max_time > min_time else constants.MAX_TIMESTAMP
        time_range_values = st.slider("Time Range:", min_value=min_time, max_value=max_time, value=(min_time, max_time), step=100, key="timerange_backtest")
        st.divider()
        st.subheader("Permutation Testing")
        perm_utils_available = perm_utils is not None
        if not perm_utils_available:
            st.caption("Permutation testing disabled (utils not found).")
        run_permutation_test = st.checkbox("Enable", key="run_permutation", value=False, on_change=clear_permutation_results, disabled=(not perm_utils_available))
        if run_permutation_test and perm_utils_available:
             total_ts = max(1, (time_range_values[1] - time_range_values[0]) // 100 + 1)
             max_block = max(10, total_ts // 10)
             perm_blocksize = st.number_input("Block Size:", min_value=1, max_value=max(1, total_ts), value=max(1, total_ts // 20), step=10, key="perm_blocksize", help=f"Shuffle block length (Max sugg: ~{max_block})")
             perm_n_iterations = st.number_input("Permutations:", min_value=1, max_value=10000, value=99, step=10, key="perm_n_iterations")
        elif run_permutation_test and not perm_utils_available:
            st.error("Permutation utils failed to import.")
    else: # Log Viewer Mode UI
         st.subheader("Select Log File")
         try:
            data_files = sorted([f for f in os.listdir(DATA_DIR_ABS) if f.endswith(".log")])
         except Exception as e:
            st.error(f"Data dir err: {e}")
            data_files = []
         if not data_files:
            st.warning(f"No logs in {DATA_DIR_ABS}.")
         else:
             selected_log_viewer_file = st.selectbox("Log File:", data_files, key="log_viewer_select", index=0, label_visibility="collapsed")
             if selected_log_viewer_file:
                 selected_data_fnames = [selected_log_viewer_file]
                 st.session_state.last_loaded_log_file = selected_log_viewer_file
         min_time, max_time = 0, constants.MAX_TIMESTAMP
         if selected_log_viewer_file:
              try:
                   with open(os.path.join(DATA_DIR_ABS, selected_log_viewer_file), 'r', encoding='utf-8') as f:
                       _, temp_mkt, _ = util._parse_data(f)
                   if temp_mkt is not None and not temp_mkt.empty and pd.api.types.is_numeric_dtype(temp_mkt.index):
                       min_time, max_time = int(temp_mkt.index.min()), int(temp_mkt.index.max())
              except Exception as e:
                  print(f"Warn: Viewer time range fail {selected_log_viewer_file}: {e}")
         min_time = max(0, min_time)
         max_time = min(constants.MAX_TIMESTAMP, max_time) if max_time > min_time else constants.MAX_TIMESTAMP
         time_range_values = st.slider("Time Range Filter:", min_value=min_time, max_value=max_time, value=(min_time, max_time), step=100, key="timerange_viewer")

    st.divider()
    st.checkbox("Show Raw Output Log", key="show_output_log_state", value=False)
    st.divider()
    run_perm_enabled_and_selected = run_permutation_test and not is_viewer_mode and perm_utils_available
    button_label = "📊 Load Log File" if is_viewer_mode else ("🧪 Run & Permute" if run_perm_enabled_and_selected else "🚀 Run Backtest")
    button_disabled = (is_viewer_mode and not selected_log_viewer_file) or (not is_viewer_mode and (not selected_trader_fname or not selected_data_fnames))
    run_button_pressed = st.button(button_label, use_container_width=True, type="primary", disabled=button_disabled)


# --- Main Area (Results) ---
st.header("📊 Results")
results_area = st.container()

s_logs_disp, m_df_disp, t_df_disp, raw_output_disp = [], pd.DataFrame(), pd.DataFrame(), ""
display_source_info = "Configure parameters and run."
data_to_display_available = False
inferred_cache_disp_orig = {} # Initialize here

# --- Main Execution Logic ---
if run_button_pressed: # Main execution block when button is pressed
    print("\n--- Run Button Pressed ---")
    clear_parsed_data()
    if is_viewer_mode: # REVERTED Log Viewer Logic
        if selected_log_viewer_file:
            log_file_path = os.path.join(DATA_DIR_ABS, selected_log_viewer_file)
            print(f"--- Loading Log File: {log_file_path} ---")
            with st.spinner(f"Parsing **{selected_log_viewer_file}**..."):
                try:
                    raw_content = load_file_content(log_file_path)
                    if raw_content.startswith("Error:"):
                        st.error(raw_content)
                    else:
                        print("Parsing log file content...")
                        s_logs, m_df, t_df = parse_backtest_output(raw_content)
                        print("Applying time filter...")
                        m_df_f = m_df[(m_df.index >= time_range_values[0]) & (m_df.index <= time_range_values[1])] if m_df is not None else pd.DataFrame()
                        t_df_f = t_df[(t_df.index >= time_range_values[0]) & (t_df.index <= time_range_values[1])] if t_df is not None else pd.DataFrame()
                        s_logs_f = [log for log in s_logs if isinstance(log.get("timestamp"), int) and time_range_values[0] <= log["timestamp"] <= time_range_values[1]]
                        # Store 4 elements: logs, market_df, trades_df, raw_content
                        st.session_state.parsed_log_data = (s_logs_f, m_df_f, t_df_f, raw_content, {}) # Keep length 5 with empty cache
                        st.success(f"Parsed & filtered **{selected_log_viewer_file}**.")
                except Exception as e:
                    st.error(f"Failed load/parse: {e}")
                    st.code(traceback.format_exc())
        else:
            st.warning("Select log file.")
    elif not is_viewer_mode and selected_trader_fname and selected_data_fnames: # Backtest Logic
        run_permutations_now = st.session_state.get('run_permutation', False) and perm_utils is not None
        st.info(f"Running original backtest for **{selected_trader_fname}**...")
        with st.spinner(f"Running original backtest..."): # Run Original Backtest
             backtester_instance = run_backtest(selected_trader_fname, selected_data_fnames, time_range_values, bot_behavior, ignore_limits_checkbox, explicit_book_override_arg=None, inferred_book_override_arg=None) # Correct call
        if not backtester_instance or not hasattr(backtester_instance, 'output') or not backtester_instance.output:
            st.error("Original backtest failed.")
            st.stop()
        print("Parsing original backtest output...")
        s_logs_orig, m_df_orig, t_df_orig = parse_backtest_output(backtester_instance.output)
        # STORE Original Inferred Cache
        inferred_cache_orig_local = {}
        if hasattr(backtester_instance, 'inferred_bot_liquidity_cache'):
             inferred_cache_orig_local = backtester_instance.inferred_bot_liquidity_cache
             print(f"Storing original inferred cache with {len(inferred_cache_orig_local)} timestamps.")
        else: print("Warning: Original backtester instance missing inferred_bot_liquidity_cache.")
        st.session_state.parsed_log_data = (s_logs_orig, m_df_orig, t_df_orig, backtester_instance.output, inferred_cache_orig_local) # Store 5 elements
        # Calculate Original PNL
        final_pnl_original = np.nan
        if m_df_orig is not None and not m_df_orig.empty and 'profit_and_loss' in m_df_orig.columns:
            try:
                pnl_s_orig = pd.to_numeric(m_df_orig['profit_and_loss'], errors='coerce')
                last_ts = pnl_s_orig.index.max() if not pnl_s_orig.empty else None
                if last_ts is not None:
                    vals = pnl_s_orig.loc[last_ts]
                    final_sum_pnl = np.nan
                    if vals is not None: final_sum_pnl = vals.sum() if isinstance(vals, pd.Series) else vals
                    if pd.notna(final_sum_pnl): final_pnl_original = float(final_sum_pnl)
            except Exception as pnl_ex: print(f"Error calc PNL: {pnl_ex}"); final_pnl_original = np.nan
        st.session_state.permutation_results['original_pnl'] = final_pnl_original
        pnl_disp = f"{final_pnl_original:.2f}" if pd.notna(final_pnl_original) else "N/A"
        st.success(f"Original backtest complete. Final Total PNL: **{pnl_disp}**")

        if run_permutations_now: # Run Permutations
            st.info(f"Starting **{perm_n_iterations}** independent permutations (Block Size: {perm_blocksize})...")
            perm_progress = st.progress(0.0, text="Preparing base data...")
            permuted_pnls_list_raw = []
            products_in_run = []
            try: # Prepare Base Data
                with st.spinner("Analyzing market data..."):
                    print("Preparing base data for independent permutations...")
                    if not hasattr(backtester_instance, 'market_data') or backtester_instance.market_data.empty: raise ValueError("Missing 'market_data'.")
                    market_data_orig = backtester_instance.market_data.copy(); products_in_run = backtester_instance.products; all_orig_timestamps = market_data_orig.index.unique()
                    if not pd.api.types.is_numeric_dtype(all_orig_timestamps): all_orig_timestamps = pd.to_numeric(all_orig_timestamps, errors='coerce').dropna().astype(int)
                    all_orig_timestamps = np.sort(all_orig_timestamps)
                    product_data_for_perm={}; product_relative_explicit_books_cache={}; product_relative_inferred_books_cache={}
                    original_inferred_cache = backtester_instance.inferred_bot_liquidity_cache if hasattr(backtester_instance, 'inferred_bot_liquidity_cache') else {}
                    for product in products_in_run: # Loop products for base data calc
                         print(f"-- Proc base: {product}"); product_market_data = market_data_orig[market_data_orig['product'] == product].copy()
                         if product_market_data.empty: print(f"Warn Prep: No data {product}."); continue
                         # Calculate VWAP if needed
                         if 'vwap' not in product_market_data.columns or product_market_data['vwap'].isna().all():
                              print(f"Calculating VWAP for {product}..."); price_vol_cols=[f'{s}_{t}_{l}' for s in ['bid','ask'] for t in ['price','volume'] for l in [1,2,3]]
                              for col in price_vol_cols:
                                   if col in product_market_data.columns:
                                        if not pd.api.types.is_numeric_dtype(product_market_data[col]): product_market_data.loc[:, col] = pd.to_numeric(product_market_data[col], errors='coerce')
                              product_market_data['vwap'] = product_market_data.apply(calculate_row_vwap, axis=1).ffill().bfill()
                         product_vwap_series = product_market_data['vwap'].reindex(all_orig_timestamps).ffill().bfill(); product_vwap_series.dropna(inplace=True)
                         if product_vwap_series.empty: print(f"Warn Prep: VWAP empty {product}."); continue
                         initial_vwap = product_vwap_series.iloc[0];
                         if pd.isna(initial_vwap): print(f"Warn Prep: Init VWAP NaN {product}."); continue
                         abs_changes = perm_utils.get_absolute_vwap_changes(product_vwap_series)
                         rel_explicit_books = perm_utils.calculate_relative_product_books(market_data_orig[market_data_orig['product'] == product], product_vwap_series) # Pass original market data for structure
                         product_original_inferred = {ts: depth for ts, prods in original_inferred_cache.items() if product in prods for depth in [prods[product]]}
                         rel_inferred_books = perm_utils.calculate_relative_inferred_books(product_original_inferred, product_vwap_series)
                         product_data_for_perm[product] = {'initial_vwap': initial_vwap, 'abs_changes': abs_changes}; product_relative_explicit_books_cache[product] = rel_explicit_books; product_relative_inferred_books_cache[product] = rel_inferred_books; print(f"OK: Base data {product}.")
                    if not product_data_for_perm: raise ValueError("No valid product data prepared.")
                    st.session_state.permutation_rerun_args = {'trader_file': selected_trader_fname, 'data_files': selected_data_fnames[:1], 'time_range': time_range_values, 'bot_behavior': bot_behavior, 'ignore_limits': ignore_limits_checkbox, 'products': list(product_data_for_perm.keys()), 'product_data': product_data_for_perm, 'product_relative_explicit_books': product_relative_explicit_books_cache, 'product_relative_inferred_books': product_relative_inferred_books_cache, 'block_size': perm_blocksize}
                    print("--- DEBUG: Perm Rerun Args Prep OK ---");
            except Exception as prep_e: st.error(f"Error preparing data: {prep_e}"); perm_progress.empty(); traceback.print_exc(); st.stop()

            perm_progress.progress(0.0, text="Running permutations..."); products_to_permute = st.session_state.permutation_rerun_args['products']
            for i in range(perm_n_iterations): # Permutation Loop
                perm_idx_seed = i; print(f"\n--- Running Perm {i+1}/{perm_n_iterations} (Seed: {perm_idx_seed}) ---"); perm_backtester = None; current_perm_pnl = np.nan
                try:
                    all_perm_vwaps={}; all_orig_maps={}; all_ts_set=set()
                    for product in products_to_permute: # Generate permuted data per product
                         prod_data = st.session_state.permutation_rerun_args['product_data'][product]
                         prod_seed = perm_idx_seed + int(hashlib.sha1(product.encode()).hexdigest(), 16); np.random.seed(prod_seed % (2**32 - 1))
                         perm_chg, orig_map = perm_utils.block_permutation(prod_data['abs_changes'], perm_blocksize)
                         recon_vwap = perm_utils.reconstruct_absolute_vwap(prod_data['initial_vwap'], perm_chg)
                         if recon_vwap.empty: raise ValueError(f"Recon fail {product} {i+1}")
                         all_perm_vwaps[product]=recon_vwap; all_orig_maps[product]=orig_map; all_ts_set.update(recon_vwap.index.unique())
                    if not all_perm_vwaps: raise ValueError(f"No VWAPs {i+1}")
                    all_ts_sorted = np.sort(list(all_ts_set))
                    print(f"Perm {i+1}: Gen Explicit books..."); perm_explicit_cache = perm_utils.generate_permuted_order_book_cache_independent(products_to_permute, all_ts_sorted, all_perm_vwaps, all_orig_maps, st.session_state.permutation_rerun_args['product_relative_explicit_books'])
                    if not perm_explicit_cache: raise ValueError(f"Explicit cache empty {i+1}")
                    print(f"Perm {i+1}: Gen Inferred books..."); perm_inferred_cache = perm_utils.generate_permuted_inferred_book_cache_independent(products_to_permute, all_ts_sorted, all_perm_vwaps, all_orig_maps, st.session_state.permutation_rerun_args['product_relative_inferred_books'])
                    if not perm_inferred_cache: raise ValueError(f"Inferred cache empty {i+1}")

                    perm_backtester = run_backtest(selected_trader_fname, selected_data_fnames[:1], time_range_values, bot_behavior, ignore_limits_checkbox, explicit_book_override_arg=perm_explicit_cache, inferred_book_override_arg=perm_inferred_cache) # Correct call
                    if perm_backtester and hasattr(perm_backtester, 'output') and perm_backtester.output: # Extract PNL
                        print(f"Perm {i+1}: Backtest OK, extracting PNL...")
                        try:
                            _, m_df_perm, _ = parse_backtest_output(perm_backtester.output)
                            if m_df_perm is not None and not m_df_perm.empty and 'profit_and_loss' in m_df_perm.columns:
                                pnl_s = pd.to_numeric(m_df_perm['profit_and_loss'], errors='coerce')
                                if not pnl_s.isna().all():
                                    last_ts = pnl_s.index.max() if not pnl_s.empty else None
                                    if last_ts is not None:
                                        vals = pnl_s.loc[last_ts]
                                        final_sum_pnl = np.nan
                                        if vals is not None: final_sum_pnl = vals.sum() if isinstance(vals, pd.Series) else vals
                                        if pd.notna(final_sum_pnl): current_perm_pnl = float(final_sum_pnl); print(f"Perm {i+1}: PNL OK: {current_perm_pnl}")
                                        else: print(f"Perm {i+1}: Final PNL sum NaN.")
                                    else: print(f"Perm {i+1}: No last TS.")
                                else: print(f"Perm {i+1}: PNL series all NaN.")
                            else: print(f"Perm {i+1}: Market DF empty/no PNL col.")
                        except Exception as pnl_ex: print(f"Perm {i+1}: PNL Error: {pnl_ex}")
                    else: print(f"Perm {i+1}: Backtest fail/no output.")
                except Exception as loop_e: print(f"ERROR perm loop {i+1}: {loop_e}"); traceback.print_exc(); current_perm_pnl = np.nan
                finally: permuted_pnls_list_raw.append(current_perm_pnl); print(f"Perm {i+1} Recorded PNL: {current_perm_pnl}"); perm_progress.progress((i+1)/perm_n_iterations, text=f"Perm {i+1}/{perm_n_iterations}");
                if perm_backtester: del perm_backtester

            # Calculate p-value and store results
            valid_perm_pnls = [p for p in permuted_pnls_list_raw if pd.notna(p) and isinstance(p, (int, float))]; p_value = np.nan; pnl_orig = st.session_state.permutation_results.get('original_pnl')
            if pd.isna(pnl_orig): print("Warn: Orig PNL NaN."); st.warning("Orig PNL NaN, no p-value.")
            elif valid_perm_pnls: count_ge = sum(p >= pnl_orig for p in valid_perm_pnls); p_value = (count_ge + 1) / (len(valid_perm_pnls) + 1); print(f"P-value: {p_value:.4f} ({count_ge} >= {pnl_orig:.2f} out of {len(valid_perm_pnls)})")
            else: print("Warn: No valid perm PNLs."); st.warning("No valid perm runs, no p-value.")
            st.session_state.permutation_results['permuted_pnls'] = valid_perm_pnls; st.session_state.permutation_results['permuted_pnls_raw'] = permuted_pnls_list_raw; st.session_state.permutation_results['p_value'] = p_value
            perm_progress.empty(); st.success("Permutation testing complete."); print(f"Final valid PNLs: {st.session_state.permutation_results['permuted_pnls']}")

# --- Display Results ---
# Determine which data source to display
active_view_index = st.session_state.get('view_permutation_index', 0)
viewed_perm_data = st.session_state.get('viewed_permutation_data')
parsed_main_data = st.session_state.get('parsed_log_data') # Tuple length 4 (viewer) or 5 (backtest)
is_displaying_perm_view = False
data_to_display_available = False
inferred_cache_disp_orig = {} # Initialize

print("\n--- DEBUG DISPLAY: Checking Data Availability ---")
print(f"  active_view_index: {active_view_index}")
print(f"  viewed_perm_data type: {type(viewed_perm_data)}")
if isinstance(viewed_perm_data, tuple) and len(viewed_perm_data) == 4: print(f"  viewed_perm_data market_df type: {type(viewed_perm_data[1])}, empty: {viewed_perm_data[1].empty if viewed_perm_data[1] is not None else 'N/A'}")
else: print(f"  viewed_perm_data is not a valid tuple.")
print(f"  parsed_main_data type: {type(parsed_main_data)}")
if isinstance(parsed_main_data, tuple) and len(parsed_main_data) in [4,5]: print(f"  parsed_main_data (len={len(parsed_main_data)}) market_df type: {type(parsed_main_data[1])}, empty: {parsed_main_data[1].empty if parsed_main_data[1] is not None else 'N/A'}")
else: print(f"  parsed_main_data is not a valid tuple (expected 4 or 5 elements). Length: {len(parsed_main_data) if isinstance(parsed_main_data, tuple) else 'N/A'}")

# Check Permutation View First
if active_view_index > 0 and viewed_perm_data and isinstance(viewed_perm_data, tuple) and len(viewed_perm_data) == 4 and viewed_perm_data[1] is not None and not viewed_perm_data[1].empty:
    print("DEBUG DISPLAY: Condition met for displaying permutation view.")
    s_logs_disp, m_df_disp, t_df_disp, raw_output_disp = viewed_perm_data
    display_source_info = f"Viewing Results for Permutation Run **#{active_view_index}**"
    data_to_display_available = True
    is_displaying_perm_view = True
# Check Original/Log View Data if not displaying perm view
elif parsed_main_data and isinstance(parsed_main_data, tuple) and len(parsed_main_data) in [4, 5]:
     print(f"DEBUG DISPLAY: Checking parsed_main_data (length {len(parsed_main_data)} tuple)...")
     market_df_from_state = parsed_main_data[1]
     if market_df_from_state is not None and isinstance(market_df_from_state, pd.DataFrame) and not market_df_from_state.empty:
        print("DEBUG DISPLAY: Condition met for displaying original/log view.")
        if len(parsed_main_data) == 5: # Original Backtest Run
            s_logs_disp, m_df_disp, t_df_disp, raw_output_disp, inferred_cache_disp_orig = parsed_main_data
        else: # Length is 4 (Log Viewer)
            s_logs_disp, m_df_disp, t_df_disp, raw_output_disp = parsed_main_data
            inferred_cache_disp_orig = {} # No inferred cache in viewer mode
        source_desc_key = st.session_state.last_loaded_log_file if st.session_state.log_viewer_mode else st.session_state.last_run_trader_file
        source_mode = "Log View" if st.session_state.log_viewer_mode else "Original Backtest"
        display_source_info = f"Viewing Results for {source_mode}: **{source_desc_key or 'N/A'}**"
        data_to_display_available = True
     else: print("DEBUG DISPLAY: Original/Log parsed data's market data check failed.")
else: print(f"DEBUG DISPLAY: Skipped original/log view check. is_displaying_perm_view={is_displaying_perm_view}, parsed_main_data valid tuple(4/5)={isinstance(parsed_main_data, tuple) and len(parsed_main_data) in [4, 5]}")

if not data_to_display_available: # Handle message if no data found
    if run_button_pressed: print("DEBUG DISPLAY: data_to_display_available is False after run button press."); display_source_info = "Run failed or produced no displayable data. Check console logs."
    else: display_source_info = "Configure parameters and run."

# --- Display Area ---
with results_area:
    if not data_to_display_available:
        st.info(display_source_info)
    else:
        st.subheader("Run Summary")
        st.caption(display_source_info)
        is_log_viewer = st.session_state.log_viewer_mode # Check current mode

        perm_results = st.session_state.get('permutation_results', {})
        tabs_list = ["📊 Charts"]
        if not is_log_viewer and not is_displaying_perm_view and perm_results and perm_results.get('permuted_pnls_raw'): # Show Perm tab only on Original run if permutations ran
            tabs_list.append("🧪 Permutation Test")
        if st.session_state.show_output_log_state or s_logs_disp or raw_output_disp:
            tabs_list.append("📜 Logs & Output")
        if not tabs_list: tabs_list = ["Info"] # Fallback
        tab_objects = st.tabs(tabs_list); tab_map = {name: tab for name, tab in zip(tabs_list, tab_objects)}

        if "📊 Charts" in tab_map: # Charts Tab
            with tab_map["📊 Charts"]:
                chart_title_prefix = f"Permutation {active_view_index}" if is_displaying_perm_view else ("Log View" if is_log_viewer else "Original Run")
                st.markdown(f"#### Charts for: {chart_title_prefix}")
                display_pnl_chart(m_df_disp, chart_title_prefix); st.divider()
                display_position_chart(t_df_disp, chart_title_prefix); st.divider()
                display_norm_price_chart(m_df_disp, chart_title_prefix); st.divider()
                display_fill_dist_chart(m_df_disp, t_df_disp, chart_title_prefix, is_permutation_run=is_displaying_perm_view, is_log_viewer_mode=is_log_viewer); st.divider() # Pass log viewer flag
                # Inferred Chart (Only for original ACTUAL backtest run)
                if not is_displaying_perm_view and not is_log_viewer:
                    # Get the ORIGINAL explicit cache from the backtester instance if possible
                    # This assumes backtester_instance is available in this scope from the original run
                    explicit_cache_orig = {}
                    if 'backtester_instance' in locals() and backtester_instance and hasattr(backtester_instance, 'explicit_order_depths_cache'):
                         explicit_cache_orig = backtester_instance.explicit_order_depths_cache
                    elif parsed_main_data and len(parsed_main_data) == 5:
                         # Fallback: Try re-parsing original explicit if instance not available? Less ideal.
                         pass # Or maybe disable chart if explicit not found

                    # inferred_cache_disp_orig was unpacked earlier
                    display_inferred_liquidity_chart(
                        m_df_disp,
                        inferred_cache_disp_orig, # Pass inferred cache
                        explicit_cache_orig,       # Pass explicit cache
                        chart_title_prefix
                        )

        if "🧪 Permutation Test" in tab_map: # Permutation Tab
            with tab_map["🧪 Permutation Test"]:
                st.subheader("Permutation Test Results"); orig_pnl_from_state = perm_results.get('original_pnl'); orig_pnl_display = f"{orig_pnl_from_state:.2f}" if pd.notna(orig_pnl_from_state) else "N/A"; p_val = perm_results.get('p_value'); valid_perm_pnls = perm_results.get('permuted_pnls', []); raw_perm_pnls = perm_results.get('permuted_pnls_raw', [])
                col1, col2 = st.columns(2); col1.metric("Original Final PNL", orig_pnl_display); col2.metric("p-value", f"{p_val:.4f}" if pd.notna(p_val) else "N/A", help="Prob. PNL >= original.")
                if valid_perm_pnls:
                     st.markdown("###### PNL Distribution")
                     try:
                          fig_hist = ff.create_distplot([valid_perm_pnls], ['Permuted PNLs'], show_hist=True, show_rug=False, show_curve=True) # Removed bin_size=None
                          fig_hist.update_layout(title="Distribution of Final PNLs", xaxis_title="Final Total PNL", yaxis_title="Density", height=400, margin=dict(l=20,r=20,t=50,b=20))
                          if pd.notna(orig_pnl_from_state): fig_hist.add_vline(x=orig_pnl_from_state, line_width=2, line_dash="dash", line_color="red", annotation_text="Original PNL", annotation_position="top right")
                          st.plotly_chart(fig_hist, use_container_width=True)
                     except Exception as hist_e: st.error(f"Hist Error: {hist_e}"); st.error(f"Data: {valid_perm_pnls}"); traceback.print_exc()
                else: st.info("No valid perm PNLs to plot.")
                with st.expander(f"View Raw PNLs ({len(raw_perm_pnls)} Runs)"): st.dataframe(pd.Series(raw_perm_pnls, name="PNL", index=range(1, len(raw_perm_pnls) + 1)))
                st.divider(); st.subheader("View Specific Permutation Run")
                rerun_args = st.session_state.get('permutation_rerun_args'); max_perms = len(raw_perm_pnls) if raw_perm_pnls else 0
                if max_perms == 0 or not rerun_args: st.info("Run permutation test first.")
                else:
                    default_view_idx = st.session_state.view_permutation_index if st.session_state.view_permutation_index > 0 else 1
                    view_index_input = st.number_input(f"Select Run (1-{max_perms}):", min_value=1, max_value=max_perms, value=default_view_idx, step=1, key="view_perm_input", help="Load detailed charts/logs.")
                    st.button("Load Selected Permutation Run", key="view_perm_button", on_click=rerun_and_view_permutation, args=(view_index_input, rerun_args), use_container_width=True, disabled=(not rerun_args or max_perms == 0)) # Use on_click

        if "📜 Logs & Output" in tab_map: # Logs Tab
            with tab_map["📜 Logs & Output"]:
                st.subheader("Execution Logs"); logs_to_use = s_logs_disp; output_content_to_use = raw_output_disp; log_source_info = display_source_info
                st.caption(f"Displaying logs for: {log_source_info}")
                if logs_to_use:
                    st.text(f"Showing {len(logs_to_use)} log entries (newest first):")
                    for i, log_entry in enumerate(reversed(logs_to_use)):
                         ts = log_entry.get("timestamp", "N/A"); sbox = log_entry.get("sandboxLog", ""); lamb = log_entry.get("lambdaLog", "")
                         if sbox or lamb:
                             with st.expander(f"Timestamp {ts}", expanded=(i < 5)):
                                 if lamb: st.text("Trader Output:"); st.code(lamb, language=None)
                                 if sbox: st.text("Sandbox Messages:"); st.code(sbox, language=None)
                else: st.info("No execution logs available.")
                if st.session_state.show_output_log_state:
                     st.divider(); st.subheader("Raw Output Log Content")
                     if output_content_to_use:
                         st.text_area("Raw Output:", output_content_to_use, height=400, key="output_log_area")
                         try: # Download Button
                              fname_trader = "log_view"; fname_log = "unknown_log"; fname_suffix = ""
                              if is_displaying_perm_view: fname_trader = st.session_state.permutation_rerun_args.get('trader_file','unk').replace('.py',''); fname_log = st.session_state.permutation_rerun_args.get('data_files',['unk'])[0].replace('.log',''); fname_suffix = f"_perm{st.session_state.view_permutation_index}"
                              elif is_log_viewer: fname_log = st.session_state.last_loaded_log_file.replace('.log','') if st.session_state.last_loaded_log_file else 'unk'
                              else: fname_trader = st.session_state.last_run_trader_file.replace('.py','') if st.session_state.last_run_trader_file else 'unk'; fname_log = selected_data_fnames[0].replace('.log','') if selected_data_fnames else 'unk'; fname_suffix = "_original"
                              download_filename = f"output_{fname_trader}_{fname_log}{fname_suffix}.log"
                              st.download_button(label="📥 Download Output Log", data=output_content_to_use, file_name=download_filename, mime="text/plain", use_container_width=True)
                         except Exception as dl_err: st.warning(f"Download link error: {dl_err}")
                     else: st.warning("No raw output.")

        if "Info" in tab_map: 
            with tab_map["Info"]: st.info("Configure and run.") # Fallback

# --- END OF app.py ---