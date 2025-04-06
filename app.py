# --- START OF FILE app.py ---
import io
import os
import time
import traceback
import hashlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

# Import backtester components
# Use try-except for flexibility if structure changes
try:
    from backtester import backtester as bt_module
    from backtester import util, constants
except ImportError:
    print("Could not import from backtester package, trying local imports...")
    try:
        import backtester as bt_module
        import util
        import constants
    except ImportError:
        st.error("Fatal Error: Could not import backtester modules. Check file structure and imports.")
        st.stop() # Stop app execution if core modules fail

# Import permutation utils, handle potential import errors
try:
    from backtester import permutation_utils as perm_utils
except ImportError:
    try:
        import permutation_utils as perm_utils # Check root directory as fallback
    except ImportError:
        st.warning("Optional: permutation_utils.py not found. Permutation testing will be disabled.")
        perm_utils = None

from typing import List, Dict, Tuple, Optional, Any

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Prosperity Backtester")

# --- Constants & Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Assume standard structure: app.py in root, backtester/ folder adjacent
TRADER_DIR_ABS = os.path.join(script_dir, "backtester", "traders")
DATA_DIR_ABS = os.path.join(script_dir, "data")
OUTPUT_DIR_ABS = os.path.join(script_dir, "output")
os.makedirs(TRADER_DIR_ABS, exist_ok=True) # Ensure directories exist
os.makedirs(DATA_DIR_ABS, exist_ok=True)
os.makedirs(OUTPUT_DIR_ABS, exist_ok=True)


# --- Helper Functions ---

# Use Streamlit's caching for parsing output - assumes output string is immutable
@st.cache_data(max_entries=5) # Cache limited number of parsed outputs
def parse_backtest_output(output_string: str) -> Tuple[List[Dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Parses the full backtester output string using the util function."""
    if not output_string:
        print("Warning: parse_backtest_output called with empty string.")
        return [], pd.DataFrame(), pd.DataFrame()
    print(f"Parsing backtest output ({len(output_string)} bytes)...") # Log parsing attempt
    try:
        output_io = io.StringIO(output_string)
        # Use the robust parser from util
        sb_logs, market_df, trades_df = util._parse_data(output_io)

        # --- Post-Parsing Type/Index Checks ---
        if market_df is not None and not market_df.empty:
            # Ensure timestamp index is integer
            if not pd.api.types.is_integer_dtype(market_df.index):
                if pd.api.types.is_float_dtype(market_df.index):
                     print("Parsing: Converting float market_df index to int.")
                     market_df.index = market_df.index.astype(int)
                else:
                     print(f"Parsing: Market DF index is {market_df.index.dtype}, attempting numeric conversion.")
                     market_df.index = pd.to_numeric(market_df.index, errors='coerce')
                     market_df.dropna(axis=0, subset=[market_df.index.name], inplace=True) # Drop rows where conversion failed
                     if not market_df.empty: market_df.index = market_df.index.astype(int)
            # Ensure numeric columns are numeric
            for col in market_df.columns:
                if 'price' in col or 'volume' in col or 'profit' in col:
                    if not pd.api.types.is_numeric_dtype(market_df[col]):
                        market_df[col] = pd.to_numeric(market_df[col], errors='coerce')
            # Ensure product is string
            if 'product' in market_df.columns:
                 market_df['product'] = market_df['product'].astype(str)

        if trades_df is not None and not trades_df.empty:
             # Ensure timestamp index is integer
            if not pd.api.types.is_integer_dtype(trades_df.index):
                 if pd.api.types.is_float_dtype(trades_df.index):
                      print("Parsing: Converting float trades_df index to int.")
                      trades_df.index = trades_df.index.astype(int)
                 else:
                     print(f"Parsing: Trade DF index is {trades_df.index.dtype}, attempting numeric conversion.")
                     trades_df.index = pd.to_numeric(trades_df.index, errors='coerce')
                     trades_df.dropna(axis=0, subset=[trades_df.index.name], inplace=True) # Drop rows where conversion failed
                     if not trades_df.empty: trades_df.index = trades_df.index.astype(int)
             # Ensure price/quantity are numeric
            for col in ['price', 'quantity']:
                 if col in trades_df.columns and not pd.api.types.is_numeric_dtype(trades_df[col]):
                     trades_df[col] = pd.to_numeric(trades_df[col], errors='coerce')
            # Ensure symbol/buyer/seller are strings
            for col in ['symbol', 'buyer', 'seller']:
                 if col in trades_df.columns:
                     # Fill NaN before converting to string to avoid literal 'nan' strings
                     trades_df[col] = trades_df[col].fillna('').astype(str)


        print("Parsing complete.")
        return sb_logs or [], market_df if market_df is not None else pd.DataFrame(), trades_df if trades_df is not None else pd.DataFrame()

    except Exception as e:
        st.error(f"Error parsing backtest output: {e}")
        print(f"--- Error Parsing Backtest Output ---")
        print(f"Output length: {len(output_string)}")
        print(output_string[:1000] + "...") # Print start of output
        print("...")
        print(output_string[-1000:]) # Print end of output
        traceback.print_exc()
        print(f"--- End Error During Parsing ---")
        return [], pd.DataFrame(), pd.DataFrame()

@st.cache_data
def load_file_content(filepath: str) -> str:
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return f.read()
    except Exception as e: return f"Error loading {filepath}: {e}"

# Note: Removed caching from run_backtest to ensure fresh runs & state
def run_backtest(trader_filename: str, data_filenames: List[str], time_range: Tuple[int, int], bot_behavior: str, ignore_limits: bool, order_depth_override: Optional[Dict[int, Dict[str, Any]]] = None, disable_inferred: bool = False) -> Optional[bt_module.Backtester]:
    run_type = "OVERRIDE" if order_depth_override is not None else "NORMAL"
    print(f"\n--- Running Backtest ({run_type}) ---"); # print(f"Trader: {trader_filename}, Logs: {data_filenames}, Time: {time_range}, BotB: {bot_behavior}, IgnoreL: {ignore_limits}, DisableInfer: {disable_inferred}")
    full_trader_path = os.path.join(TRADER_DIR_ABS, trader_filename); full_data_paths = [os.path.join(DATA_DIR_ABS, fname) for fname in data_filenames]
    if not os.path.exists(full_trader_path): st.error(f"Trader file not found: {full_trader_path}"); return None
    if not full_data_paths or not os.path.exists(full_data_paths[0]): st.error(f"Primary data file missing or path error: {full_data_paths}"); return None
    try:
        t_start_init = time.time()
        backtester_instance = bt_module.Backtester(trader_fname=trader_filename, data_fnames=data_filenames, timerange=time_range, bot_behavior=bot_behavior, ignore_limits=ignore_limits)
        print(f"Instance created in {time.time()-t_start_init:.2f}s.")
        t_start_run = time.time()
        backtester_instance.run(order_depth_override_cache=order_depth_override, disable_inferred_book=disable_inferred)
        print(f"Simulation completed in {time.time()-t_start_run:.2f}s.")
        return backtester_instance
    except Exception as e: st.error(f"Error during backtest execution: {e}"); st.code(traceback.format_exc()); traceback.print_exc(); return None


# --- START: Charting Functions ---

def calculate_row_vwap(row: pd.Series) -> float:
    n = 0.0; d = 0.0
    try:
        for i in range(1, 4):
            bp, bv = row.get(f'bid_price_{i}', np.nan), row.get(f'bid_volume_{i}', np.nan)
            ap, av = row.get(f'ask_price_{i}', np.nan), row.get(f'ask_volume_{i}', np.nan)
            if pd.notna(bp) and pd.notna(bv) and bv > 0: n += bp * bv; d += bv
            if pd.notna(ap) and pd.notna(av) and av > 0: n += ap * av; d += av
        return n / d if d > 0 else np.nan
    except Exception: return np.nan
# --- START: Charting Functions ---

# calculate_row_vwap remains the same

def display_pnl_chart(market_df: pd.DataFrame, title_prefix: str):
    """Displays the PNL chart (Product-specific + Total Overlay)."""
    st.markdown("##### Profit and Loss (PNL)")
    required_cols = ['product', 'profit_and_loss'] # Check for columns, index is handled next

    if market_df.empty:
        st.info(f"PNL Chart: No Activity Log data ({title_prefix}).")
        return

    # --- Ensure 'timestamp' is a column ---
    pnl_plot_data = market_df.copy()
    if pnl_plot_data.index.name == 'timestamp':
        pnl_plot_data.reset_index(inplace=True) # Move timestamp index to column
    elif 'timestamp' not in pnl_plot_data.columns:
         st.warning(f"PNL Chart: 'timestamp' index or column not found ({title_prefix}).")
         return
    # --- Timestamp column is now guaranteed (if execution continues) ---

    if not all(col in pnl_plot_data.columns for col in required_cols):
        st.warning(f"PNL Chart: Required columns missing ({required_cols}) in ({title_prefix}). Columns found: {pnl_plot_data.columns.tolist()}")
        return

    # Ensure timestamp is suitable for plotting (numeric/datetime)
    if not pd.api.types.is_numeric_dtype(pnl_plot_data['timestamp']):
        pnl_plot_data['timestamp'] = pd.to_numeric(pnl_plot_data['timestamp'], errors='coerce')

    # Ensure PnL is numeric
    if not pd.api.types.is_numeric_dtype(pnl_plot_data['profit_and_loss']):
        pnl_plot_data['profit_and_loss'] = pd.to_numeric(pnl_plot_data['profit_and_loss'], errors='coerce')

    pnl_plot_data.dropna(subset=['profit_and_loss', 'product', 'timestamp'], inplace=True)

    if pnl_plot_data.empty:
        st.info(f"PNL Chart: No valid PNL data points remain after cleaning ({title_prefix}).")
        return

    try:
        fig_pnl = px.line(pnl_plot_data, x='timestamp', y='profit_and_loss', color='product',
                          title=f"{title_prefix} - Individual Product PNL & Total PNL")
        # ... (rest of the PNL chart function remains the same) ...
        fig_pnl.update_layout(hovermode="x unified", margin=dict(l=20,r=20,t=50,b=20), # Adjusted margins
                              xaxis_title="Timestamp", yaxis_title="PNL", legend_title="Product", height=350)
        fig_pnl.update_traces(hovertemplate="Product: <b>%{fullData.name}</b><br>Timestamp: %{x}<br>PNL: %{y:.2f}<extra></extra>")

        # Calculate and add total PNL trace
        total_pnl_trace_data = pnl_plot_data.groupby('timestamp')['profit_and_loss'].sum().reset_index()
        if not total_pnl_trace_data.empty:
            fig_pnl.add_trace(go.Scatter(x=total_pnl_trace_data['timestamp'], y=total_pnl_trace_data['profit_and_loss'],
                                mode='lines', name='Overall Total PNL',
                                line=dict(color='rgba(0,0,0,0.6)', width=4, dash='dot'),
                                hovertemplate="<b>Overall Total PNL</b><br>Timestamp: %{x}<br>Total PNL: %{y:.2f}<extra></extra>")
            )
        st.plotly_chart(fig_pnl, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating PNL chart ({title_prefix}): {e}")
        traceback.print_exc()


def display_position_chart(trades_df: pd.DataFrame, title_prefix: str):
    """Displays the position chart based on SUBMISSION trades."""
    st.markdown("##### Positions (Based on 'SUBMISSION' Trades)")
    if trades_df.empty:
        st.info(f"Position Chart: No trade history data available ({title_prefix}).")
        return

    required_cols = ['symbol', 'price', 'quantity', 'buyer', 'seller'] # Check columns, index handled next

    # --- Ensure 'timestamp' is a column ---
    player_trades_base = trades_df.copy()
    if player_trades_base.index.name == 'timestamp':
        player_trades_base.reset_index(inplace=True) # Move timestamp index to column
    elif 'timestamp' not in player_trades_base.columns:
         st.warning(f"Position Chart: 'timestamp' index or column not found ({title_prefix}).")
         return
    # --- Timestamp column is now guaranteed ---

    if not all(col in player_trades_base.columns for col in required_cols):
        st.warning(f"Position Chart: Required columns missing ({required_cols}) in ({title_prefix}). Columns found: {player_trades_base.columns.tolist()}")
        return

    player_trades = player_trades_base[
        (player_trades_base['buyer'] == "SUBMISSION") | (player_trades_base['seller'] == "SUBMISSION")
    ].copy() # Filter after ensuring timestamp column exists

    if player_trades.empty:
        st.info(f"Position Chart: No 'SUBMISSION' trades found ({title_prefix}).")
        return

    try:
        # Ensure types before calculation
        player_trades['quantity'] = pd.to_numeric(player_trades['quantity'], errors='coerce').fillna(0).astype(int)
        player_trades['symbol'] = player_trades['symbol'].astype(str)
        player_trades['buyer'] = player_trades['buyer'].astype(str)
        player_trades['seller'] = player_trades['seller'].astype(str)
        player_trades['timestamp'] = pd.to_numeric(player_trades['timestamp'], errors='coerce') # Ensure numeric timestamp
        player_trades.dropna(subset=['timestamp'], inplace=True) # Drop if timestamp invalid

        # Calculate signed quantity based on buyer/seller
        player_trades['signed_quantity'] = player_trades.apply(
            lambda r: r['quantity'] if r['buyer'] == 'SUBMISSION' else -r['quantity'], axis=1
        )

        # Sort by timestamp before calculating cumulative sum
        player_trades.sort_values(by='timestamp', inplace=True)

        # Calculate cumulative position per product
        player_trades['position'] = player_trades.groupby('symbol')['signed_quantity'].cumsum()

        # Check if position calculation resulted in valid data
        if player_trades['position'].isna().all():
            st.warning(f"Position Chart: Position calculation resulted in all NaNs ({title_prefix}).")
            return

        fig_pos = px.line(player_trades, x='timestamp', y='position', color='symbol',
                          title=f"{title_prefix} - Player Positions Over Time")
        # ... (rest of position chart function remains the same) ...
        fig_pos.update_layout(hovermode="x unified", margin=dict(l=20,r=20,t=50,b=20), # Adjusted margins
                              xaxis_title="Timestamp", yaxis_title="Position", legend_title="Product", height=300)
        fig_pos.update_traces(hovertemplate="<b>%{fullData.name}</b><br>Timestamp: %{x}<br>Position: %{y}<extra></extra>")
        st.plotly_chart(fig_pos, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating Position chart ({title_prefix}): {e}")
        traceback.print_exc()


def display_norm_price_chart(market_df: pd.DataFrame, title_prefix: str):
    """Displays the normalized price change chart (VWAP or Mid-Price)."""
    st.markdown("##### Market Prices (Normalized Change Since Start)")
    if market_df.empty:
        st.info(f"Norm Price Chart: No market data available ({title_prefix}).")
        return

    # --- Ensure 'timestamp' is a column ---
    market_data_processed = market_df.copy()
    if market_data_processed.index.name == 'timestamp':
        market_data_processed.reset_index(inplace=True) # Move timestamp index to column
    elif 'timestamp' not in market_data_processed.columns:
         st.warning(f"Norm Price Chart: 'timestamp' index or column not found ({title_prefix}).")
         return
    # --- Timestamp column is now guaranteed ---

    # Ensure timestamp is numeric
    if not pd.api.types.is_numeric_dtype(market_data_processed['timestamp']):
         market_data_processed['timestamp'] = pd.to_numeric(market_data_processed['timestamp'], errors='coerce')
         market_data_processed.dropna(subset=['timestamp'], inplace=True)

    # --- Determine Price Column (VWAP or Mid) ---
    price_col_to_use = None
    y_axis_title = "Price Change ($)"
    # ... (VWAP calculation/checking logic remains the same) ...
    # Check if VWAP needs calculation (robustly check for existence and NaN content)
    vwap_needs_calc = True
    if 'vwap' in market_data_processed.columns:
         if pd.api.types.is_numeric_dtype(market_data_processed['vwap']):
             if not market_data_processed['vwap'].isna().all():
                 print(f"Norm Price Chart: Using existing 'vwap' column ({title_prefix}).")
                 vwap_needs_calc = False
                 price_col_to_use = 'vwap'
                 y_axis_title = f"{title_prefix} - VWAP Change ($)"
         else: # VWAP column exists but isn't numeric
             print(f"Norm Price Chart: 'vwap' column exists but is not numeric ({market_data_processed['vwap'].dtype}), attempting calculation.")
             market_data_processed['vwap'] = pd.to_numeric(market_data_processed['vwap'], errors='coerce')
             if not market_data_processed['vwap'].isna().all():
                  print(f"Norm Price Chart: Successfully coerced 'vwap' to numeric.")
                  vwap_needs_calc = False
                  price_col_to_use = 'vwap'
                  y_axis_title = f"{title_prefix} - VWAP Change ($)"


    if vwap_needs_calc:
        print(f"Norm Price Chart: Calculating VWAP ({title_prefix})...")
        try:
            # Ensure necessary price/vol columns are numeric before applying VWAP calc
            price_vol_cols = [f'{s}_{t}_{l}' for s in ['bid','ask'] for t in ['price','volume'] for l in [1,2,3]]
            for col in price_vol_cols:
                if col in market_data_processed.columns and not pd.api.types.is_numeric_dtype(market_data_processed[col]):
                     market_data_processed[col] = pd.to_numeric(market_data_processed[col], errors='coerce')

            vwap_s = market_data_processed.apply(calculate_row_vwap, axis=1)
            market_data_processed['vwap'] = vwap_s
            # Fill missing VWAPs forward/backward within each product
            if 'product' in market_data_processed.columns:
                market_data_processed['vwap'] = market_data_processed.groupby('product')['vwap'].ffill().bfill()
            else:
                 st.warning("Norm Price Chart: Cannot ffill VWAP without 'product' column.")

            if 'vwap' in market_data_processed.columns and not market_data_processed['vwap'].isna().all():
                print(f"Norm Price Chart: VWAP calculated successfully.")
                price_col_to_use = 'vwap'
                y_axis_title = f"{title_prefix} - VWAP Change ($)"
            else:
                print(f"Norm Price Chart: VWAP calculation failed or resulted in all NaNs.")
        except Exception as e:
            print(f"Error calculating VWAP for Norm Price Chart ({title_prefix}): {e}")
            market_data_processed['vwap'] = np.nan # Ensure column exists but is NaN

    # Fallback to Mid-Price if VWAP failed or wasn't chosen
    if price_col_to_use is None:
        print(f"Norm Price Chart: VWAP unavailable or failed, checking for Mid-Price ({title_prefix}).")
        if 'mid_price' in market_data_processed.columns:
            # Ensure Mid Price is numeric and filled if needed
            if not pd.api.types.is_numeric_dtype(market_data_processed['mid_price']):
                market_data_processed['mid_price'] = pd.to_numeric(market_data_processed['mid_price'], errors='coerce')
            # Optionally fill mid-price (or leave gaps)
            # if 'product' in market_data_processed.columns:
            #     market_data_processed['mid_price'] = market_data_processed.groupby('product')['mid_price'].ffill().bfill()
            if not market_data_processed['mid_price'].isna().all():
                 print(f"Norm Price Chart: Using 'mid_price' column.")
                 price_col_to_use = 'mid_price'
                 y_axis_title = f"{title_prefix} - Mid Price Change ($)"
            else:
                 print(f"Norm Price Chart: 'mid_price' column is all NaN.")
        else:
             print(f"Norm Price Chart: 'mid_price' column not found.")


    # --- Plotting ---
    if price_col_to_use and 'product' in market_data_processed.columns:
        price_data = market_data_processed[['timestamp', 'product', price_col_to_use]].copy()
        price_data.dropna(subset=[price_col_to_use, 'product', 'timestamp'], inplace=True)

        if not price_data.empty:
            # --- Normalization ---
            # Sort is crucial before finding the first value per group
            price_data.sort_values(by=['product', 'timestamp'], inplace=True)

            # Calculate change relative to the *first* price observed for each product
            price_data['price_change'] = price_data.groupby('product')[price_col_to_use].transform(lambda x: x - x.iloc[0])

            if price_data['price_change'].isna().all():
                 st.warning(f"Norm Price Chart: Normalized price calculation resulted in all NaNs ({title_prefix}).")
                 return

            # --- Create Plot ---
            try:
                fig_price_norm = px.line(price_data, x='timestamp', y='price_change', color='product')
                # ... (rest of norm price chart function remains the same) ...
                # Calculate y-axis range dynamically for better visualization
                min_y = price_data['price_change'].min()
                max_y = price_data['price_change'].max()
                y_range_padding = (max_y - min_y) * 0.05 + 1 # Add small padding
                ymin = min_y - y_range_padding if pd.notna(min_y) else -1
                ymax = max_y + y_range_padding if pd.notna(max_y) else 1

                fig_price_norm.update_layout(
                    title=y_axis_title, hovermode="x unified", margin=dict(l=20,r=20,t=50,b=20), # Adjusted margins
                    xaxis_title="Timestamp", yaxis_title="Price Change ($)", legend_title="Product", height=300,
                    yaxis_zeroline=True, yaxis_zerolinecolor='Gray', yaxis_zerolinewidth=1,
                    yaxis_range=[ymin, ymax] # Apply dynamic range
                )
                # Add hover data showing the original price
                fig_price_norm.update_traces(
                    hovertemplate=(f"<b>%{{fullData.name}}</b><br>Timestamp: %{{x}}<br>Change: %{{y:+.2f}}<br>{price_col_to_use.upper()}: %{{customdata[0]:.2f}}<extra></extra>"),
                    customdata=price_data[[price_col_to_use]] # Pass original price value
                )
                st.plotly_chart(fig_price_norm, use_container_width=True)


            except Exception as e:
                 st.error(f"Error creating Norm Price chart ({title_prefix}): {e}")
                 traceback.print_exc()

        else:
            st.info(f"Norm Price Chart: No valid price data ({price_col_to_use}) found after cleaning ({title_prefix}).")
    else:
        st.warning(f"Norm Price Chart: Cannot find suitable price column (VWAP or Mid-Price) or product column ({title_prefix}). Cannot generate chart.")


def display_fill_dist_chart(market_df: pd.DataFrame, trades_df: pd.DataFrame, title_prefix: str, is_permutation_run: bool = False):
    """Displays the fill distance from VWAP chart."""
    st.markdown("##### Fill Distance from VWAP")
    if market_df.empty:
        st.info(f"Fill Dist Chart: Market data needed ({title_prefix}).")
        return
    if trades_df.empty:
        st.info(f"Fill Dist Chart: Trade data needed ({title_prefix}).")
        return

    # --- Ensure 'timestamp' is a column ---
    market_data_processed = market_df.copy()
    trades_processed = trades_df.copy()
    if market_data_processed.index.name == 'timestamp': market_data_processed.reset_index(inplace=True)
    if trades_processed.index.name == 'timestamp': trades_processed.reset_index(inplace=True)
    if 'timestamp' not in market_data_processed.columns or 'timestamp' not in trades_processed.columns:
        st.warning(f"Fill Dist Chart: Timestamp column missing ({title_prefix}).")
        return
    # Ensure numeric timestamp
    market_data_processed['timestamp'] = pd.to_numeric(market_data_processed['timestamp'], errors='coerce')
    trades_processed['timestamp'] = pd.to_numeric(trades_processed['timestamp'], errors='coerce')
    market_data_processed.dropna(subset=['timestamp'], inplace=True)
    trades_processed.dropna(subset=['timestamp'], inplace=True)
    # --- Timestamp column is now guaranteed ---


    # --- Ensure/Calculate VWAP on Market Data ---
    vwap_available = False
    # ... (VWAP calculation/checking logic remains the same) ...
    if 'vwap' in market_data_processed.columns and pd.api.types.is_numeric_dtype(market_data_processed['vwap']) and not market_data_processed['vwap'].isna().all():
        print(f"Fill Dist Chart: Using existing VWAP ({title_prefix}).")
        vwap_available = True
    else:
        print(f"Fill Dist Chart: Calculating VWAP ({title_prefix})...")
        try:
            # Ensure numeric types for calculation
            price_vol_cols = [f'{s}_{t}_{l}' for s in ['bid','ask'] for t in ['price','volume'] for l in [1,2,3]]
            for col in price_vol_cols:
                 if col in market_data_processed.columns and not pd.api.types.is_numeric_dtype(market_data_processed[col]):
                      market_data_processed[col] = pd.to_numeric(market_data_processed[col], errors='coerce')
            vwap_s = market_data_processed.apply(calculate_row_vwap, axis=1)
            market_data_processed['vwap'] = vwap_s
            if 'product' in market_data_processed.columns:
                market_data_processed['vwap'] = market_data_processed.groupby('product')['vwap'].ffill().bfill()

            if 'vwap' in market_data_processed.columns and not market_data_processed['vwap'].isna().all():
                 print(f"Fill Dist Chart: VWAP calculation successful.")
                 vwap_available = True
            else: print(f"Fill Dist Chart: VWAP calculation failed or resulted in all NaNs.")
        except Exception as e: print(f"Error calculating VWAP for Fill Dist ({title_prefix}): {e}")


    if not vwap_available:
        st.warning(f"Fill Dist Chart: VWAP unavailable ({title_prefix}). Cannot generate chart.")
        return

    # --- Process Trades ---
    required_trade_cols = ['timestamp', 'symbol', 'price', 'quantity', 'buyer', 'seller']
    if not all(col in trades_processed.columns for col in required_trade_cols):
        st.warning(f"Fill Dist Chart: Trades DF missing required columns ({title_prefix}).")
        return

    # Filter for player trades
    player_trades = trades_processed[
        (trades_processed['buyer'] == "SUBMISSION") | (trades_processed['seller'] == "SUBMISSION")
    ].copy()
    if player_trades.empty:
        st.info(f"Fill Dist Chart: No SUBMISSION trades found ({title_prefix}).")
        return

    # Determine fill type (counterparty)
    def get_fill_type(row):
        if row['buyer'] == 'SUBMISSION': return row.get('seller', 'Unknown')
        elif row['seller'] == 'SUBMISSION': return row.get('buyer', 'Unknown')
        return 'Other' # Should not happen with initial filter
    player_trades['fill_type'] = player_trades.apply(get_fill_type, axis=1)

    # Filter for expected fill types based on run mode
    # Permutation runs only match against the override book (labeled ExplicitBook)
    expected_fills = ['ExplicitBook', 'InferredBot'] if not is_permutation_run else ['ExplicitBook']
    player_trades = player_trades[player_trades['fill_type'].isin(expected_fills)].copy()

    if player_trades.empty:
        st.info(f"Fill Dist Chart: No trades found with expected fill types {expected_fills} ({title_prefix}).")
        return

    # Ensure numeric types for price/quantity
    player_trades['price'] = pd.to_numeric(player_trades['price'], errors='coerce')
    player_trades['quantity'] = pd.to_numeric(player_trades['quantity'], errors='coerce')
    player_trades.dropna(subset=['price', 'quantity', 'timestamp', 'symbol'], inplace=True) # Drop essential NaNs

    if player_trades.empty:
        st.info(f"Fill Dist Chart: No valid player trades remain after cleaning ({title_prefix}).")
        return

    # --- Merge Trades with Market VWAP ---
    # Prepare VWAP data (timestamp, product, vwap)
    vwap_data_for_merge = market_data_processed[['timestamp', 'product', 'vwap']].dropna().copy()
    if 'product' not in vwap_data_for_merge.columns:
         st.warning(f"Fill Dist Chart: 'product' column missing in VWAP data ({title_prefix}). Cannot merge.")
         return

    merged_trades = pd.merge(
        player_trades,
        vwap_data_for_merge,
        left_on=['timestamp', 'symbol'],
        right_on=['timestamp', 'product'],
        how='left' # Keep all player trades, match VWAP where possible
    )

    # Drop rows where merge failed (no VWAP for that product/timestamp) or price was NaN
    merged_trades.dropna(subset=['vwap', 'price'], inplace=True)
    if merged_trades.empty:
        st.info(f"Fill Dist Chart: No trades remain after merging with VWAP data ({title_prefix}). Might indicate timestamp mismatches or missing VWAP.")
        return

    # --- Calculate Distance & Aggregate ---
    try:
        merged_trades['distance'] = (merged_trades['price'] - merged_trades['vwap']).abs()
        # ... (rest of fill dist chart function remains the same) ...
        # Define buckets and labels
        max_b = 5 # Max distance bucket (e.g., $5+)
        labels = {i: f"${i:.2f} - ${i+1-0.01:.2f}" for i in range(max_b)}
        labels[max_b] = f"${max_b:.2f}+"
        # Apply bucketing
        merged_trades['dist_bucket'] = np.floor(merged_trades['distance']).astype(int).clip(upper=max_b)
        merged_trades['dist_label'] = merged_trades['dist_bucket'].map(labels)
        # Ensure correct order for plotting
        label_order = [labels[i] for i in range(max_b + 1)]

        # Aggregate volume per bucket/fill_type/product
        # Use 'symbol' for product grouping
        agg_cols = ['dist_label', 'fill_type', 'symbol']
        volume_summary = merged_trades.groupby(agg_cols)['quantity'].sum().reset_index()
        volume_summary.rename(columns={'quantity':'total_volume', 'symbol': 'product'}, inplace=True) # Rename for clarity

        # Also calculate overall summary across all products
        overall_summary = volume_summary.groupby(['dist_label', 'fill_type'])['total_volume'].sum().reset_index()
        overall_summary['product'] = 'Overall' # Add marker column

        # Combine overall and per-product summaries
        combined_summary = pd.concat([overall_summary, volume_summary], ignore_index=True)

        # Define order for facets
        prod_names = sorted(merged_trades['symbol'].unique())
        display_order = ['Overall'] + prod_names

        if combined_summary.empty:
            st.info(f"Fill Dist Chart: No volume data after aggregation ({title_prefix}).")
            return

        # --- Create Chart ---
        # Define colors (adjust as needed)
        color_map = {"ExplicitBook": px.colors.qualitative.Plotly[0], "InferredBot": px.colors.qualitative.Plotly[1]}

        fig_fill = px.bar(combined_summary, x='dist_label', y='total_volume', color='fill_type',
                          facet_col='product', # Facet by product (including 'Overall')
                          barmode='group',
                          title=f"{title_prefix} - Fill Volume by Distance from VWAP",
                          labels={'dist_label':'Absolute Distance from VWAP ($)',
                                  'total_volume':'Total Volume Traded',
                                  'fill_type':'Fill Source',
                                  'product':'Product Group'}, # Updated label
                          category_orders={"dist_label": label_order, "product": display_order}, # Order buckets and facets
                          color_discrete_map=color_map) # Use explicit color map

        fig_fill.update_layout(margin=dict(l=20,r=20,t=50,b=20), height=400) # Adjusted margins
        # Clean up facet titles
        fig_fill.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        # Ensure y-axis starts at 0
        fig_fill.update_yaxes(rangemode='tozero')

        st.plotly_chart(fig_fill, use_container_width=True)


    except Exception as e:
         st.error(f"Error creating Fill Distance chart ({title_prefix}): {e}")
         traceback.print_exc()


# --- END: Charting Functions ---

# ... (rest of app.py remains the same) ...

# --- Session State Initialization ---
def initialize_session_state():
    if 'log_viewer_mode' not in st.session_state: st.session_state.log_viewer_mode = False
    if 'parsed_log_data' not in st.session_state: st.session_state.parsed_log_data = ([], pd.DataFrame(), pd.DataFrame(), "") # s_logs, m_df, t_df, raw_content
    if 'last_loaded_log_file' not in st.session_state: st.session_state.last_loaded_log_file = None
    if 'last_run_trader_file' not in st.session_state: st.session_state.last_run_trader_file = None
    if 'permutation_results' not in st.session_state: st.session_state.permutation_results = {'original_pnl': None, 'permuted_pnls': [], 'permuted_pnls_raw': [], 'p_value': None}
    if 'run_permutation' not in st.session_state: st.session_state.run_permutation = False
    if 'show_output_log_state' not in st.session_state: st.session_state.show_output_log_state = False
    if 'view_permutation_index' not in st.session_state: st.session_state.view_permutation_index = 0 # 0 or None indicates no specific perm view active
    if 'viewed_permutation_data' not in st.session_state: st.session_state.viewed_permutation_data = ([], pd.DataFrame(), pd.DataFrame(), "") # s_logs, m_df, t_df, raw_output
    if 'permutation_rerun_args' not in st.session_state: st.session_state.permutation_rerun_args = {}
    if 'view_perm_input' not in st.session_state: st.session_state.view_perm_input = 1 # Default input to 1
initialize_session_state()

# --- Clear/Rerun Functions ---
def clear_permutation_results():
    print("Clearing permutation results from session state.")
    st.session_state.permutation_results = {'original_pnl': None, 'permuted_pnls': [], 'permuted_pnls_raw': [], 'p_value': None}
    st.session_state.view_permutation_index = 0
    st.session_state.viewed_permutation_data = ([], pd.DataFrame(), pd.DataFrame(), "")
    st.session_state.permutation_rerun_args = {} # Clear args too
    st.session_state.view_perm_input = 1 # Reset input field

def clear_parsed_data():
    print("Clearing parsed data and permutation results from session state.")
    st.session_state.parsed_log_data = ([], pd.DataFrame(), pd.DataFrame(), "")
    st.session_state.last_loaded_log_file = None
    st.session_state.last_run_trader_file = None
    clear_permutation_results() # Also clear perm results when clearing main data
def rerun_and_view_permutation(perm_index_to_view: int, perm_args: dict):
    """Regenerates and reruns a specific independent permutation index."""
    st.session_state.viewed_permutation_data = ([], pd.DataFrame(), pd.DataFrame(), "")
    st.session_state.view_permutation_index = 0
    if perm_index_to_view <= 0: st.warning("Permutation index must be >= 1."); return
    if not perm_args: st.warning("Cannot view: Rerun args not found."); return
    if not perm_utils: st.error("Cannot view: Permutation utils unavailable."); return

    # Validate required keys for independent permutations
    required_keys = [
        'product_data', # Dict: {product: {'initial_vwap': float, 'abs_changes': Series}}
        'product_relative_books', # Dict: {product: {orig_ts: {'buys':..., 'sells':...}}}
        'block_size', 'trader_file', 'data_files', 'time_range',
        'bot_behavior', 'ignore_limits', 'products' # List of products permuted
    ]
    missing_keys = [k for k in required_keys if k not in perm_args]
    if missing_keys: st.error(f"Missing rerun args: {missing_keys}"); return
    if not isinstance(perm_args['product_data'], dict) or not perm_args['product_data']:
        st.error("Invalid 'product_data' in rerun args."); return

    max_perms_run = len(st.session_state.permutation_results.get('permuted_pnls_raw', []))
    if perm_index_to_view > max_perms_run: st.warning(f"Cannot view {perm_index_to_view}. Only {max_perms_run} were run."); return

    perm_progress = st.progress(0.0, text=f"Regenerating permutation {perm_index_to_view} data...")
    try:
        products_to_permute = perm_args['products']
        all_permuted_vwaps: Dict[str, pd.Series] = {}
        all_original_indices_maps: Dict[str, pd.Series] = {}
        all_permuted_timestamps_set = set()
        base_seed = perm_index_to_view - 1 # 0-based seed for iteration i

        # --- Regenerate Data Per Product ---
        for idx, product in enumerate(products_to_permute):
            perm_progress.progress(0.1 + 0.4 * (idx / len(products_to_permute)), text=f"Permuting {product}...")
            prod_data = perm_args['product_data'].get(product)
            if not prod_data or 'abs_changes' not in prod_data or 'initial_vwap' not in prod_data:
                print(f"Warn Rerun: Missing data for {product}, skipping its permutation.")
                continue

            # Use a combined seed for independence
            # Hash ensures different products get different sequences for the same base_seed
            product_seed = base_seed + int(hashlib.sha1(product.encode()).hexdigest(), 16)
            np.random.seed(product_seed % (2**32 - 1)) # Ensure seed is within valid range

            permuted_changes, orig_map = perm_utils.block_permutation(
                prod_data['abs_changes'], perm_args['block_size']
            )
            reconstructed_vwap = perm_utils.reconstruct_absolute_vwap(
                prod_data['initial_vwap'], permuted_changes
            )

            if reconstructed_vwap.empty:
                print(f"Warn Rerun: Reconstruction failed for {product}.")
                continue

            all_permuted_vwaps[product] = reconstructed_vwap
            all_original_indices_maps[product] = orig_map
            all_permuted_timestamps_set.update(reconstructed_vwap.index.unique())

        if not all_permuted_vwaps: raise ValueError("No product VWAPs were reconstructed.")

        # --- Generate Books ---
        perm_progress.progress(0.6, text=f"Generating permuted order books...")
        all_permuted_timestamps_sorted = np.sort(list(all_permuted_timestamps_set))

        permuted_book_cache = perm_utils.generate_permuted_order_book_cache_independent(
            products_to_permute,
            all_permuted_timestamps_sorted,
            all_permuted_vwaps,
            all_original_indices_maps,
            perm_args['product_relative_books'] # Pass the per-product relative books
        )
        if not permuted_book_cache: raise ValueError("Generated permuted book cache is empty.")

        # --- Run Backtest ---
        perm_progress.progress(0.8, text=f"Running backtest...")
        view_backtester = run_backtest(
             perm_args['trader_file'], perm_args['data_files'][:1], perm_args['time_range'],
             perm_args['bot_behavior'], perm_args['ignore_limits'],
             order_depth_override=permuted_book_cache, disable_inferred=True
        )

        # --- Parse & Store ---
        perm_progress.progress(0.95, text=f"Parsing results...")
        if view_backtester and hasattr(view_backtester, 'output') and view_backtester.output:
            s_logs, m_df, t_df = parse_backtest_output(view_backtester.output)
            st.session_state.viewed_permutation_data = (s_logs, m_df, t_df, view_backtester.output)
            st.session_state.view_permutation_index = perm_index_to_view
            st.success(f"Loaded results for permutation {perm_index_to_view}.")
        else: st.error(f"Failed rerun for permutation {perm_index_to_view}.")

    except Exception as e: st.error(f"Error viewing permutation {perm_index_to_view}: {e}"); traceback.print_exc()
    finally: perm_progress.empty()



# --- UI Layout ---
st.title("🌊 Prosperity Backtester & Permutation Test 📈")

# Sidebar for Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    # Mode Selection
    is_viewer_mode = st.toggle("Log Viewer Mode", key="log_viewer_mode", value=False, help="If enabled, parse and display a single log file instead of running a trader.", on_change=clear_parsed_data)
    st.caption("Run simulation with trader." if not is_viewer_mode else "Parse and display a single log file.")
    st.divider()

    # Initialize config variables to be populated based on mode
    selected_trader_fname = None
    selected_data_fnames = []
    selected_log_viewer_file = None
    perm_blocksize = 10 # Default block size
    perm_n_iterations = 99 # Default iterations
    time_range_values = (0, constants.MAX_TIMESTAMP) # Default full range
    bot_behavior = 'lte'
    ignore_limits_checkbox = False
    run_permutation_test = False

    # --- Backtest Mode UI ---
    if not is_viewer_mode:
        st.subheader("1. Select Trader")
        try:
            trader_files = sorted([f for f in os.listdir(TRADER_DIR_ABS) if f.endswith(".py") and f != "__init__.py"])
        except FileNotFoundError:
            st.error(f"Trader directory not found at {TRADER_DIR_ABS}")
            trader_files = []
        except Exception as e:
            st.error(f"Error listing trader files: {e}")
            trader_files = []

        if not trader_files:
            st.warning(f"No Python files found in trader directory: {TRADER_DIR_ABS}")
        else:
            selected_trader_fname = st.selectbox("Trader Python File:", trader_files, key="trader_select", index=0, label_visibility="collapsed")
            # Store the selected trader for display purposes
            st.session_state.last_run_trader_file = selected_trader_fname
            # Expander to view trader code
            if selected_trader_fname:
                trader_code = load_file_content(os.path.join(TRADER_DIR_ABS, selected_trader_fname))
                with st.expander("View Trader Code"):
                    st.code(trader_code, language="python")

        st.subheader("2. Select Data Log(s)")
        try:
            data_files = sorted([f for f in os.listdir(DATA_DIR_ABS) if f.endswith(".log")])
        except FileNotFoundError:
            st.error(f"Data directory not found at {DATA_DIR_ABS}")
            data_files = []
        except Exception as e:
             st.error(f"Error listing data files: {e}")
             data_files = []

        if not data_files:
            st.warning(f"No log files found in data directory: {DATA_DIR_ABS}")
        else:
            # Default to first file if available
            default_selection = [data_files[0]] if data_files else None
            selected_data_fnames = st.multiselect("Log Files (first file is primary):", data_files, default=default_selection, key="data_select", label_visibility="collapsed")

        st.subheader("3. Settings")
        # Use columns for better layout
        col_a, col_b = st.columns(2)
        with col_a:
            # Ensure bot behavior options match the backtester literals
            bot_behavior = st.selectbox("Bot Matching (Inferred):", ["none", "eq", "lt", "lte"], index=3, key="bot_behavior", help="Rule for matching against inferred bot liquidity (N/A for Permutation runs). 'lte' is standard.")
        with col_b:
            ignore_limits_checkbox = st.checkbox("Ignore Pos Limits", value=False, key="ignore_limits", help="Disable position limit checks.")

        # --- Time Range Slider (Backtest) ---
        min_time, max_time = 0, constants.MAX_TIMESTAMP
        # Attempt to get range from primary selected log
        if selected_data_fnames:
            try:
                 print(f"Attempting to get time range from: {selected_data_fnames[0]}")
                 # Use cached parsing if possible? No, parse fresh to get range.
                 with open(os.path.join(DATA_DIR_ABS, selected_data_fnames[0]), 'r', encoding='utf-8') as f:
                     _, temp_mkt, _ = util._parse_data(f) # Use util parser directly
                 if temp_mkt is not None and not temp_mkt.empty and pd.api.types.is_numeric_dtype(temp_mkt.index):
                     min_time = int(temp_mkt.index.min())
                     max_time = int(temp_mkt.index.max())
                     print(f"Time range from log: {min_time} - {max_time}")
                 else: print("Could not determine time range from log index.")
            except FileNotFoundError: print("Primary log file not found for time range.")
            except Exception as e: print(f"Warning: Failed to get time range from {selected_data_fnames[0]}: {e}")
        # Ensure slider values are within reasonable bounds
        min_time = max(0, min_time)
        max_time = min(constants.MAX_TIMESTAMP, max_time) if max_time > min_time else constants.MAX_TIMESTAMP

        time_range_values = st.slider("Time Range:", min_value=min_time, max_value=max_time, value=(min_time, max_time), step=100, key="timerange_backtest")

        # --- Permutation Test UI (Backtest Only) ---
        st.divider()
        st.subheader("Permutation Testing")
        perm_utils_available = perm_utils is not None
        if not perm_utils_available:
             st.caption("Permutation testing disabled (permutation_utils.py not found).")

        run_permutation_test = st.checkbox("Enable Permutation Test", key="run_permutation", value=False, on_change=clear_permutation_results, disabled=(not perm_utils_available))

        if run_permutation_test and perm_utils_available:
             # Calculate approx number of timestamps
             total_ts = max(1, (time_range_values[1] - time_range_values[0]) // 100 + 1)
             # Suggest max block size (e.g., 10% of duration, min 10)
             max_block = max(10, total_ts // 10)
             perm_blocksize = st.number_input("Block Size (Timesteps):", min_value=1, max_value=max(1, total_ts), value=max(1, total_ts // 20), step=10, key="perm_blocksize", help=f"Length of consecutive price change blocks to shuffle (Max suggested: ~{max_block}).")
             perm_n_iterations = st.number_input("Number of Permutations:", min_value=1, max_value=10000, value=99, step=10, key="perm_n_iterations", help="Number of shuffled price histories to run.")
        elif run_permutation_test and not perm_utils_available:
             st.error("Permutation testing enabled, but permutation utilities failed to import.")

    # --- Log Viewer Mode UI ---
    else: # is_viewer_mode == True
         st.subheader("Select Log File")
         try:
             data_files = sorted([f for f in os.listdir(DATA_DIR_ABS) if f.endswith(".log")])
         except FileNotFoundError:
             st.error(f"Data directory not found at {DATA_DIR_ABS}")
             data_files = []
         except Exception as e:
             st.error(f"Error listing data files: {e}")
             data_files = []

         if not data_files:
             st.warning(f"No log files found in data directory: {DATA_DIR_ABS}")
         else:
             selected_log_viewer_file = st.selectbox("Log File to View:", data_files, key="log_viewer_select", index=0, label_visibility="collapsed")
             if selected_log_viewer_file:
                 selected_data_fnames = [selected_log_viewer_file] # Store single file name
                 st.session_state.last_loaded_log_file = selected_log_viewer_file

         # --- Time Range Slider (Log Viewer) ---
         min_time, max_time = 0, constants.MAX_TIMESTAMP
         if selected_log_viewer_file: # Determine range from selected log
              try:
                   print(f"Attempting to get time range from viewer log: {selected_log_viewer_file}")
                   with open(os.path.join(DATA_DIR_ABS, selected_log_viewer_file), 'r', encoding='utf-8') as f:
                       _, temp_mkt, _ = util._parse_data(f)
                   if temp_mkt is not None and not temp_mkt.empty and pd.api.types.is_numeric_dtype(temp_mkt.index):
                       min_time = int(temp_mkt.index.min())
                       max_time = int(temp_mkt.index.max())
                       print(f"Time range from viewer log: {min_time} - {max_time}")
                   else: print("Could not determine time range from viewer log index.")
              except FileNotFoundError: print("Viewer log file not found for time range.")
              except Exception as e: print(f"Warning: Failed to get time range from viewer log {selected_log_viewer_file}: {e}")
         # Ensure slider values are within reasonable bounds
         min_time = max(0, min_time)
         max_time = min(constants.MAX_TIMESTAMP, max_time) if max_time > min_time else constants.MAX_TIMESTAMP

         time_range_values = st.slider("Time Range Filter:", min_value=min_time, max_value=max_time, value=(min_time, max_time), step=100, key="timerange_viewer")

    # --- Common UI Elements ---
    st.divider()
    st.checkbox("Show Raw Output Log in Results", key="show_output_log_state", value=False)
    st.divider()

    # Determine button label and disabled state
    run_perm_enabled_and_selected = run_permutation_test and not is_viewer_mode and perm_utils_available
    button_label = "📊 Load Log File" if is_viewer_mode else ("🧪 Run Backtest & Permutations" if run_perm_enabled_and_selected else "🚀 Run Backtest")
    # Disable button if essential inputs are missing
    button_disabled = (is_viewer_mode and not selected_log_viewer_file) or \
                      (not is_viewer_mode and (not selected_trader_fname or not selected_data_fnames))

    run_button_pressed = st.button(button_label, use_container_width=True, type="primary", disabled=button_disabled)


# --- Main Area (Results) ---
st.header("📊 Results")
results_area = st.container() # Container to hold results display

# Initialize variables to hold data for display
s_logs_disp, m_df_disp, t_df_disp, raw_output_disp = [], pd.DataFrame(), pd.DataFrame(), ""
display_source_info = "Configure parameters and run." # Default message
data_to_display_available = False


# --- Main Execution Logic (Triggered by Button Press) ---
if run_button_pressed:
    print("\n--- Run Button Pressed ---")
    clear_parsed_data() # Clear all previous results on new run trigger

    if is_viewer_mode: # --- Log Viewer Mode ---
        if selected_log_viewer_file:
            log_file_path = os.path.join(DATA_DIR_ABS, selected_log_viewer_file)
            print(f"--- Loading Log File: {log_file_path} ---")
            with st.spinner(f"Parsing **{selected_log_viewer_file}**..."):
                try:
                    raw_content = load_file_content(log_file_path)
                    if raw_content.startswith("Error:"):
                        st.error(raw_content) # Show error from file loading
                    else:
                        s_logs, m_df, t_df = parse_backtest_output(raw_content)
                        # Apply time filter *after* parsing the whole file
                        m_df_f = m_df[(m_df.index >= time_range_values[0]) & (m_df.index <= time_range_values[1])] if m_df is not None else pd.DataFrame()
                        t_df_f = t_df[(t_df.index >= time_range_values[0]) & (t_df.index <= time_range_values[1])] if t_df is not None else pd.DataFrame()
                        # Filter sandbox logs by time range
                        s_logs_f = [log for log in s_logs if isinstance(log.get("timestamp"), int) and time_range_values[0] <= log["timestamp"] <= time_range_values[1]]
                        # Store filtered data in session state
                        st.session_state.parsed_log_data = (s_logs_f, m_df_f, t_df_f, raw_content) # Store filtered data + raw
                        st.success(f"Successfully parsed and filtered **{selected_log_viewer_file}**.")
                except Exception as e:
                    st.error(f"Failed to load or parse log file: {e}")
                    st.code(traceback.format_exc())
        else:
            st.warning("Please select a log file to view.")

    elif not is_viewer_mode and selected_trader_fname and selected_data_fnames: # --- Backtest Mode ---
        run_permutations_now = st.session_state.get('run_permutation', False) and perm_utils_available

        # --- 1. Run Original Backtest ---
        st.info(f"Running original backtest for **{selected_trader_fname}**...")
        with st.spinner(f"Running original backtest..."):
             backtester_instance = run_backtest(
                 selected_trader_fname, selected_data_fnames, time_range_values,
                 bot_behavior, ignore_limits_checkbox,
                 order_depth_override=None, disable_inferred=False # Standard run
             )

        if not backtester_instance or not hasattr(backtester_instance, 'output') or not backtester_instance.output:
             st.error("Original backtest failed or produced no output. Check console logs.")
             # Optionally clear state? or leave partial results if they exist?
             # clear_parsed_data() # Clears everything
             st.stop() # Stop further execution if original fails

        print("Parsing original backtest output...")
        s_logs_orig, m_df_orig, t_df_orig = parse_backtest_output(backtester_instance.output)
        # Store original run results in session state
        st.session_state.parsed_log_data = (s_logs_orig, m_df_orig, t_df_orig, backtester_instance.output)

        # --- Extract Final PNL from original run ---
        final_pnl_original = np.nan
        if m_df_orig is not None and not m_df_orig.empty and 'profit_and_loss' in m_df_orig.columns:
            try:
                # Ensure PnL is numeric before calculations
                pnl_series_orig = pd.to_numeric(m_df_orig['profit_and_loss'], errors='coerce')
                # Find the last timestamp in the index
                last_timestamp = pnl_series_orig.index.max() if not pnl_series_orig.empty else None
                if last_timestamp is not None:
                    # Sum PNL values for all products at the last timestamp
                    final_pnl_values_at_last_ts = pnl_series_orig.loc[last_timestamp]
                    # Handle scalar vs Series case (if only one product)
                    final_pnl_original = final_pnl_values_at_last_ts.sum() if isinstance(final_pnl_values_at_last_ts, pd.Series) else final_pnl_values_at_last_ts

                    # Ensure it's a standard float or NaN
                    if isinstance(final_pnl_original, (pd.Series, pd.DataFrame)): # Should not happen after sum
                         final_pnl_original = np.nan
                    elif not pd.notna(final_pnl_original):
                         final_pnl_original = np.nan
                    else:
                         final_pnl_original = float(final_pnl_original)

            except Exception as pnl_calc_err:
                 print(f"Error calculating final PNL from original run: {pnl_calc_err}")
                 final_pnl_original = np.nan # Ensure NaN on error

        st.session_state.permutation_results['original_pnl'] = final_pnl_original
        pnl_display_str = f"{final_pnl_original:.2f}" if pd.notna(final_pnl_original) else "N/A"
        st.success(f"Original backtest complete. Final Total PNL: **{pnl_display_str}**")


        # --- 2. Run Permutations (if enabled) ---
        if run_permutations_now:
            st.info(f"Starting **{perm_n_iterations}** independent permutations (Block Size: {perm_blocksize})...")
            perm_progress = st.progress(0.0, text="Preparing base data for permutations...")
            permuted_pnls_list_raw = []
            products_in_run = []
            # --- MODIFIED: Prepare Base Data (Independent Products) ---
            try:
                with st.spinner("Analyzing market data for permutations..."):
                    print("Preparing base data for independent permutations...")
                    if not hasattr(backtester_instance, 'market_data') or backtester_instance.market_data.empty:
                         raise ValueError("Original backtester instance missing 'market_data'.")
                    market_data_orig = backtester_instance.market_data.copy()
                    products_in_run = backtester_instance.products # Use products from the instance

                    product_data_for_perm: Dict[str, Dict[str, Any]] = {}
                    product_relative_books_cache: Dict[str, Dict[int, Any]] = {}

                    for product in products_in_run:
                         print(f"-- Processing base data for: {product}")
                         product_market_data = market_data_orig[market_data_orig['product'] == product]
                         if product_market_data.empty:
                              print(f"Warn Prep: No market data rows found for {product}. Skipping permutation.")
                              continue

                         # Ensure/Calculate VWAP for this product
                         if 'vwap' not in product_market_data.columns or product_market_data['vwap'].isna().all():
                              print(f"Calculating VWAP for {product}...")
                              price_vol_cols = [f'{s}_{t}_{l}' for s in ['bid','ask'] for t in ['price','volume'] for l in [1,2,3]]
                              for col in price_vol_cols:
                                   if col in product_market_data.columns and not pd.api.types.is_numeric_dtype(product_market_data[col]):
                                       product_market_data[col] = pd.to_numeric(product_market_data[col], errors='coerce')
                              product_market_data['vwap'] = product_market_data.apply(calculate_row_vwap, axis=1)
                              # Fill within the product's own data
                              product_market_data['vwap'] = product_market_data['vwap'].ffill().bfill()

                         product_vwap_series = product_market_data['vwap'].reindex(market_data_orig.index.unique()).ffill().bfill()
                         product_vwap_series.dropna(inplace=True)
                         if product_vwap_series.empty:
                              print(f"Warn Prep: VWAP series empty for {product}. Skipping permutation.")
                              continue

                         initial_vwap = product_vwap_series.iloc[0]
                         if pd.isna(initial_vwap):
                              print(f"Warn Prep: Initial VWAP NaN for {product}. Skipping permutation.")
                              continue

                         abs_changes = perm_utils.get_absolute_vwap_changes(product_vwap_series)
                         rel_books = perm_utils.calculate_relative_product_books(product_market_data, product_vwap_series)

                         product_data_for_perm[product] = {'initial_vwap': initial_vwap, 'abs_changes': abs_changes}
                         product_relative_books_cache[product] = rel_books
                         print(f"OK: Base data prepared for {product}.")

                    if not product_data_for_perm: raise ValueError("No valid product data could be prepared for permutation.")

                    # Store args needed for reruns
                    st.session_state.permutation_rerun_args = {
                        'trader_file': selected_trader_fname,
                        'data_files': selected_data_fnames[:1], 'time_range': time_range_values,
                        'bot_behavior': bot_behavior, 'ignore_limits': ignore_limits_checkbox,
                        'products': list(product_data_for_perm.keys()), # Store list of products successfully prepped
                        'product_data': product_data_for_perm,
                        'product_relative_books': product_relative_books_cache,
                        'block_size': perm_blocksize,
                    }
                    print("Base data prepared successfully for all valid products.")

            except Exception as prep_e:
                st.error(f"Error preparing data for independent permutations: {prep_e}"); perm_progress.empty(); traceback.print_exc(); st.stop()
            # --- End Prepare Data ---


            # --- MODIFIED: Permutation Loop (Independent Products) ---
            perm_progress.progress(0.0, text="Running permutation backtests...")
            products_to_permute = st.session_state.permutation_rerun_args['products'] # Use list from prep

            for i in range(perm_n_iterations):
                perm_index_for_seed = i # 0-based seed for iteration i
                print(f"\n--- Running Independent Permutation {i+1}/{perm_n_iterations} (Base Seed: {perm_index_for_seed}) ---")
                perm_backtester = None; current_perm_pnl = np.nan

                try:
                    # --- Generate Permuted Data Per Product ---
                    all_permuted_vwaps: Dict[str, pd.Series] = {}
                    all_original_indices_maps: Dict[str, pd.Series] = {}
                    all_permuted_timestamps_set = set()

                    for product in products_to_permute:
                         prod_data = st.session_state.permutation_rerun_args['product_data'][product]
                         # Use a combined seed for independence
                         product_seed = perm_index_for_seed + int(hashlib.sha1(product.encode()).hexdigest(), 16)
                         np.random.seed(product_seed % (2**32 - 1))

                         permuted_changes, orig_map = perm_utils.block_permutation(prod_data['abs_changes'], perm_blocksize)
                         reconstructed_vwap = perm_utils.reconstruct_absolute_vwap(prod_data['initial_vwap'], permuted_changes)

                         if reconstructed_vwap.empty: raise ValueError(f"Reconstruction failed for {product} in iter {i+1}")

                         all_permuted_vwaps[product] = reconstructed_vwap
                         all_original_indices_maps[product] = orig_map
                         all_permuted_timestamps_set.update(reconstructed_vwap.index.unique())

                    if not all_permuted_vwaps: raise ValueError("No product VWAPs reconstructed in iter {i+1}")
                    # --- End Generate Data ---

                    # --- Generate Books ---
                    all_permuted_timestamps_sorted = np.sort(list(all_permuted_timestamps_set))
                    permuted_book_cache = perm_utils.generate_permuted_order_book_cache_independent(
                         products_to_permute, all_permuted_timestamps_sorted,
                         all_permuted_vwaps, all_original_indices_maps,
                         st.session_state.permutation_rerun_args['product_relative_books']
                    )
                    if not permuted_book_cache: raise ValueError("Generated permuted book cache empty in iter {i+1}")
                    # --- End Generate Books ---

                    # --- Run Backtest ---
                    perm_backtester = run_backtest(
                         selected_trader_fname, selected_data_fnames[:1], time_range_values,
                         bot_behavior, ignore_limits_checkbox,
                         order_depth_override=permuted_book_cache, disable_inferred=True
                    )
                    # --- End Backtest ---

                    # --- MODIFIED & ROBUST: Extract PNL ---
                    current_perm_pnl = np.nan # Default to NaN

                    if perm_backtester and hasattr(perm_backtester, 'output') and perm_backtester.output:
                        print(f"Perm {i+1}: Backtest ran, attempting PNL extraction...")
                        try:
                            _, m_df_perm, _ = parse_backtest_output(perm_backtester.output)

                            if m_df_perm is not None and not m_df_perm.empty and 'profit_and_loss' in m_df_perm.columns:
                                print(f"Perm {i+1}: Market DF parsed, PNL column found.")
                                pnl_series_perm = pd.to_numeric(m_df_perm['profit_and_loss'], errors='coerce')

                                if not pnl_series_perm.isna().all(): # Check if there's any non-NaN PNL data
                                    last_ts_perm = pnl_series_perm.index.max() if not pnl_series_perm.empty else None

                                    if last_ts_perm is not None:
                                        print(f"Perm {i+1}: Last timestamp found: {last_ts_perm}")
                                        # Use .loc[], which can return a Series or scalar
                                        final_pnl_values_at_last_ts = pnl_series_perm.loc[last_ts_perm]

                                        # Check if the result is valid before summing/converting
                                        if final_pnl_values_at_last_ts is not None:
                                            # Sum if it's a Series (multiple products), otherwise use the scalar value
                                            final_pnl_sum = final_pnl_values_at_last_ts.sum() if isinstance(final_pnl_values_at_last_ts, pd.Series) else final_pnl_values_at_last_ts

                                            if pd.notna(final_pnl_sum):
                                                current_perm_pnl = float(final_pnl_sum)
                                                print(f"Perm {i+1}: Successfully extracted PNL: {current_perm_pnl}")
                                            else:
                                                print(f"Perm {i+1}: Final PNL value at last timestamp was NaN.")
                                        else:
                                            print(f"Perm {i+1}: Could not retrieve PNL value at last timestamp ({last_ts_perm}).")
                                    else:
                                        print(f"Perm {i+1}: Could not determine last timestamp from PNL series.")
                                else:
                                    print(f"Perm {i+1}: PNL series contained only NaNs.")
                            else:
                                print(f"Perm {i+1}: Market DF empty or 'profit_and_loss' column missing after parsing.")

                        except Exception as parse_pnl_err:
                            print(f"Perm {i+1}: Error during PNL parsing/extraction: {parse_pnl_err}")
                            # current_perm_pnl remains np.nan
                    else:
                        print(f"Perm {i+1}: Backtest failed or produced no output.")
                    # --- End PNL Extraction ---

                except Exception as loop_e:
                    print(f"ERROR processing permutation loop {i+1}: {loop_e}")
                    traceback.print_exc()
                    current_perm_pnl = np.nan # Ensure NaN on outer loop error
                finally:
                    # Append the final determined PNL (float or np.nan) to the raw list
                    permuted_pnls_list_raw.append(current_perm_pnl)
                    print(f"Permutation {i+1} Final PNL Recorded: {current_perm_pnl}") # Explicit log
                    if perm_backtester: del perm_backtester # Cleanup
                perm_progress.progress((i + 1) / perm_n_iterations, text=f"Running permutation {i+1}/{perm_n_iterations}")
            # --- End Permutation Loop ---

            # --- Calculate p-value ---
            # Filter out NaNs before calculation
            valid_perm_pnls = [p for p in permuted_pnls_list_raw if pd.notna(p) and isinstance(p, (int, float))]
            p_value = np.nan
            pnl_orig = st.session_state.permutation_results.get('original_pnl')

            if pd.isna(pnl_orig):
                 print("Warning: Original PNL is NaN. Cannot calculate p-value.")
                 st.warning("Original PNL calculation failed, cannot compute p-value.")
            elif valid_perm_pnls:
                 count_ge = sum(p >= pnl_orig for p in valid_perm_pnls)
                 p_value = (count_ge + 1) / (len(valid_perm_pnls) + 1)
                 print(f"Permutation p-value calculation: {p_value:.4f} ({count_ge} >= {pnl_orig:.2f} out of {len(valid_perm_pnls)} valid runs)")
            else:
                 print("Warning: No valid permuted PNLs were generated. Cannot calculate p-value.")
                 st.warning("No valid permutation runs completed, cannot compute p-value.")

            # Store results: VALID PNLs for histogram, RAW PNLs for inspection
            st.session_state.permutation_results['permuted_pnls'] = valid_perm_pnls
            st.session_state.permutation_results['permuted_pnls_raw'] = permuted_pnls_list_raw # Store the full list including NaNs
            st.session_state.permutation_results['p_value'] = p_value
            perm_progress.empty()
            st.success("Permutation testing complete.")
            print(f"Final stored valid PNLs for histogram: {st.session_state.permutation_results['permuted_pnls']}") # Log final list
    else:
        # Handle cases where button pressed but conditions not met (e.g., missing files)
        if not is_viewer_mode and (not selected_trader_fname or not selected_data_fnames):
             st.warning("Please select a trader and at least one data log file.")
        # Button press logic finished


# --- Display Results ---
# Determine which data source to display based on session state

active_view_index = st.session_state.get('view_permutation_index', 0)
viewed_perm_data = st.session_state.get('viewed_permutation_data')
parsed_main_data = st.session_state.get('parsed_log_data')

is_displaying_perm_view = False
if active_view_index > 0 and viewed_perm_data and isinstance(viewed_perm_data, tuple) and len(viewed_perm_data) == 4 and viewed_perm_data[1] is not None:
    # Check if the DataFrame in the tuple is not empty
    if not viewed_perm_data[1].empty:
        s_logs_disp, m_df_disp, t_df_disp, raw_output_disp = viewed_perm_data
        display_source_info = f"Viewing Results for Permutation Run **#{active_view_index}**"
        data_to_display_available = True
        is_displaying_perm_view = True
    else:
         print(f"Attempted to display perm view {active_view_index}, but market data was empty.")
         # Fall back to original data if perm view data is invalid/empty
         active_view_index = 0 # Reset the view index state
         st.session_state.view_permutation_index = 0
         st.warning(f"Data for permutation {active_view_index} seems empty, showing original run instead.")


# Fallback to main parsed data if not viewing a specific permutation or if perm view failed
if not is_displaying_perm_view and parsed_main_data and isinstance(parsed_main_data, tuple) and len(parsed_main_data) == 4 and parsed_main_data[1] is not None:
     # Check if the DataFrame in the tuple is not empty
     if not parsed_main_data[1].empty:
        s_logs_disp, m_df_disp, t_df_disp, raw_output_disp = parsed_main_data
        source_desc_key = st.session_state.last_loaded_log_file if st.session_state.log_viewer_mode else st.session_state.last_run_trader_file
        source_mode = "Log File" if st.session_state.log_viewer_mode else "Original Backtest"
        display_source_info = f"Viewing Results for {source_mode}: **{source_desc_key or 'N/A'}**"
        data_to_display_available = True
     else:
          print("Original parsed data's market data is empty.")


# --- Display Area ---
with results_area:
    if not data_to_display_available:
        st.info(display_source_info) # Show default message or error if run failed mid-way
    else:
        st.subheader("Run Summary")
        st.caption(display_source_info)

        # --- Determine Tabs ---
        perm_results = st.session_state.get('permutation_results', {})
        tabs_list = ["📊 Charts"] # Always show main charts tab
        # Show perm tab only if permutations were run (check raw list) and not in viewer mode
        if not st.session_state.log_viewer_mode and perm_results and perm_results.get('permuted_pnls_raw'):
             tabs_list.append("🧪 Permutation Test")
        # Add Logs tab if configured or if data is present
        if st.session_state.show_output_log_state or s_logs_disp or raw_output_disp:
             tabs_list.append("📜 Logs & Output")
        else: # Ensure at least one tab exists
             if not tabs_list: tabs_list = ["Info"]

        # Create tabs
        tab_objects = st.tabs(tabs_list)
        tab_map = {name: tab for name, tab in zip(tabs_list, tab_objects)}


        # --- Charts Tab ---
        if "📊 Charts" in tab_map:
            with tab_map["📊 Charts"]:
                chart_title_prefix = f"Permutation {active_view_index}" if is_displaying_perm_view else ("Log View" if st.session_state.log_viewer_mode else "Original Run")
                st.markdown(f"#### Charts for: {chart_title_prefix}")
                # Pass the correct dataframes (m_df_disp, t_df_disp)
                display_pnl_chart(m_df_disp, chart_title_prefix)
                st.divider()
                display_position_chart(t_df_disp, chart_title_prefix)
                st.divider()
                display_norm_price_chart(m_df_disp, chart_title_prefix)
                st.divider()
                display_fill_dist_chart(m_df_disp, t_df_disp, chart_title_prefix, is_permutation_run=is_displaying_perm_view)


        # --- Permutation Test Tab ---
        if "🧪 Permutation Test" in tab_map:
            with tab_map["🧪 Permutation Test"]:
                st.subheader("Permutation Test Results")
                orig_pnl_from_state = perm_results.get('original_pnl')
                orig_pnl_display = f"{orig_pnl_from_state:.2f}" if pd.notna(orig_pnl_from_state) else "N/A (Run Failed?)"
                p_val = perm_results.get('p_value')
                valid_perm_pnls = perm_results.get('permuted_pnls', []) # Valid PNLs for histogram
                raw_perm_pnls = perm_results.get('permuted_pnls_raw', []) # All PNLs including NaNs

                col1, col2 = st.columns(2)
                col1.metric("Original Final PNL", orig_pnl_display)
                col2.metric("Permutation p-value", f"{p_val:.4f}" if pd.notna(p_val) else "N/A", help="Prob. of observing original PNL or higher under random market conditions (lower is better).")

                if valid_perm_pnls: # Check if the list of *valid* PNLs is not empty
                     st.markdown("###### Distribution of Permuted Final PNLs")
                     try:
                          # --- PROBLEM AREA ---
                          # The input to create_distplot should be a list of lists/arrays.
                          # valid_perm_pnls is currently just a flat list [509.0, 192.5]
                          # It needs to be wrapped in another list: [[509.0, 192.5]]
                          fig_hist = ff.create_distplot(
                              [valid_perm_pnls], # Wrap the list here!
                              ['Permuted PNLs'], # List of names for each data series
                              bin_size=None,
                              show_hist=True, show_rug=False, show_curve=False
                          )
                          # --- END PROBLEM AREA ---

                          fig_hist.update_layout(title="Distribution of Final PNLs from Permutations", xaxis_title="Final Total PNL", yaxis_title="Density", height=400, margin=dict(l=20,r=20,t=50,b=20))
                          if pd.notna(orig_pnl_from_state):
                              fig_hist.add_vline(x=orig_pnl_from_state, line_width=2, line_dash="dash", line_color="red", annotation_text="Original PNL", annotation_position="top right")
                          st.plotly_chart(fig_hist, use_container_width=True)
                     except Exception as hist_e:
                          # Provide more context in the error message
                          st.error(f"Could not create PNL distribution chart: {hist_e}")
                          st.error(f"Data passed to create_distplot: {valid_perm_pnls}") # Show the data
                          traceback.print_exc() # Print traceback in UI console
                else:
                     st.info("No valid permutation PNLs were recorded (all runs may have failed). Cannot plot distribution.")
                # Expander to show raw PNLs (including NaNs)
                with st.expander(f"View Raw PNLs for all {len(raw_perm_pnls)} Permutation Runs"):
                    st.dataframe(pd.Series(raw_perm_pnls, name="PNL", index=range(1, len(raw_perm_pnls) + 1)))

                # --- View Specific Permutation ---
                st.divider()
                st.subheader("View Specific Permutation Run Details")
                rerun_args = st.session_state.get('permutation_rerun_args')
                max_perms = len(raw_perm_pnls) if raw_perm_pnls else 0

                if max_perms == 0 or not rerun_args:
                    st.info("Run permutation test first or check for errors during runs.")
                else:
                    # Use number_input for selecting index
                    # Default value should be the currently viewed index if > 0, else 1
                    default_view_idx = st.session_state.view_permutation_index if st.session_state.view_permutation_index > 0 else 1
                    view_index_input = st.number_input(
                         f"Select Permutation Run to View (1 - {max_perms}):",
                         min_value=1, max_value=max_perms,
                         value=default_view_idx, step=1,
                         key="view_perm_input", # Use this key to track user input
                         help="Select a run number and click Load to see its detailed charts and logs."
                    )
                    if st.button("Load Selected Permutation Run", key="view_perm_button", use_container_width=True):
                         if view_index_input is not None and 1 <= view_index_input <= max_perms:
                             # Call the function to rerun and update state
                             rerun_and_view_permutation(view_index_input, rerun_args)
                             # Force rerun of the script to update the display sections based on new state
                             st.rerun()
                         else:
                             st.warning(f"Please enter a valid permutation number between 1 and {max_perms}.")

        # --- Logs & Output Tab ---
        if "📜 Logs & Output" in tab_map:
            with tab_map["📜 Logs & Output"]:
                st.subheader("Execution Logs")
                # Determine which logs/output to show based on view state
                logs_to_use = s_logs_disp
                output_content_to_use = raw_output_disp
                log_source_info = display_source_info # Reuse source info

                st.caption(f"Displaying logs for: {log_source_info}")

                if logs_to_use:
                    max_logs_to_show = 500 # Limit number of entries shown directly
                    display_logs_limited = logs_to_use # No limit for now, consider adding back if slow
                    # if len(logs_to_use) > max_logs_to_show:
                    #     st.warning(f"Showing only last {max_logs_to_show} log entries for performance.")
                    #     display_logs_limited = logs_to_use[-max_logs_to_show:]

                    # Display in reverse chronological order (newest first)
                    st.text(f"Showing {len(display_logs_limited)} log entries (newest first):")
                    for i, log_entry in enumerate(reversed(display_logs_limited)):
                         ts = log_entry.get("timestamp", "N/A")
                         sbox = log_entry.get("sandboxLog", "")
                         lamb = log_entry.get("lambdaLog", "")
                         # Only show expander if there's content
                         if sbox or lamb:
                             # Default to collapsed, maybe expand latest (i==0)?
                             with st.expander(f"Timestamp {ts}", expanded=(i < 5)): # Expand first 5
                                 if lamb: # Check if string is non-empty
                                     st.text("Trader Output (lambdaLog):")
                                     st.code(lamb, language=None)
                                 if sbox: # Check if string is non-empty
                                     st.text("Sandbox Messages (sandboxLog):")
                                     st.code(sbox, language=None)
                         # else: Skip empty entries entirely
                else:
                    st.info("No execution logs available for this view.")

                # Show Raw Output Log if enabled
                if st.session_state.show_output_log_state:
                     st.divider()
                     st.subheader("Raw Output Log Content")
                     if output_content_to_use:
                         st.text_area("Raw Output:", output_content_to_use, height=400, key="output_log_area")
                         # --- Download Button ---
                         try:
                              # Generate filename based on source
                              fname_trader = "log_view"
                              fname_log = "unknown_log"
                              if is_displaying_perm_view:
                                   fname_trader = st.session_state.permutation_rerun_args.get('trader_file', 'unknown_trader').replace('.py','')
                                   fname_log = st.session_state.permutation_rerun_args.get('data_files', ['unknown_log'])[0].replace('.log','')
                                   fname_suffix = f"_perm{st.session_state.view_permutation_index}"
                              elif st.session_state.log_viewer_mode:
                                   fname_trader = "log_view"
                                   fname_log = st.session_state.last_loaded_log_file.replace('.log','') if st.session_state.last_loaded_log_file else 'unknown_log'
                                   fname_suffix = ""
                              else: # Original backtest run
                                   fname_trader = st.session_state.last_run_trader_file.replace('.py','') if st.session_state.last_run_trader_file else 'unknown_trader'
                                   fname_log = selected_data_fnames[0].replace('.log','') if selected_data_fnames else 'unknown_log'
                                   fname_suffix = "_original"

                              download_filename = f"output_{fname_trader}_{fname_log}{fname_suffix}.log"

                              st.download_button(
                                  label="📥 Download Full Output Log",
                                  data=output_content_to_use,
                                  file_name=download_filename,
                                  mime="text/plain",
                                  use_container_width=True
                              )
                         except Exception as dl_err:
                              st.warning(f"Could not generate download link: {dl_err}")

                     else:
                         st.warning("No raw output content available.")

        # Fallback Info Tab if others are missing
        if "Info" in tab_map:
             with tab_map["Info"]:
                  st.info("No results to display. Please run a backtest or load a log file.")

# --- END OF app.py ---