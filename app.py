# app.py
import io
import os
import time
import traceback # For better error logging

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Assuming backtester module is in ./backtester/
from backtester import backtester as bt_module
from backtester import util, constants
from typing import List, Dict, Tuple, Optional

st.set_page_config(layout="wide")

script_dir = os.path.dirname(os.path.abspath(__file__))

# --- DEFINE ABSOLUTE PATHS FOR RESOURCES ---
TRADER_DIR_ABS = os.path.join(script_dir, "backtester", "traders")
DATA_DIR_ABS = os.path.join(script_dir, "data")
OUTPUT_DIR_ABS = os.path.join(script_dir, "output")
os.makedirs(OUTPUT_DIR_ABS, exist_ok=True)

# --- Helper Functions ---
# (parse_backtest_output, load_file_content, run_backtest remain the same)
@st.cache_data
def parse_backtest_output(output_string: str) -> Tuple[List[Dict], pd.DataFrame, pd.DataFrame]:
    """Parses the full backtester output string using the util function."""
    if not output_string:
        return [], pd.DataFrame(), pd.DataFrame()
    try:
        output_io = io.StringIO(output_string)
        sb_logs, market_df, trades_df = util._parse_data(output_io) # We still parse all parts

        if not market_df.empty and pd.api.types.is_float_dtype(market_df.index.dtype):
            market_df.index = market_df.index.astype(int)
        if not trades_df.empty and pd.api.types.is_float_dtype(trades_df.index.dtype):
            trades_df.index = trades_df.index.astype(int)

        # Return all parsed data, even if not all displayed
        return sb_logs, market_df, trades_df
    except Exception as e:
        st.error(f"Error parsing backtest output: {e}")
        print(f"--- Error Parsing Output ---"); print(output_string[:1000] + "..."); traceback.print_exc(); print(f"--- End Error ---")
        return [], pd.DataFrame(), pd.DataFrame()

@st.cache_data
def load_file_content(filepath: str) -> str:
    """Loads the raw content of any file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return f.read()
    except FileNotFoundError: return f"Error: File not found at {filepath}."
    except Exception as e: return f"Error loading file content from {filepath}: {e}"

def run_backtest(trader_filename: str, data_filenames: List[str], time_range: Tuple[int, int], bot_behavior: str, ignore_limits: bool):
    """ Creates and runs the backtester instance. """
    # ...(logic remains the same)...
    print("\n--- Running New Backtest ---"); print(f"Trader File: {trader_filename}"); print(f"Data Files: {data_filenames}"); print(f"Time Range: {time_range}"); print(f"Bot Behavior: {bot_behavior}"); print(f"Ignore Limits: {ignore_limits}")
    full_trader_path = os.path.join(TRADER_DIR_ABS, trader_filename); full_data_paths = [os.path.join(DATA_DIR_ABS, fname) for fname in data_filenames]
    if not os.path.exists(full_trader_path): st.error(f"Trader file not found: {full_trader_path}"); return None
    for p in full_data_paths:
        if not os.path.exists(p): st.error(f"Data file not found: {p}"); return None
    backtester_instance = None
    try:
        print("Creating Backtester instance..."); start_init_time = time.time()
        backtester_instance = bt_module.Backtester(trader_fname=trader_filename, data_fnames=data_filenames, timerange=time_range, bot_behavior=bot_behavior, ignore_limits=ignore_limits)
        end_init_time = time.time(); print(f"Instance created in {end_init_time - start_init_time:.2f}s.")
        print("Starting simulation run..."); start_run_time = time.time()
        backtester_instance.run()
        end_run_time = time.time(); print(f"Simulation completed in {end_run_time - start_run_time:.2f}s.")
        return backtester_instance
    except Exception as e: st.error(f"Error during backtest: {e}"); st.code(traceback.format_exc()); return None

# --- Session State Init ---
# ... (remains the same) ...
if 'log_viewer_mode' not in st.session_state: st.session_state.log_viewer_mode = False
if 'parsed_log_data' not in st.session_state: st.session_state.parsed_log_data = ([], pd.DataFrame(), pd.DataFrame(), "")
if 'last_loaded_log_file' not in st.session_state: st.session_state.last_loaded_log_file = None
if 'show_output_log_state' not in st.session_state:
    st.session_state.show_output_log_state = False
def clear_parsed_data(): st.session_state.parsed_log_data = ([], pd.DataFrame(), pd.DataFrame(), ""); st.session_state.last_loaded_log_file = None
def calculate_row_vwap(row):
    """Calculates VWAP for a single row using up to 3 levels."""
    n = 0.0
    d = 0.0
    try:
        # Bids
        for i in range(1, 4):
            bp, bv = row.get(f'bid_price_{i}', np.nan), row.get(f'bid_volume_{i}', np.nan)
            if pd.notna(bp) and pd.notna(bv) and bv > 0:
                n += bp * bv
                d += bv
        # Asks
        for i in range(1, 4):
            ap, av = row.get(f'ask_price_{i}', np.nan), row.get(f'ask_volume_{i}', np.nan)
            if pd.notna(ap) and pd.notna(av) and av > 0:
                n += ap * av # Use positive volume for asks too
                d += av
        return n / d if d > 0 else np.nan
    except Exception: # Catch potential type errors during calculation
        return np.nan
    
# --- UI Layout ---
st.title("🌊 Prosperity Algorithm Backtester & Log Viewer 📈")
leftcol, rightcol = st.columns([2, 5], gap="medium")

with leftcol:
    # ... (Configuration elements remain the same: Mode Selection, Trader, Data, Settings, Time Range, Run Button) ...
    st.header("Configuration"); is_viewer_mode = st.checkbox("View Log File Only (No Trader)", key="log_viewer_mode", on_change=clear_parsed_data); mode_description = "Run a full backtest simulation." if not is_viewer_mode else "Parse and display data from a single log file."; st.caption(mode_description); st.divider()
    st.subheader("1. Select Trader" if not is_viewer_mode else "1. Trader Selection (Disabled)")
    try: trader_files = sorted([f for f in os.listdir(TRADER_DIR_ABS) if os.path.isfile(os.path.join(TRADER_DIR_ABS, f)) and f.endswith(".py")]);
    except FileNotFoundError: st.error(f"Trader directory not found: `{TRADER_DIR_ABS}`"); trader_files = []
    selected_trader_fname = None
    if not trader_files and not is_viewer_mode: st.warning(f"No Python traders found in `{TRADER_DIR_ABS}`.")
    else:
        selected_trader_fname = st.selectbox("Trader File:", trader_files, key="trader_select", label_visibility="collapsed", disabled=is_viewer_mode)
        if selected_trader_fname and not is_viewer_mode:
            trader_code_path = os.path.join(TRADER_DIR_ABS, selected_trader_fname); trader_code = load_file_content(trader_code_path)
            with st.expander("View Trader Code"): st.code(trader_code, language="python")
    st.subheader("2. Select Data Log(s)" if not is_viewer_mode else "2. Select Log File")
    try: data_files = sorted([f for f in os.listdir(DATA_DIR_ABS) if os.path.isfile(os.path.join(DATA_DIR_ABS, f)) and f.endswith(".log")])
    except FileNotFoundError: st.error(f"Data directory not found: `{DATA_DIR_ABS}`"); data_files = []
    selected_data_fnames = []; selected_log_viewer_file = None
    if not data_files: st.warning(f"No log files (`.log`) found in `{DATA_DIR_ABS}`.")
    else:
        if is_viewer_mode: selected_log_viewer_file = st.selectbox("Log File:", data_files, key="log_viewer_select", label_visibility="collapsed");
        else: selected_data_fnames = st.multiselect("Log Files:", data_files, default=data_files[0] if data_files else None, key="data_select", label_visibility="collapsed")
        if selected_log_viewer_file: selected_data_fnames = [selected_log_viewer_file] # Use single file if viewer mode
    st.subheader("3. Settings" if not is_viewer_mode else "3. Settings (Backtest Only)")
    col_a, col_b = st.columns(2)
    with col_a: bot_behavior_options = ["none", "eq", "lt", "lte"]; bot_behavior = st.selectbox("Bot Matching:", bot_behavior_options, index=3, key="bot_behavior", help="Rule for matching leftover orders against inferred bot liquidity.", disabled=is_viewer_mode)
    with col_b: ignore_limits_checkbox = st.checkbox("Ignore Limits", value=False, key="ignore_limits", help="If checked, position limits are not enforced.", disabled=is_viewer_mode)
    min_time_data, max_time_data = 0, 199900; file_for_time_range = None
    if is_viewer_mode and selected_log_viewer_file: file_for_time_range = selected_log_viewer_file
    elif not is_viewer_mode and selected_data_fnames: file_for_time_range = selected_data_fnames[0]
    if file_for_time_range:
         try:
              temp_data_path = os.path.join(DATA_DIR_ABS, file_for_time_range);
              with open(temp_data_path, 'r', encoding='utf-8') as f: _, temp_mkt, _ = util._parse_data(f)
              if not temp_mkt.empty and pd.api.types.is_numeric_dtype(temp_mkt.index): min_time_data = int(temp_mkt.index.min()); max_time_data = int(temp_mkt.index.max())
         except Exception as e: print(f"Warning: Could not determine time range from {file_for_time_range}: {e}")
    time_range_values = st.slider("Time Range Filter:", min_value=min_time_data, max_value=max_time_data, value=(min_time_data, max_time_data), step=100, key="timerange")
    st.divider(); button_label = "📊 Load & View Log" if is_viewer_mode else "🚀 Run Backtest"; button_disabled = (is_viewer_mode and not selected_log_viewer_file) or (not is_viewer_mode and (not selected_trader_fname or not selected_data_fnames))
    run_button_pressed = st.button(button_label, use_container_width=True, type="primary", disabled=button_disabled)
    st.checkbox("Show Full Output Log in Results", key="show_output_log_state", value=False)

# --- Main Execution & Display Logic ---
backtester_instance = None

if run_button_pressed:
    st.session_state.parsed_log_data = ([], pd.DataFrame(), pd.DataFrame(), "")
    if is_viewer_mode:
        if selected_log_viewer_file:
            log_file_path = os.path.join(DATA_DIR_ABS, selected_log_viewer_file); st.session_state.last_loaded_log_file = selected_log_viewer_file; print(f"--- Loading Log File: {log_file_path} ---")
            try:
                with st.spinner(f"Loading and parsing **{selected_log_viewer_file}**..."):
                    raw_content = load_file_content(log_file_path)
                    if raw_content.startswith("Error:"): st.error(raw_content)
                    else:
                        s_logs, m_df, t_df = parse_backtest_output(raw_content)
                        m_df_filtered = m_df[(m_df.index >= time_range_values[0]) & (m_df.index <= time_range_values[1])]
                        t_df_filtered = t_df[(t_df.index >= time_range_values[0]) & (t_df.index <= time_range_values[1])]
                        # We still parse s_logs but won't display them in a dedicated tab
                        st.session_state.parsed_log_data = (s_logs, m_df_filtered, t_df_filtered, raw_content) # Store un-filtered s_logs if needed later
                        st.success(f"Loaded and parsed {selected_log_viewer_file}.")
            except Exception as e: st.error(f"Failed to load/parse log: {e}"); st.code(traceback.format_exc()); st.session_state.parsed_log_data = ([], pd.DataFrame(), pd.DataFrame(), "")
        else: st.warning("Please select a log file.")
    else: # Backtest Mode
        if selected_trader_fname and selected_data_fnames:
            with st.spinner(f"Running backtest: **{selected_trader_fname}**..."):
                 backtester_instance = run_backtest(selected_trader_fname, selected_data_fnames, time_range_values, bot_behavior, ignore_limits_checkbox)
            if backtester_instance and hasattr(backtester_instance, 'output') and backtester_instance.output: st.success("Backtest simulation completed.")
            elif backtester_instance: st.warning("Backtest finished but produced no output.")
        else: st.warning("Please select trader and data file(s).")


# --- Display Results ---
with rightcol:
    st.header("Results")
    display_data = None; raw_output_content = ""; source_description = ""
    sandbox_logs_parsed, market_data_parsed, trade_history_parsed = [], pd.DataFrame(), pd.DataFrame() # Init vars

    if is_viewer_mode and st.session_state.parsed_log_data[1] is not None and not st.session_state.parsed_log_data[1].empty:
        sandbox_logs_parsed, market_data_parsed, trade_history_parsed, raw_output_content = st.session_state.parsed_log_data
        display_data = True; source_description = f"Displaying data from: **{st.session_state.last_loaded_log_file or 'N/A'}**"
    elif backtester_instance and hasattr(backtester_instance, 'output') and backtester_instance.output:
        try:
            with st.spinner("Parsing backtest output..."):
                raw_output_content = backtester_instance.output
                # Still parse all parts, even if not displayed
                sandbox_logs_parsed, market_data_parsed, trade_history_parsed = parse_backtest_output(raw_output_content)
                display_data = True; source_description = f"Displaying backtest results for: **{selected_trader_fname or 'N/A'}**"
        except Exception as parse_err: st.error(f"Failed to parse output: {parse_err}"); display_data = False
    elif run_button_pressed and not display_data: pass
    else: st.info("Configure settings and run backtest or load log file."); display_data = False

    # --- Display Area ---
    if display_data:
        st.caption(source_description)

        # --- CHANGE: Only create the Charts tab ---
        tab1 = st.tabs(["📊 Charts"])[0] # Get the single tab object
        # --- END CHANGE ---

        with tab1: # Changed from tab1
            # --- Charts Section (PNL, Position, Market Price) ---
            # (Keep the latest correct plotting logic for PNL, Position, Price)
            # Example for PNL (ensure it uses the product-specific logic)
            st.markdown("##### Profit and Loss (PNL)")
            required_cols = ['product', 'profit_and_loss']
            if not all(col in market_data_parsed.columns for col in required_cols): st.warning(f"Required PNL columns not found.")
            elif market_data_parsed.empty: st.warning("No Activity Log data for PNL chart.")
            else:
                pnl_plot_data = market_data_parsed.copy(); pnl_plot_data.index.name = 'timestamp'; pnl_plot_data.reset_index(inplace=True)
                pnl_plot_data.dropna(subset=['profit_and_loss', 'product', 'timestamp'], inplace=True)
                if pnl_plot_data.empty: st.warning("No valid PNL data points remain.")
                else:
                    fig_pnl = px.line(pnl_plot_data, x='timestamp', y='profit_and_loss', color='product', title="Individual Product PNL Over Time & Total")
                    fig_pnl.update_layout(hovermode="x unified", margin=dict(l=0, r=0, t=30, b=20), xaxis_title=None, yaxis_title="PNL", legend_title="Product", height=350)
                    fig_pnl.update_traces(hovertemplate="Product: <b>%{fullData.name}</b><br>Timestamp: %{x}<br>PNL: %{y:.2f}<extra></extra>")
                    total_pnl_trace_data = pnl_plot_data.groupby('timestamp')['profit_and_loss'].sum().reset_index()
                    if not total_pnl_trace_data.empty:
                         fig_pnl.add_scatter(x=total_pnl_trace_data['timestamp'], y=total_pnl_trace_data['profit_and_loss'], mode='lines', name='Overall Total PNL', line=dict(color='white', width=4), hovertemplate="<b>Overall Total PNL</b><br>Timestamp: %{x}<br>Total PNL: %{y:.2f}<extra></extra>")
                    st.plotly_chart(fig_pnl, use_container_width=True)

            # Position Chart
            st.markdown("##### Positions ('SUBMISSION' Trades)")
            if not trade_history_parsed.empty:
                 player_trades = trade_history_parsed[(trade_history_parsed['buyer'] == "SUBMISSION") | (trade_history_parsed['seller'] == "SUBMISSION")].copy()
                 if not player_trades.empty:
                      player_trades['quantity'] = pd.to_numeric(player_trades['quantity'], errors='coerce').fillna(0).astype(int); player_trades['price'] = pd.to_numeric(player_trades['price'], errors='coerce').fillna(0).astype(int)
                      player_trades['signed_quantity'] = player_trades.apply(lambda row: row['quantity'] if row['buyer'] == 'SUBMISSION' else -row['quantity'], axis=1); player_trades.sort_index(inplace=True)
                      pos_frames = []; products = sorted(player_trades['symbol'].unique())
                      for product in products: prod_trades = player_trades[player_trades['symbol'] == product].copy(); prod_trades['position'] = prod_trades['signed_quantity'].cumsum(); pos_frames.append(prod_trades)
                      if pos_frames:
                           all_positions = pd.concat(pos_frames).sort_index(); all_positions['timestamp'] = all_positions.index
                           fig_pos = px.line(all_positions, x='timestamp', y='position', color='symbol', title="Player Positions (from SUBMISSION trades)")
                           fig_pos.update_layout(hovermode="x unified", margin=dict(l=0, r=0, t=30, b=20), xaxis_title=None, yaxis_title="Position", legend_title="Product", height=300)
                           fig_pos.update_traces(hovertemplate="<b>%{fullData.name}</b><br>Pos: %{y}<extra></extra>"); st.plotly_chart(fig_pos, use_container_width=True)
                 else: st.info("No 'SUBMISSION' trades found.")
            else: st.info("No trade history data for position chart.")

            # Price Chart (VWAP or Mid Price)
            vwap_calculated = False
            if not market_data_parsed.empty:
                 print("DEBUG: Attempting row-wise VWAP calculation...")
                 # Ensure necessary columns are numeric first (handle potential strings/objects)
                 price_cols = [f'{p}_{i}' for p in ['bid_price', 'ask_price'] for i in range(1, 4)]
                 vol_cols = [f'{p}_{i}' for p in ['bid_volume', 'ask_volume'] for i in range(1, 4)]
                 cols_to_convert = [c for c in price_cols + vol_cols if c in market_data_parsed.columns]
                 for col in cols_to_convert:
                     # Only convert if not already numeric to avoid unnecessary operations
                     if not pd.api.types.is_numeric_dtype(market_data_parsed[col]):
                          market_data_parsed[col] = pd.to_numeric(market_data_parsed[col], errors='coerce')

                 try:
                      # Apply the function row by row
                      vwap_series = market_data_parsed.apply(calculate_row_vwap, axis=1)
                      # Add VWAP column and ffill/bfill per product
                      market_data_parsed['vwap'] = vwap_series
                      market_data_parsed['vwap'] = market_data_parsed.groupby('product')['vwap'].ffill().bfill()
                      vwap_calculated = True
                      print("DEBUG: Row-wise VWAP calculation successful.")
                 except Exception as e:
                      print(f"ERROR: Row-wise VWAP calculation failed: {e}")
                      traceback.print_exc()
                      if 'vwap' not in market_data_parsed.columns:
                           market_data_parsed['vwap'] = np.nan # Ensure column exists even if failed

           # app.py -> inside rightcol / if display_data / with tab1 ...

            # --- Log Price Chart (VWAP or Mid - NORMALIZED) ---
            st.markdown("##### Market Prices (Normalized Change Since Start)") # Changed title
            price_col_to_use = None
            y_axis_title = "Price Change" # Changed y-axis label
            price_data_normalized = pd.DataFrame() # Initialize empty df

            # Prefer VWAP if successfully calculated and not all NaN
            if vwap_calculated and 'vwap' in market_data_parsed.columns and not market_data_parsed['vwap'].isna().all():
                price_col_to_use = 'vwap'
                y_axis_title = "Log VWAP Change ($)"
            elif 'mid_price' in market_data_parsed.columns and not market_data_parsed['mid_price'].isna().all():
                 if not pd.api.types.is_numeric_dtype(market_data_parsed['mid_price']): market_data_parsed['mid_price'] = pd.to_numeric(market_data_parsed['mid_price'], errors='coerce')
                 price_col_to_use = 'mid_price'
                 y_axis_title = "Mid Price Change ($)"

            if price_col_to_use and not market_data_parsed.empty:
                 price_data = market_data_parsed[['product', price_col_to_use]].copy()
                 price_data.index.name = 'timestamp' # Ensure index is named
                 price_data.dropna(subset=[price_col_to_use, 'product'], inplace=True)

                 if not price_data.empty:
                      # --- Normalize Price Data ---
                      def normalize_group(group):
                          first_valid_index = group[price_col_to_use].first_valid_index()
                          if first_valid_index is None:
                              group['normalized_price'] = np.nan
                          else:
                              first_price = group.loc[first_valid_index, price_col_to_use]
                              group['normalized_price'] = group[price_col_to_use] - first_price
                          return group

                      price_data_normalized = price_data.groupby('product', group_keys=False).apply(normalize_group)

                      # --- FIX: Drop timestamp COLUMN if it exists, then reset index ---
                      if 'timestamp' in price_data_normalized.columns:
                          price_data_normalized_for_plot = price_data_normalized.drop(columns=['timestamp']).reset_index()
                      else:
                          price_data_normalized_for_plot = price_data_normalized.reset_index()
                      # --- END FIX ---


                      price_data_normalized_for_plot.dropna(subset=['normalized_price'], inplace=True)


                 if not price_data_normalized_for_plot.empty:
                      # Plot using the reset index df
                      fig_price_norm = px.line(price_data_normalized_for_plot,
                                              x='timestamp',
                                              y='normalized_price',
                                              color='product')

                      # ... (rest of layout update, hover data prep, plotting remains the same) ...
                      min_norm_y = price_data_normalized_for_plot['normalized_price'].min(); max_norm_y = price_data_normalized_for_plot['normalized_price'].max()
                      yrange_min = min_norm_y - (abs(min_norm_y)*0.1+0.1) if pd.notna(min_norm_y) else -1; yrange_max = max_norm_y + (abs(max_norm_y)*0.1+0.1) if pd.notna(max_norm_y) else 1
                      fig_price_norm.update_layout(title=y_axis_title, hovermode="x unified", margin=dict(l=0,r=0,t=30,b=20), xaxis_title=None, yaxis_title="Log Price Change ($)", legend_title="Product", height=300, yaxis_zeroline=True, yaxis_zerolinecolor='Gray', yaxis_zerolinewidth=1, yaxis_range=[yrange_min, yrange_max])

                      price_data_orig_for_merge = price_data.reset_index()
                      price_data_normalized_for_plot_with_orig = pd.merge(
                          price_data_normalized_for_plot[['timestamp', 'product', 'normalized_price']],
                          price_data_orig_for_merge[['timestamp', 'product', price_col_to_use]],
                          on=['timestamp', 'product'], how='left'
                      )
                      price_data_normalized_for_plot_with_orig.sort_values(['product', 'timestamp'], inplace=True)
                      fig_price_norm.update_traces(
                          hovertemplate=(f"<b>%{{fullData.name}}</b><br>Timestamp: %{{x}}<br>Change: %{{y:+.2f}}<br>{price_col_to_use.upper()}: %{{customdata[0]:.2f}}<extra></extra>"),
                          customdata=price_data_normalized_for_plot_with_orig[[price_col_to_use]].values
                      )
                      st.plotly_chart(fig_price_norm, use_container_width=True)

                 else: st.warning(f"No valid data for normalized prices using {price_col_to_use}.")
            else: st.warning("Could not find suitable columns ('vwap' or 'mid_price') for normalized price chart.")

            # --- Combined Fill Distance Chart ---
            # ... (Fill distance chart logic remains the same) ...

            # --- Combined Fill Distance Chart ---
            st.divider()
            st.markdown("##### Fill Distance from VWAP (Submission Trades)")
            # Check if VWAP was calculated successfully for this chart
            if not vwap_calculated or 'vwap' not in market_data_parsed.columns or market_data_parsed['vwap'].isna().all(): # Check flag and column
                 st.warning("VWAP data is missing or could not be calculated. Cannot generate fill distance chart.")
            elif trade_history_parsed.empty:
                 st.warning("Trade data is missing, cannot generate fill distance chart.")
            else:
                 # ... (rest of Fill Distance chart logic remains the same, it uses the 'vwap' column) ...
                 player_trades = trade_history_parsed[ (trade_history_parsed['buyer'] == "SUBMISSION") | (trade_history_parsed['seller'] == "SUBMISSION") ].copy(); player_trades.index.name = 'timestamp'; player_trades.reset_index(inplace=True)
                 if player_trades.empty: st.info("No SUBMISSION trades found.")
                 else:
                      def get_fill_type(row): # ... (fill type logic) ...
                           if row['buyer']=='SUBMISSION' and row['seller'] in ['ExplicitBook','InferredBot']: return row['seller']
                           elif row['seller']=='SUBMISSION' and row['buyer'] in ['ExplicitBook','InferredBot']: return row['buyer']
                           else: return 'Other/Unknown'
                      player_trades['fill_type'] = player_trades.apply(get_fill_type, axis=1); player_trades = player_trades[player_trades['fill_type'] != 'Other/Unknown']; player_trades['quantity'] = pd.to_numeric(player_trades['quantity'], errors='coerce').fillna(0)
                      vwap_data_for_merge = market_data_parsed[['product', 'vwap']].reset_index() # Use updated VWAP
                      merged_trades = pd.merge(player_trades, vwap_data_for_merge, left_on=['timestamp', 'symbol'], right_on=['timestamp', 'product'], how='left')
                      if 'product' in merged_trades.columns: merged_trades.drop(columns=['product'], inplace=True)
                      merged_trades.dropna(subset=['vwap', 'price'], inplace=True)
                      if merged_trades.empty: st.warning("No valid trades remaining after merging with VWAP data.")
                      else:
                           merged_trades['distance'] = abs(merged_trades['price'] - merged_trades['vwap']); max_bucket = 5
                           merged_trades['distance_bucket'] = np.floor(merged_trades['distance']).astype(int).clip(upper=max_bucket)
                           bucket_labels = {i: f"${i}-{i+1-0.01:.2f}" for i in range(max_bucket)}; bucket_labels[max_bucket] = f"${max_bucket}+"
                           merged_trades['distance_label'] = merged_trades['distance_bucket'].map(bucket_labels); label_order = [bucket_labels[i] for i in range(max_bucket + 1)]
                           overall_summary = merged_trades.groupby(['distance_label', 'fill_type'])['quantity'].sum().reset_index(); overall_summary['display_group'] = 'Overall'
                           product_summary = merged_trades.groupby(['distance_label', 'fill_type', 'symbol'])['quantity'].sum().reset_index(); product_summary.rename(columns={'symbol': 'display_group'}, inplace=True)
                           combined_summary = pd.concat([overall_summary, product_summary], ignore_index=True); combined_summary.rename(columns={'quantity': 'total_volume'}, inplace=True)
                           product_names = sorted(merged_trades['symbol'].unique()); display_group_order = ['Overall'] + product_names
                           if combined_summary.empty: st.info("No volume data found.")
                           else:
                                fig_fill_dist_combined = px.bar(combined_summary, x='distance_label', y='total_volume', color='fill_type', facet_col='display_group', barmode='group', title="Volume Filled by Distance from VWAP (Overall & Per Product)", labels={'distance_label': 'Abs Distance from VWAP ($)', 'total_volume': 'Total Volume Filled', 'fill_type': 'Fill Source', 'display_group': 'Group'}, category_orders={"distance_label": label_order, "display_group": display_group_order})
                                fig_fill_dist_combined.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=400); fig_fill_dist_combined.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                                st.plotly_chart(fig_fill_dist_combined, use_container_width=True)



            # app.py

# ... inside rightcol / if display_data / with tab1 ...

            # (Existing charts: PNL, Position, Normalized Price, Fill Distance)

            # --- Trade History Table with VWAP Distance ---
            st.divider()
            st.subheader("Trade History Details")

            # Check prerequisites: need trades and calculated VWAP
            vwap_available_for_trades = ('vwap' in market_data_parsed.columns and not market_data_parsed['vwap'].isna().all())

            if trade_history_parsed.empty:
                st.info("No trade history data available.")
            elif not vwap_available_for_trades:
                st.warning("VWAP data not available, cannot calculate distance. Displaying basic trade history.")
                # Display basic table without distance
                trade_history_display_basic = trade_history_parsed.sort_index().reset_index()
                def highlight_submission_basic(row): # Basic highlighter
                     color = 'lightgreen' if row['buyer'] == 'SUBMISSION' or row['seller'] == 'SUBMISSION' else ''
                     return [f'background-color: {color}'] * len(row)
                st.dataframe(trade_history_display_basic.style.apply(highlight_submission_basic, axis=1), height=400, hide_index=True)
            else:
                # VWAP is available, proceed with distance calculation
                try:
                    # 1. Prepare Trade Data (reset index)
                    trade_history_display = trade_history_parsed.sort_index().reset_index()
                    # Ensure price/quantity are numeric
                    trade_history_display['price'] = pd.to_numeric(trade_history_display['price'], errors='coerce')
                    trade_history_display['quantity'] = pd.to_numeric(trade_history_display['quantity'], errors='coerce')
                    trade_history_display.dropna(subset=['price', 'quantity'], inplace=True) # Drop rows where conversion failed


                    # 2. Prepare VWAP Data for Merge
                    # Select only needed columns and reset index
                    vwap_data_for_merge = market_data_parsed[['product', 'vwap']].reset_index()
                    vwap_data_for_merge.dropna(subset=['vwap'], inplace=True) # Ensure we only merge valid VWAPs


                    # 3. Merge VWAP onto Trades
                    trade_history_with_vwap = pd.merge(
                        trade_history_display,
                        vwap_data_for_merge,
                        left_on=['timestamp', 'symbol'],
                        right_on=['timestamp', 'product'],
                        how='left' # Keep all trades, add VWAP where available
                    )
                    # Drop the redundant 'product' column from VWAP data
                    if 'product' in trade_history_with_vwap.columns:
                         trade_history_with_vwap.drop(columns=['product'], inplace=True)

                    # 4. Calculate Distance
                    # Calculate only where VWAP is not NaN (due to left merge or original NaNs)
                    trade_history_with_vwap['Dist from VWAP'] = np.where(
                         pd.notna(trade_history_with_vwap['vwap']),
                         abs(trade_history_with_vwap['price'] - trade_history_with_vwap['vwap']),
                         np.nan # Assign NaN if VWAP was missing
                    )

                    # 5. Prepare Final Columns for Display
                    # Select and order columns, maybe drop the temporary VWAP column
                    cols_to_display = ['timestamp', 'symbol', 'price', 'quantity', 'buyer', 'seller', 'Dist from VWAP']
                    # Keep only columns that actually exist in the dataframe
                    cols_to_display = [col for col in cols_to_display if col in trade_history_with_vwap.columns]
                    trade_history_final_display = trade_history_with_vwap[cols_to_display]

                    # 6. Apply Styling (including highlighting)
                    def highlight_submission_dist(row):
                        color = 'lightgreen' if row.get('buyer') == 'SUBMISSION' or row.get('seller') == 'SUBMISSION' else ''
                        return [f'background-color: {color}'] * len(row)

                    st.dataframe(
                        trade_history_final_display.style.apply(highlight_submission_dist, axis=1)\
                                                        .format({"Dist from VWAP": "{:.2f}"}), # Format distance nicely
                        height=400,
                        hide_index=True # Hide the default 0, 1, ... index
                    )
                except Exception as e:
                     st.error(f"Error generating trade history with distance: {e}")
                     traceback.print_exc()
                     # Fallback to basic table on error
                     st.warning("Displaying basic trade history due to error.")
                     trade_history_display_basic = trade_history_parsed.sort_index().reset_index()
                     def highlight_submission_basic(row): color = 'lightgreen' if row['buyer'] == 'SUBMISSION' or row['seller'] == 'SUBMISSION' else ''; return [f'background-color: {color}'] * len(row)
                     st.dataframe(trade_history_display_basic.style.apply(highlight_submission_basic, axis=1), height=400, hide_index=True)

            # --- End Trade History Table ---
            if st.session_state.show_output_log_state:
                st.divider()
            output_label = "Full Backtest Output Log" if not is_viewer_mode else "Raw Log File Content"
            st.subheader(output_label)
            if raw_output_content:
                st.text_area("Log Content:", raw_output_content, height=600, key="output_log_area")
            else:
                st.warning("No output content available to display.")
    # --- End Tab1 ---

    # (rest of app.py)