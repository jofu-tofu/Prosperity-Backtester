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
from typing import List, Dict, Tuple, Optional # Added Optional

st.set_page_config(layout="wide")

script_dir = os.path.dirname(os.path.abspath(__file__))

# --- DEFINE ABSOLUTE PATHS FOR RESOURCES ---
TRADER_DIR_ABS = os.path.join(script_dir, "backtester", "traders")
DATA_DIR_ABS = os.path.join(script_dir, "data")
OUTPUT_DIR_ABS = os.path.join(script_dir, "output")
os.makedirs(OUTPUT_DIR_ABS, exist_ok=True)

# --- Helper Functions ---

@st.cache_data
def parse_backtest_output(output_string: str) -> Tuple[List[Dict], pd.DataFrame, pd.DataFrame]:
    """Parses the full backtester output string using the util function."""
    # ...(rest of the function remains the same)...
    if not output_string:
        return [], pd.DataFrame(), pd.DataFrame()
    try:
        output_io = io.StringIO(output_string)
        # Ensure util._parse_data is accessible and works as expected
        sb_logs, market_df, trades_df = util._parse_data(output_io)

        # Convert timestamp index to int if it was parsed as float
        if not market_df.empty and pd.api.types.is_float_dtype(market_df.index.dtype):
            market_df.index = market_df.index.astype(int)

        if not trades_df.empty and pd.api.types.is_float_dtype(trades_df.index.dtype):
            trades_df.index = trades_df.index.astype(int)

        return sb_logs, market_df, trades_df
    except Exception as e:
        st.error(f"Error parsing backtest output: {e}")
        print(f"--- Error Parsing Output ---")
        print(output_string[:1000] + "...") # Print beginning of output for context
        traceback.print_exc() # Print full traceback to console
        print(f"--- End Error ---")
        return [], pd.DataFrame(), pd.DataFrame()

@st.cache_data
def load_file_content(filepath: str) -> str:
    """Loads the raw content of any file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f: # Added encoding
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found at {filepath}."
    except Exception as e:
        return f"Error loading file content from {filepath}: {e}"

# Remove caching from run_backtest if trader code changes should always trigger a rerun
# @st.cache_resource(ttl=3600)
def run_backtest(trader_filename: str, data_filenames: List[str], time_range: Tuple[int, int], bot_behavior: str, ignore_limits: bool):
    """
    Creates and runs the backtester instance.
    Expects filenames relative to their respective directories.
    """
    # ...(rest of the function remains the same)...
    print("\n--- Running New Backtest ---")
    print(f"Trader File: {trader_filename}")
    print(f"Data Files: {data_filenames}")
    print(f"Time Range: {time_range}")
    print(f"Bot Behavior: {bot_behavior}")
    print(f"Ignore Limits: {ignore_limits}")

    full_trader_path = os.path.join(TRADER_DIR_ABS, trader_filename)
    full_data_paths = [os.path.join(DATA_DIR_ABS, fname) for fname in data_filenames]

    if not os.path.exists(full_trader_path):
        st.error(f"Trader file not found at expected path: {full_trader_path}")
        return None
    for p in full_data_paths:
        if not os.path.exists(p):
            st.error(f"Data file not found at expected path: {p}")
            return None

    backtester_instance = None # Initialize outside try
    try:
        print("Creating Backtester instance...")
        start_init_time = time.time()
        backtester_instance = bt_module.Backtester(
            trader_fname=trader_filename, # Pass just the filename
            data_fnames=data_filenames,   # Pass just the filenames
            timerange=time_range,
            bot_behavior=bot_behavior,
            ignore_limits=ignore_limits,
        )
        end_init_time = time.time()
        print(f"Backtester instance created in {end_init_time - start_init_time:.2f} seconds.")

        # --- Run the simulation ---
        print("Starting backtester run simulation...")
        start_run_time = time.time()
        backtester_instance.run() # Run the simulation
        end_run_time = time.time()
        print(f"Backtester run simulation completed in {end_run_time - start_run_time:.2f} seconds.")
        # --- Simulation finished ---

        return backtester_instance # Return the instance which now *should* have .output populated

    except Exception as e:
        st.error(f"Error during backtest execution: {e}")
        st.code(traceback.format_exc())
        return None # Return None on error

# --- Initialize Session State ---
if 'log_viewer_mode' not in st.session_state:
    st.session_state.log_viewer_mode = False
if 'parsed_log_data' not in st.session_state:
    # Store tuple: (sandbox_logs, market_df, trades_df, raw_content)
    st.session_state.parsed_log_data = ([], pd.DataFrame(), pd.DataFrame(), "")
if 'last_loaded_log_file' not in st.session_state:
    st.session_state.last_loaded_log_file = None


# --- UI Layout ---
st.title("🌊 Prosperity Algorithm Backtester & Log Viewer 📈")
leftcol, rightcol = st.columns([2, 5], gap="medium")

# Clear parsed data if switching modes
def clear_parsed_data():
    st.session_state.parsed_log_data = ([], pd.DataFrame(), pd.DataFrame(), "")
    st.session_state.last_loaded_log_file = None

with leftcol:
    st.header("Configuration")

    # --- Mode Selection ---
    is_viewer_mode = st.checkbox(
        "View Log File Only (No Trader)",
        key="log_viewer_mode",
        on_change=clear_parsed_data # Clear data when mode changes
    )
    mode_description = "Run a full backtest simulation with a selected trader." if not is_viewer_mode else "Parse and display data from a single existing log file."
    st.caption(mode_description)
    st.divider()

    # --- Trader Selection (Disabled in Viewer Mode) ---
    st.subheader("1. Select Trader" if not is_viewer_mode else "1. Trader Selection (Disabled)")
    try:
        trader_files = sorted([
            f for f in os.listdir(TRADER_DIR_ABS)
            if os.path.isfile(os.path.join(TRADER_DIR_ABS, f)) and f.endswith(".py")
        ])
    except FileNotFoundError:
        st.error(f"Trader directory not found: `{TRADER_DIR_ABS}`")
        trader_files = []

    if not trader_files and not is_viewer_mode:
         st.warning(f"No Python traders found in `{TRADER_DIR_ABS}`.")
         selected_trader_fname = None
    else:
         selected_trader_fname = st.selectbox(
             "Trader File:",
             trader_files,
             key="trader_select",
             label_visibility="collapsed",
             disabled=is_viewer_mode # Disable in viewer mode
         )
         # Display trader code only if a trader is selected and not in viewer mode
         if selected_trader_fname and not is_viewer_mode:
             trader_code_path = os.path.join(TRADER_DIR_ABS, selected_trader_fname)
             trader_code = load_file_content(trader_code_path) # Use generic loader
             with st.expander("View Trader Code"):
                 st.code(trader_code, language="python")

    # --- Data Selection (Single file in Viewer Mode) ---
    st.subheader("2. Select Data Log(s)" if not is_viewer_mode else "2. Select Log File")
    try:
        data_files = sorted([
            f for f in os.listdir(DATA_DIR_ABS)
            if os.path.isfile(os.path.join(DATA_DIR_ABS, f)) and f.endswith(".log")
        ])
    except FileNotFoundError:
        st.error(f"Data directory not found: `{DATA_DIR_ABS}`")
        data_files = []

    selected_data_fnames = [] # Initialize
    selected_log_viewer_file = None # Initialize

    if not data_files:
         st.warning(f"No log files (`.log`) found in `{DATA_DIR_ABS}`.")
    else:
        if is_viewer_mode:
            selected_log_viewer_file = st.selectbox(
                 "Log File:",
                 data_files,
                 key="log_viewer_select",
                 label_visibility="collapsed"
            )
            if selected_log_viewer_file:
                 selected_data_fnames = [selected_log_viewer_file] # Keep structure consistent downstream if needed
        else:
            selected_data_fnames = st.multiselect(
                 "Log Files (First is primary for market data):",
                 data_files,
                 default=data_files[0] if data_files else None,
                 key="data_select",
                 label_visibility="collapsed"
            )

    # --- Configuration (Partially Disabled in Viewer Mode) ---
    st.subheader("3. Settings" if not is_viewer_mode else "3. Settings (Backtest Only)")
    col_a, col_b = st.columns(2)
    with col_a:
         bot_behavior_options = ["none", "eq", "lt", "lte"]
         bot_behavior = st.selectbox(
             "Bot Reaction Matching",
             bot_behavior_options,
             index=3, # Default 'lte'
             key="bot_behavior",
             help="Rule for matching your leftover orders against trades seen in log history (simulating bot reactions): 'none'=no matching, 'eq'=match only equal prices, 'lt'=match strictly better prices, 'lte'=match equal or better prices.",
             disabled=is_viewer_mode
         )
    with col_b:
        ignore_limits_checkbox = st.checkbox(
            "Ignore Position Limits",
            value=False,
            key="ignore_limits",
            help="If checked, the backtester will not enforce product position limits.",
            disabled=is_viewer_mode
            )

    # Time Range Slider (Potentially useful in both modes)
    min_time_data, max_time_data = 0, 199900 # Sensible defaults
    file_for_time_range = None
    if is_viewer_mode and selected_log_viewer_file:
         file_for_time_range = selected_log_viewer_file
    elif not is_viewer_mode and selected_data_fnames:
         file_for_time_range = selected_data_fnames[0]

    if file_for_time_range:
         try:
              temp_data_path = os.path.join(DATA_DIR_ABS, file_for_time_range)
              # Use the parsing function but only care about market_df for range
              with open(temp_data_path, 'r', encoding='utf-8') as f:
                   _, temp_mkt, _ = util._parse_data(f)
                   if not temp_mkt.empty and pd.api.types.is_numeric_dtype(temp_mkt.index):
                       min_time_data = int(temp_mkt.index.min())
                       max_time_data = int(temp_mkt.index.max())
                   else:
                        print("Warning: Could not read numeric timestamp index from selected data file.")
         except Exception as e:
              print(f"Warning: Could not determine time range from {file_for_time_range}: {e}")

    time_range_values = st.slider(
          "Time Range Filter (Timestamp):",
          min_value=min_time_data, max_value=max_time_data,
          value=(min_time_data, max_time_data), step=100, key="timerange",
          help="Filters the displayed data within this range. For backtesting, also restricts simulation time."
     )

    # --- Execution Control ---
    st.divider()
    button_label = "📊 Load & View Log" if is_viewer_mode else "🚀 Run Backtest"
    button_disabled = (is_viewer_mode and not selected_log_viewer_file) or \
                      (not is_viewer_mode and (not selected_trader_fname or not selected_data_fnames))

    run_button_pressed = st.button(
        button_label,
        use_container_width=True,
        type="primary",
        disabled=button_disabled
    )


# --- Main Execution & Display Logic ---
backtester_instance = None # Initialize backtester instance variable

if run_button_pressed:
    st.session_state.parsed_log_data = ([], pd.DataFrame(), pd.DataFrame(), "") # Clear previous viewer data

    if is_viewer_mode:
        # --- Log Viewer Mode ---
        if selected_log_viewer_file:
            log_file_path = os.path.join(DATA_DIR_ABS, selected_log_viewer_file)
            st.session_state.last_loaded_log_file = selected_log_viewer_file
            print(f"--- Loading Log File: {log_file_path} ---")
            try:
                with st.spinner(f"Loading and parsing **{selected_log_viewer_file}**..."):
                    raw_content = load_file_content(log_file_path)
                    if raw_content.startswith("Error:"):
                        st.error(raw_content)
                    else:
                        # Use parse_backtest_output to leverage existing parsing logic
                        s_logs, m_df, t_df = parse_backtest_output(raw_content)
                        # Apply time filter immediately after parsing
                        m_df_filtered = m_df[(m_df.index >= time_range_values[0]) & (m_df.index <= time_range_values[1])]
                        t_df_filtered = t_df[(t_df.index >= time_range_values[0]) & (t_df.index <= time_range_values[1])]
                        # Filter sandbox logs based on timestamp if possible
                        s_logs_filtered = [log for log in s_logs if isinstance(log.get("timestamp"), int) and time_range_values[0] <= log["timestamp"] <= time_range_values[1]]

                        st.session_state.parsed_log_data = (s_logs_filtered, m_df_filtered, t_df_filtered, raw_content)
                        st.success(f"Successfully loaded and parsed {selected_log_viewer_file}.")
            except Exception as e:
                 st.error(f"Failed to load or parse log file: {e}")
                 st.code(traceback.format_exc())
                 st.session_state.parsed_log_data = ([], pd.DataFrame(), pd.DataFrame(), "") # Clear on error
        else:
            st.warning("Please select a log file to view.")

    else:
        # --- Backtest Mode ---
        if selected_trader_fname and selected_data_fnames:
            with st.spinner(f"Running backtest: **{selected_trader_fname}** on **{', '.join(selected_data_fnames)}**..."):
                 backtester_instance = run_backtest(
                     selected_trader_fname,
                     selected_data_fnames,
                     time_range_values, # Pass the time range filter
                     bot_behavior,
                     ignore_limits_checkbox
                 )
            if backtester_instance and hasattr(backtester_instance, 'output') and backtester_instance.output:
                 st.success("Backtest simulation completed.")
            elif backtester_instance:
                 st.warning("Backtest finished but produced no output. Check console/logs.")
            # else: Error message already shown in run_backtest
        else:
            st.warning("Please select a trader and at least one data file for backtesting.")


# --- Display Results ---
with rightcol:
    st.header("Results")

    # Determine the source of data to display
    display_data = None
    raw_output_content = ""
    source_description = ""

    if is_viewer_mode and st.session_state.parsed_log_data[1] is not None: # Check if viewer data is loaded
        sandbox_logs_parsed, market_data_parsed, trade_history_parsed, raw_output_content = st.session_state.parsed_log_data
        display_data = True # Flag that we have viewer data
        source_description = f"Displaying data from: **{st.session_state.last_loaded_log_file or 'N/A'}**"
    elif backtester_instance and hasattr(backtester_instance, 'output') and backtester_instance.output:
        try:
            with st.spinner("Parsing backtest output..."):
                raw_output_content = backtester_instance.output
                sandbox_logs_parsed, market_data_parsed, trade_history_parsed = parse_backtest_output(raw_output_content)
                display_data = True # Flag that we have backtest data
                source_description = f"Displaying backtest results for: **{selected_trader_fname or 'N/A'}**"
        except Exception as parse_err:
             st.error(f"Failed to parse the generated backtest output: {parse_err}")
             st.text("Showing raw output instead:")
             st.text_area("Raw Output:", raw_output_content, height=500)
             display_data = False # Cannot display parsed data
    elif run_button_pressed and not display_data: # If button was pressed but no data generated/loaded
         pass # Errors should have been shown already
    else: # Initial state or after mode switch
        st.info("Configure settings and run backtest or load a log file using the controls on the left.")
        display_data = False

    # --- Display Area ---
    if display_data:
        st.caption(source_description)
        tab1, tab2, tab3 = st.tabs(["📊 Charts", "📜 Logs", "📄 Raw Output & Download"])

        with tab1:
            st.subheader("Charts")
            if market_data_parsed.empty and trade_history_parsed.empty:
                 st.warning("No data available to display charts (check time filter or log content).")
            else:
                margin = dict(l=0, r=0, t=30, b=20)

                # --- PNL Chart ---
                # app.py

# ... inside the rightcol block -> tab1 ...

                # --- PNL Chart ---
                st.markdown("##### Profit and Loss (PNL)")

                # Columns needed: 'product' and 'profit_and_loss' (and the index 'timestamp')
                required_cols = ['product', 'profit_and_loss']
                if not all(col in market_data_parsed.columns for col in required_cols):
                     st.warning(f"Required columns for PNL chart ({', '.join(required_cols)}) not found in Activity Log data.")
                elif market_data_parsed.empty:
                     st.warning("No Activity Log data available for PNL chart.")
                else:
                    # Ensure index is named 'timestamp' and reset it to use as a column for plotting
                    pnl_plot_data = market_data_parsed.copy()
                    pnl_plot_data.index.name = 'timestamp'
                    pnl_plot_data.reset_index(inplace=True)

                    # Filter out rows where profit_and_loss might be NaN, if any
                    pnl_plot_data.dropna(subset=['profit_and_loss', 'product', 'timestamp'], inplace=True)

                    if pnl_plot_data.empty:
                         st.warning("No valid PNL data points remain after cleaning.")
                    else:
                         # 1. Plot PNL lines for each product
                         fig_pnl = px.line(pnl_plot_data,
                                           x='timestamp',
                                           y='profit_and_loss', # Plot the PNL value from the log/backtest
                                           color='product',     # Use product column for color
                                           title="PNL Over Time (Per Product Entry & Total)")

                         fig_pnl.update_layout(hovermode="x unified", margin=margin, xaxis_title=None, yaxis_title="PNL", legend_title="Product", height=350)
                         # Update hover template for individual product lines
                         fig_pnl.update_traces(hovertemplate="<b>%{fullData.name}</b><br>Timestamp: %{x}<br>PNL: %{y:.2f}<extra></extra>")


                         # 2. Calculate and add the Total PNL trace
                         # Group by timestamp and sum the 'profit_and_loss' for that timestamp
                         # Note: If 'profit_and_loss' already represents the total portfolio PNL in each row,
                         # summing might double-count. We should take the value from one representative row per timestamp.
                         # Let's assume 'profit_and_loss' IS the total portfolio PNL as per the log format.
                         # So, we take the value at each timestamp (e.g., the last one if duplicates exist)

                         total_pnl_trace_data = pnl_plot_data.filter(['timestamp', 'profit_and_loss']).groupby('timestamp').agg({
                             'profit_and_loss': 'sum' # Sum profit_and_loss across all products for each timestamp
                            }).reset_index()
                         if not total_pnl_trace_data.empty:
                              # Add the total PNL as a new trace to the existing figure
                              fig_pnl.add_scatter(x=total_pnl_trace_data['timestamp'],
                                                  y=total_pnl_trace_data['profit_and_loss'],
                                                  mode='lines',
                                                  name='Total PNL', # Name for the legend
                                                  line=dict(color='white', width=3, dash='dash'), # Style the total line
                                                  hovertemplate="<b>Total PNL</b><br>Timestamp: %{x}<br>PNL: %{y:.2f}<extra></extra>") # Custom hover

                         # Display the combined figure
                         st.plotly_chart(fig_pnl, use_container_width=True)
                # --- Position Chart ---
                # ...(rest of position chart logic remains the same)...
                st.markdown("##### Positions ('SUBMISSION' Trades)")
                if not trade_history_parsed.empty:
                    # ... (existing position calculation logic) ...
                    player_trades = trade_history_parsed[
                        (trade_history_parsed['buyer'] == "SUBMISSION") | (trade_history_parsed['seller'] == "SUBMISSION")
                    ].copy()
                    if not player_trades.empty:
                        # ... (existing position calculation logic) ...

                        player_trades['quantity'] = pd.to_numeric(player_trades['quantity'], errors='coerce').fillna(0).astype(int)
                        player_trades['price'] = pd.to_numeric(player_trades['price'], errors='coerce').fillna(0).astype(int)
                        player_trades['signed_quantity'] = player_trades.apply(
                            lambda row: row['quantity'] if row['buyer'] == 'SUBMISSION' else -row['quantity'], axis=1)
                        player_trades.sort_index(inplace=True)

                        pos_frames = []
                        products = sorted(player_trades['symbol'].unique()) # Sort products for consistent colors
                        for product in products:
                             prod_trades = player_trades[player_trades['symbol'] == product].copy()
                             prod_trades['position'] = prod_trades['signed_quantity'].cumsum()
                             pos_frames.append(prod_trades)

                        if pos_frames:
                             all_positions = pd.concat(pos_frames).sort_index()
                             all_positions['timestamp'] = all_positions.index

                             fig_pos = px.line(all_positions, x='timestamp', y='position', color='symbol', title="Player Positions (from SUBMISSION trades)")
                             fig_pos.update_layout(hovermode="x unified", margin=margin, xaxis_title=None, yaxis_title="Position", legend_title="Product", height=300)
                             fig_pos.update_traces(hovertemplate="<b>%{fullData.name}</b><br>Pos: %{y}<extra></extra>")
                             st.plotly_chart(fig_pos, use_container_width=True)
                        # else: # No need for else if pos_frames cannot be empty here
                    else:
                         st.info("No trades involving 'SUBMISSION' found in the trade history.")
                else:
                    st.info("No trade history data available for position chart.")
                # --- Log Price Chart ---
                # This uses market data, correct for both modes
                st.markdown("##### Market Prices (Log Scale)")
                # ...(rest of the price chart logic remains the same)...
                price_col_found = None
                price_data_filtered = pd.DataFrame()
                y_axis_title = "Log Price"
                # Check for VWAP columns first
                if not market_data_parsed.empty and all(col in market_data_parsed.columns for col in ['product', 'bid_price_1', 'bid_volume_1', 'ask_price_1', 'ask_volume_1', 
                                                                                                      'bid_price_2', 'bid_volume_2', 'ask_price_2', 'ask_volume_2', 
                                                                                                      'bid_price_3', 'bid_volume_3', 'ask_price_3', 'ask_volume_3']):
                    # Calculate VWAP safely (avoid division by zero)
                    mkt_data_copy = market_data_parsed.copy() # Work on a copy
                    mkt_data_copy.fillna(0, inplace=True) # Fill NaNs for calculation
                    n, d = 0,0
                    for i in range(1,4):
                        n += mkt_data_copy[f'bid_price_{i}']*mkt_data_copy[f'bid_volume_{i}'] # Bid prices and volumes
                        n += mkt_data_copy[f'ask_price_{i}']*mkt_data_copy[f'ask_volume_{i}'] # Ask prices and volumes
                        d += mkt_data_copy[f'bid_volume_{i}'] + mkt_data_copy[f'ask_volume_{i}'] # Total volumes for normalization
                    mkt_data_copy['vwap'] = n / d.replace(0, np.nan) # Avoid div by zero
                    # Forward fill VWAP *within each product group*
                    mkt_data_copy['vwap'] = mkt_data_copy.groupby('product')['vwap'].ffill()
                    mkt_data_copy['vwap'] = mkt_data_copy.groupby('product')['vwap'].bfill() # Backfill remaining NaNs at start
                    price_col_found = 'vwap'
                    y_axis_title = "Log VWAP"
                    price_data = mkt_data_copy

                # Fallback to mid_price
                elif not market_data_parsed.empty and 'mid_price' in market_data_parsed.columns and 'product' in market_data_parsed.columns:
                     mkt_data_copy = market_data_parsed.copy()
                     mkt_data_copy['mid_price'] = mkt_data_copy['mid_price'].fillna(method='ffill') # Fill NaNs
                     price_col_found = 'mid_price'
                     y_axis_title = "Log Mid Price"
                     price_data = mkt_data_copy
                     # st.info("VWAP calculation failed or columns missing, using Log Mid Price instead.") # Less noisy

                if price_col_found:
                     price_data['timestamp'] = price_data.index
                     # Handle 0 or negative before log - clip replaces values outside range
                     price_data[f'{price_col_found}_positive'] = price_data[price_col_found].clip(lower=1e-9)
                     price_data['log_price'] = np.log(price_data[f'{price_col_found}_positive'])

                     # Filter out rows where log price might still be NaN/inf after clipping (if original was NaN/inf/neg)
                     price_data_filtered = price_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_price', price_col_found, 'product'])
                     price_data_filtered['log_price'] = price_data_filtered.groupby('product')['log_price'].transform(lambda x: x - x.iloc[0])
                     if not price_data_filtered.empty:
                          fig_price = px.line(price_data_filtered, x='timestamp', y='log_price', color='product')
                          fig_price.update_layout(title=y_axis_title, hovermode="x unified", margin=margin, xaxis_title=None, yaxis_title="Log Price", legend_title="Product", height=300)
                          # Ensure customdata has the correct shape and content
                          customdata_hover = price_data_filtered[[price_col_found]].values # Use .values for correct shape
                          fig_price.update_traces(
                              hovertemplate=(
                                  f"<b>%{{fullData.name}}</b><br>LogPrice: %{{y:.3f}}<br>"
                                  f"{'VWAP' if price_col_found == 'vwap' else 'Mid'}: %{{customdata[0]:.2f}}<extra></extra>"
                              ),
                              customdata=customdata_hover # Pass original price for hover
                          )
                          st.plotly_chart(fig_price, use_container_width=True)
                     else:
                          st.warning(f"No valid data to plot for log prices using {price_col_found} (check time filter or log content).")
                else:
                     st.warning("Could not find suitable columns ('vwap' or 'mid_price') or 'product' column for price chart.")


        with tab2:
            st.subheader("Execution Logs")
            if sandbox_logs_parsed:
                 # Show logs in reverse order (most recent first)
                 for i, log_entry in enumerate(reversed(sandbox_logs_parsed)):
                     ts = log_entry.get("timestamp", "N/A")
                     sbox_log = log_entry.get("sandboxLog", "")
                     lambda_log = log_entry.get("lambdaLog", "")
                     # Filter out empty logs if desired
                     if not sbox_log.strip() and not lambda_log.strip():
                         continue
                     with st.expander(f"Timestamp {ts}", expanded=(i==0)): # Expand the latest log
                         if lambda_log and lambda_log.strip():
                             st.text("Trader Output (lambdaLog):")
                             st.code(lambda_log.strip(), language=None)
                         if sbox_log and sbox_log.strip():
                             st.text("Sandbox Messages:")
                             st.code(sbox_log.strip(), language=None)
            else:
                st.info("No sandbox logs found or remaining after time filter.")

            st.subheader("Full Trade History (Parsed)")
            if not trade_history_parsed.empty:
                # Highlight player trades function
                def highlight_submission(row):
                     color = 'lightgreen' if row['buyer'] == 'SUBMISSION' or row['seller'] == 'SUBMISSION' else ''
                     return [f'background-color: {color}'] * len(row)
                trade_history_display = trade_history_parsed.sort_index().reset_index()
                # --- END FIX ---

                # Apply style to the DataFrame with the reset index
                # Add hide_index=True to not show the default 0, 1, 2... index
                st.dataframe(
                    trade_history_display.style.apply(highlight_submission, axis=1),
                    height=400,
                    hide_index=True  # Add this!
                )
            else:
                st.info("No trade history found or remaining after time filter.")


        with tab3:
             st.subheader("Raw Output")
             if raw_output_content:
                 st.text_area("Full backtest output log:" if not is_viewer_mode else "Raw Log File Content:", raw_output_content, height=600, key="output_area")

                 # Generate default filename
                 log_file_name = st.session_state.last_loaded_log_file if is_viewer_mode else \
                                 (selected_data_fnames[0] if not is_viewer_mode and selected_data_fnames else 'multi_log')
                 log_file_tag = log_file_name.replace('.log','')
                 trader_tag = "log_view" if is_viewer_mode else (selected_trader_fname.replace('.py','') if selected_trader_fname else "no_trader")
                 download_filename = f"output_{trader_tag}_{log_file_tag}.log"

                 st.download_button(
                     label="📥 Download Output/Log",
                     data=raw_output_content,
                     file_name=download_filename,
                     mime="text/plain",
                     use_container_width=True
                 )
             else:
                 st.warning("No output content available to display or download.")