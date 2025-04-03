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
# Make sure __init__.py exists in the backtester folder
from backtester import backtester as bt_module
from backtester import util, constants
from typing import List, Dict, Tuple
st.set_page_config(layout="wide")

# app.py
# ... (imports)

script_dir = os.path.dirname(os.path.abspath(__file__))

# --- DEFINE ABSOLUTE PATHS FOR RESOURCES ---
TRADER_DIR_ABS = os.path.join(script_dir, "backtester", "traders") # <<< CORRECTED PATH
DATA_DIR_ABS = os.path.join(script_dir, "data")
OUTPUT_DIR_ABS = os.path.join(script_dir, "output")
os.makedirs(OUTPUT_DIR_ABS, exist_ok=True)


# --- Helper Functions ---

@st.cache_data
def parse_backtest_output(output_string: str) -> Tuple[List[Dict], pd.DataFrame, pd.DataFrame]:
    """Parses the full backtester output string using the util function."""
    if not output_string:
        return [], pd.DataFrame(), pd.DataFrame()
    try:
        output_io = io.StringIO(output_string)
        # Ensure correct return order matches expected usage
        sb_logs, market_df, trades_df = util._parse_data(output_io)
        # Explicitly convert timestamp index to int if needed after parsing
        if isinstance(market_df.index, pd.Float64Index):
            market_df.index = market_df.index.astype(int)
        if isinstance(trades_df.index, pd.Float64Index):
             trades_df.index = trades_df.index.astype(int)
        return sb_logs, market_df, trades_df
    except Exception as e:
        st.error(f"Error parsing backtest output: {e}")
        print(f"--- Error Parsing Output ---")
        print(output_string[:1000] + "...") # Print beginning of output for context
        traceback.print_exc()
        print(f"--- End Error ---")
        return [], pd.DataFrame(), pd.DataFrame()

@st.cache_data
def load_trader_code(filepath: str) -> str:
    """Loads the content of the trader file."""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: Trader file not found at {filepath}."
    except Exception as e:
        return f"Error loading trader code from {filepath}: {e}"

@st.cache_resource(ttl=3600) # Cache resource for 1 hour or until inputs change
def run_backtest(trader_filename: str, data_filenames: List[str], time_range: Tuple[int, int], bot_behavior: str, ignore_limits: bool):
    """
    Creates and runs the backtester instance.
    Expects filenames relative to their respective directories.
    """
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

    try:
        start_time = time.time()
        # Pass only filenames to Backtester, constructor handles joining now
        backtester_instance = bt_module.Backtester(
            trader_fname=trader_filename, # Pass just the filename
            data_fnames=data_filenames,   # Pass just the filenames
            timerange=time_range,
            bot_behavior=bot_behavior,
            ignore_limits=ignore_limits,
        )
        # run() method generates the .output attribute at the end
        end_time = time.time()
        print(f"Backtest completed in {end_time - start_time:.2f} seconds.")
        return backtester_instance
    except Exception as e:
        st.error(f"Error during backtest execution: {e}")
        st.code(traceback.format_exc())
        return None

# --- UI Layout ---
st.title("🌊 Prosperity Algorithm Backtester 📈")
leftcol, rightcol = st.columns([2, 5], gap="medium")
backtester_instance = None # Initialize

with leftcol:
    st.header("Configuration")
    st.markdown(
        """
        Select your trader, one or more data logs, configure settings, and run.
        Trade history from **all** selected logs reconstructs bot reactions.
        Market data (order books) uses the **first** selected log.
        """
    )

    # --- Trader Selection ---
    st.subheader("1. Select Trader")
    try:
        trader_files = sorted([
            f for f in os.listdir(TRADER_DIR_ABS)
            if os.path.isfile(os.path.join(TRADER_DIR_ABS, f)) and f.endswith(".py")
        ])
    except FileNotFoundError:
        st.error(f"Trader directory not found: `{TRADER_DIR_ABS}`")
        trader_files = []

    if not trader_files:
         st.warning(f"No Python traders found in `{TRADER_DIR_ABS}`.")
         selected_trader_fname = None
    else:
         selected_trader_fname = st.selectbox(
             "Trader File:",
             trader_files,
             key="trader_select",
             label_visibility="collapsed" # Hide label as subheader is above
         )
         # Display trader code
         if selected_trader_fname:
             trader_code_path = os.path.join(TRADER_DIR_ABS, selected_trader_fname)
             trader_code = load_trader_code(trader_code_path)
             with st.expander("View Trader Code"):
                 st.code(trader_code, language="python")

    # --- Data Selection ---
    st.subheader("2. Select Data Log(s)")
    try:
        data_files = sorted([
            f for f in os.listdir(DATA_DIR_ABS)
            if os.path.isfile(os.path.join(DATA_DIR_ABS, f)) and f.endswith(".log")
        ])
    except FileNotFoundError:
        st.error(f"Data directory not found: `{DATA_DIR_ABS}`")
        data_files = []

    if not data_files:
         st.warning(f"No log files (`.log`) found in `{DATA_DIR_ABS}`.")
         selected_data_fnames = []
    else:
         selected_data_fnames = st.multiselect(
             "Log Files:",
             data_files,
             default=data_files[0] if data_files else None,
             key="data_select",
             label_visibility="collapsed"
         )

    # --- Configuration ---
    st.subheader("3. Settings")
    col_a, col_b = st.columns(2)
    with col_a:
         bot_behavior_options = ["none", "eq", "lt", "lte"]
         bot_behavior = st.selectbox(
             "Bot Reaction Matching",
             bot_behavior_options,
             index=3, # Default 'lte'
             key="bot_behavior",
             help="Rule for matching your leftover orders against trades seen in log history (simulating bot reactions): 'none'=no matching, 'eq'=match only equal prices, 'lt'=match strictly better prices, 'lte'=match equal or better prices."
         )
    with col_b:
        ignore_limits_checkbox = st.checkbox(
            "Ignore Position Limits",
            value=False,
            key="ignore_limits",
            help="If checked, the backtester will not enforce product position limits."
            )

    # Time Range Slider
    min_time_data, max_time_data = 0, 199900 # Sensible defaults
    if selected_data_fnames:
         try:
              # Use absolute path to get time range from first file
              temp_data_path = os.path.join(DATA_DIR_ABS, selected_data_fnames[0])
              with open(temp_data_path, 'r') as f:
                   _, temp_mkt, _ = util._parse_data(f)
                   if not temp_mkt.empty and pd.api.types.is_numeric_dtype(temp_mkt.index):
                       min_time_data = int(temp_mkt.index.min())
                       max_time_data = int(temp_mkt.index.max())
                   else:
                        print("Warning: Could not read numeric timestamp index from first data file.")
         except Exception as e:
              print(f"Warning: Could not determine time range from {selected_data_fnames[0]}: {e}")

    time_range_values = st.slider(
          "Time Range (Timestamp):",
          min_value=min_time_data, max_value=max_time_data,
          value=(min_time_data, max_time_data), step=100, key="timerange"
     )

    # --- Execution Control ---
    st.divider()
    run_button_pressed = st.button(
        "🚀 Run Backtest",
        use_container_width=True,
        type="primary",
        disabled=(not selected_trader_fname or not selected_data_fnames) # Disable if no trader/data selected
    )


# --- Main Execution & Display Logic ---
if run_button_pressed:
    # Construct full paths before passing to cached function if needed, or pass relative names
    # run_backtest function now handles path construction internally
    with st.spinner(f"Running backtest: **{selected_trader_fname}** on **{', '.join(selected_data_fnames)}**..."):
         backtester_instance = run_backtest(
             selected_trader_fname,
             selected_data_fnames,
             time_range_values,
             bot_behavior,
             ignore_limits_checkbox
         )

# --- Display Results ---
with rightcol:
    st.header("Results")
    if backtester_instance is None:
        st.info("Run a backtest using the controls on the left.")
    elif not hasattr(backtester_instance, 'output') or not backtester_instance.output:
         st.error("Backtest ran but did not produce any output. Check console for errors.")
    else:
        # Parse the final output string from the completed backtester instance
        try:
            # Add a spinner while parsing the output, which can take a moment
            with st.spinner("Parsing backtest output..."):
                sandbox_logs_parsed, market_data_parsed, trade_history_parsed = parse_backtest_output(backtester_instance.output)
        except Exception as parse_err:
             st.error(f"Failed to parse the generated backtest output: {parse_err}")
             st.text("Showing raw output instead:")
             st.text_area("Raw Output:", backtester_instance.output, height=500)
             # Set dfs to empty to prevent chart errors
             market_data_parsed = pd.DataFrame()
             trade_history_parsed = pd.DataFrame()


        tab1, tab2, tab3 = st.tabs(["📊 Charts", "📜 Logs", "📄 Raw Output & Download"])

        with tab1:
            st.subheader("Charts")
            if market_data_parsed.empty and trade_history_parsed.empty:
                 st.warning("No data available to display charts.")
            else:
                margin = dict(l=0, r=0, t=30, b=20) # Adjusted margins slightly

                # --- PNL Chart ---
                st.markdown("##### Profit and Loss (PNL)")
                pnl_cols = [col for col in market_data_parsed.columns if col.startswith('pnl_') or col == 'profit_and_loss']
                if not pnl_cols:
                     st.warning("Could not find PNL data in activity log.")
                     pnl_data_long = pd.DataFrame()
                else:
                    pnl_col_to_use = 'profit_and_loss' # Final aggregated column name from backtester
                    if pnl_col_to_use not in market_data_parsed.columns:
                        st.warning(f"Final column '{pnl_col_to_use}' not found, trying to aggregate.")
                        # Try aggregating if individual pnl_ columns exist
                        individual_pnl_cols = [col for col in market_data_parsed.columns if col.startswith('pnl_')]
                        if individual_pnl_cols:
                             market_data_parsed['profit_and_loss'] = market_data_parsed[individual_pnl_cols].sum(axis=1)
                             pnl_col_to_use = 'profit_and_loss'
                        else:
                             pnl_col_to_use = None # Cannot plot PNL


                    if pnl_col_to_use:
                        # Plotting only the total PNL for simplicity now
                        pnl_plot_data = market_data_parsed[[pnl_col_to_use]].copy()
                        pnl_plot_data['timestamp'] = pnl_plot_data.index

                        fig_pnl = px.line(pnl_plot_data, x='timestamp', y=pnl_col_to_use)
                        fig_pnl.update_layout(hovermode="x unified", margin=margin, xaxis_title=None, yaxis_title="Total PNL", legend_title="Metric", height=300)
                        fig_pnl.update_traces(hovertemplate="Timestamp: %{x}<br>PNL: %{y:.2f}<extra></extra>")
                        st.plotly_chart(fig_pnl, use_container_width=True)

                # --- Position Chart ---
                st.markdown("##### Positions")
                if not trade_history_parsed.empty:
                    player_trades = trade_history_parsed[
                        (trade_history_parsed['buyer'] == "SUBMISSION") | (trade_history_parsed['seller'] == "SUBMISSION")
                    ].copy()
                    if not player_trades.empty:
                        player_trades['quantity'] = pd.to_numeric(player_trades['quantity'], errors='coerce').fillna(0).astype(int)
                        player_trades['price'] = pd.to_numeric(player_trades['price'], errors='coerce').fillna(0).astype(int)

                        player_trades['signed_quantity'] = player_trades.apply(
                            lambda row: -row['quantity'] if row['seller'] == 'SUBMISSION' else row['quantity'], axis=1)

                        player_trades.sort_index(inplace=True)

                        # Calculate cumulative position correctly respecting timestamps
                        pos_frames = []
                        products = player_trades['symbol'].unique()
                        for product in products:
                             prod_trades = player_trades[player_trades['symbol'] == product].copy()
                             prod_trades['position'] = prod_trades['signed_quantity'].cumsum()
                             pos_frames.append(prod_trades)

                        if pos_frames:
                             all_positions = pd.concat(pos_frames)
                             all_positions['timestamp'] = all_positions.index

                             fig_pos = px.line(all_positions, x='timestamp', y='position', color='symbol', title="Player Positions")
                             fig_pos.update_layout(hovermode="x unified", margin=margin, xaxis_title=None, yaxis_title="Position", legend_title="Product", height=300)
                             fig_pos.update_traces(hovertemplate="<b>%{fullData.name}</b><br>Pos: %{y}<extra></extra>")
                             st.plotly_chart(fig_pos, use_container_width=True)
                        else:
                             st.warning("No trades to calculate positions from.")
                    else:
                         st.info("No trades involving 'SUBMISSION' found in the trade history.")
                else:
                    st.info("No trade history data available for position chart.")

                # --- Log Price Chart ---
                st.markdown("##### Market Prices (Log Scale)")
                price_col_found = None
                # Check for VWAP columns first
                if all(col in market_data_parsed.columns for col in ['bid_price_1', 'bid_volume_1', 'ask_price_1', 'ask_volume_1']):
                    # Calculate VWAP safely (avoid division by zero)
                    market_data_parsed = market_data_parsed.fillna(0) # Fill NaNs for calculation
                    n = market_data_parsed["bid_price_1"]*market_data_parsed["bid_volume_1"] + market_data_parsed["ask_price_1"]*market_data_parsed["ask_volume_1"]
                    d = market_data_parsed["bid_volume_1"] + market_data_parsed["ask_volume_1"]
                    market_data_parsed['vwap'] = n / d.replace(0, np.nan) # Avoid div by zero
                    market_data_parsed['vwap'] = market_data_parsed.groupby('product')['vwap'].ffill().bfill() # Fill NaNs with neighbours
                    price_col_found = 'vwap'
                    y_axis_title = "Log VWAP"

                # Fallback to mid_price
                elif 'mid_price' in market_data_parsed.columns:
                     market_data_parsed['mid_price'] = market_data_parsed['mid_price'].fillna(0) # Fill NaNs
                     price_col_found = 'mid_price'
                     y_axis_title = "Log Mid Price"
                     st.info("VWAP calculation failed or columns missing, using Log Mid Price instead.")

                if price_col_found:
                     price_data = market_data_parsed[['product', price_col_found]].copy()
                     price_data['timestamp'] = price_data.index
                     # Handle 0 or negative before log - clip replaces values outside range
                     price_data[f'{price_col_found}_positive'] = price_data[price_col_found].clip(lower=1e-9)
                     price_data['log_price'] = np.log(price_data[f'{price_col_found}_positive'])

                     # Filter out rows where log price might still be NaN/inf after clipping (if original was NaN/inf/neg)
                     price_data_filtered = price_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_price'])

                     if not price_data_filtered.empty:
                          fig_price = px.line(price_data_filtered, x='timestamp', y='log_price', color='product')
                          fig_price.update_layout(title=y_axis_title, hovermode="x unified", margin=margin, xaxis_title=None, yaxis_title="Log Price", legend_title="Product", height=300)
                          fig_price.update_traces(
                              hovertemplate=(
                                  f"<b>%{{fullData.name}}</b><br>LogPrice: %{{y:.3f}}<br>"
                                  f"{'VWAP' if price_col_found == 'vwap' else 'Mid'}: %{{customdata[0]:.2f}}<extra></extra>"
                              ),
                              customdata=price_data_filtered[[price_col_found]] # Pass original price for hover
                          )
                          st.plotly_chart(fig_price, use_container_width=True)
                     else:
                          st.warning(f"No valid data to plot for log prices using {price_col_found}.")
                else:
                     st.warning("Could not find suitable columns ('vwap' or 'mid_price') for price chart.")

        with tab2:
            st.subheader("Execution Logs")
            if sandbox_logs_parsed:
                 # Show logs in reverse order (most recent first)
                 for i, log_entry in enumerate(reversed(sandbox_logs_parsed)):
                     ts = log_entry.get("timestamp", "N/A")
                     sbox_log = log_entry.get("sandboxLog", "")
                     lambda_log = log_entry.get("lambdaLog", "")
                     with st.expander(f"Timestamp {ts}", expanded=(i==0)): # Expand the latest log
                         if lambda_log and lambda_log.strip():
                             st.text("Trader Output (lambdaLog):")
                             st.code(lambda_log.strip(), language=None)
                         if sbox_log and sbox_log.strip():
                             st.text("Sandbox Messages:")
                             st.code(sbox_log.strip(), language=None)
            else:
                st.info("No sandbox logs found in output.")

            st.subheader("Full Trade History (Parsed)")
            if not trade_history_parsed.empty:
                # Highlight player trades
                def highlight_submission(row):
                     color = 'lightgreen' if row['buyer'] == 'SUBMISSION' or row['seller'] == 'SUBMISSION' else ''
                     return [f'background-color: {color}'] * len(row)
                st.dataframe(trade_history_parsed.sort_index().style.apply(highlight_submission, axis=1), height=400)
            else:
                st.info("No trade history found in output.")


        with tab3:
             st.subheader("Raw Output")
             if hasattr(backtester_instance, 'output') and backtester_instance.output:
                 log_content = backtester_instance.output
                 st.text_area("Full backtest output log:", log_content, height=600, key="output_area")

                 # Generate default filename
                 data_file_tag = selected_data_fnames[0].replace('.log','') if selected_data_fnames else 'multi_log'
                 download_filename = f"backtest_{selected_trader_fname.replace('.py','')}_{data_file_tag}.log"

                 st.download_button(
                     label="📥 Download Full Log",
                     data=log_content,
                     file_name=download_filename,
                     mime="text/plain",
                     use_container_width=True
                 )
             else:
                 st.warning("Backtester did not produce final output content.")