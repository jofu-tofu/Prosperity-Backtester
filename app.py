# --- START OF COMPLETE app.py (Style Corrected) ---
import io
import os
import time
import traceback

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

# Import backtester components
from backtester import backtester as bt_module, util, constants

# Import permutation utils, handle potential import errors
try:
    from backtester import permutation_utils as perm_utils
except ImportError:
    try:
        import permutation_utils as perm_utils # Check root directory as fallback
    except ImportError:
        st.warning("Optional: permutation_utils.py not found. Permutation testing will be disabled.")
        perm_utils = None

from typing import List, Dict, Tuple, Optional

# --- Page Config ---
st.set_page_config(layout="wide")

# --- Constants & Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
TRADER_DIR_ABS = os.path.join(script_dir, "backtester", "traders")
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
        sb_logs, market_df, trades_df = util._parse_data(output_io)
        # Ensure index types are correct after parsing
        if not market_df.empty and pd.api.types.is_float_dtype(market_df.index.dtype):
            market_df.index = market_df.index.astype(int)
        if not trades_df.empty and pd.api.types.is_float_dtype(trades_df.index.dtype):
            trades_df.index = trades_df.index.astype(int)
        return sb_logs, market_df, trades_df
    except Exception as e:
        st.error(f"Error parsing backtest output: {e}")
        print(f"--- Error Parsing Output ---")
        print(output_string[:1000] + "...")
        traceback.print_exc()
        print(f"--- End Error ---")
        return [], pd.DataFrame(), pd.DataFrame()

@st.cache_data
def load_file_content(filepath: str) -> str:
    """Loads the raw content of any text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found at {filepath}."
    except Exception as e:
        return f"Error loading file content from {filepath}: {e}"

def calculate_row_vwap(row):
    """Calculates VWAP for a single row (pandas apply)."""
    n = 0.0
    d = 0.0
    try:
        for i in range(1, 4): # Levels 1 to 3
            bp, bv = row.get(f'bid_price_{i}', np.nan), row.get(f'bid_volume_{i}', np.nan)
            if pd.notna(bp) and pd.notna(bv) and bv > 0:
                n += bp * bv
                d += bv
            ap, av = row.get(f'ask_price_{i}', np.nan), row.get(f'ask_volume_{i}', np.nan)
            if pd.notna(ap) and pd.notna(av) and av > 0:
                n += ap * av
                d += av
        if d > 0:
             return n / d
        else:
             return np.nan
    except Exception: # Catch any calculation errors
        return np.nan

# Note: Removed caching from run_backtest to ensure fresh runs
def run_backtest(
    trader_filename: str,
    data_filenames: List[str],
    time_range: Tuple[int, int],
    bot_behavior: str,
    ignore_limits: bool,
    order_depth_override: Optional[Dict] = None, # For permutations
    disable_inferred: bool = False # For permutations/debugging
    ):
    """Creates and runs the backtester instance."""
    run_type = "OVERRIDE" if order_depth_override is not None else "NORMAL"
    print(f"\n--- Running Backtest ({run_type}) ---")
    print(f"Trader: {trader_filename}, Logs: {data_filenames}, Time: {time_range}, BotB: {bot_behavior}, IgnoreL: {ignore_limits}, DisableInfer: {disable_inferred}")

    full_trader_path = os.path.join(TRADER_DIR_ABS, trader_filename)
    full_data_paths = [os.path.join(DATA_DIR_ABS, fname) for fname in data_filenames]

    # Validate paths
    if not os.path.exists(full_trader_path):
        st.error(f"Trader file not found: {full_trader_path}")
        return None
    if not full_data_paths or not os.path.exists(full_data_paths[0]):
        st.error(f"Primary data file missing or path error for: {full_data_paths}")
        return None

    backtester_instance = None
    try:
        print("Creating Backtester instance...")
        t_start_init = time.time()
        backtester_instance = bt_module.Backtester(
            trader_fname=trader_filename,
            data_fnames=data_filenames,
            timerange=time_range,
            bot_behavior=bot_behavior,
            ignore_limits=ignore_limits,
        )
        print(f"Instance created in {time.time()-t_start_init:.2f}s.")

        # Call the backtester's run method with optional override args
        print(f"Starting simulation ({run_type})...")
        t_start_run = time.time()
        backtester_instance.run(
            order_depth_override_cache=order_depth_override,
            disable_inferred_book=disable_inferred
        )
        print(f"Simulation completed in {time.time()-t_start_run:.2f}s.")
        return backtester_instance

    except Exception as e:
        st.error(f"Error during backtest execution: {e}")
        st.code(traceback.format_exc())
        return None


# --- START: Refactored Charting Functions ---

def display_pnl_chart(market_df: pd.DataFrame, title_prefix: str):
    """Displays the PNL chart (Product-specific + Total Overlay)."""
    st.markdown("##### Profit and Loss (PNL)")
    required_cols = ['product', 'profit_and_loss']
    if not all(col in market_df.columns for col in required_cols):
        st.warning(f"PNL Chart: Required columns missing ({title_prefix}).")
        return
    if market_df.empty:
        st.warning(f"PNL Chart: No Activity Log data ({title_prefix}).")
        return

    pnl_plot_data = market_df.copy()
    pnl_plot_data.index.name = 'timestamp'
    pnl_plot_data.reset_index(inplace=True)
    pnl_plot_data.dropna(subset=['profit_and_loss', 'product', 'timestamp'], inplace=True)

    if pnl_plot_data.empty:
        st.warning(f"PNL Chart: No valid PNL data points remain after cleaning ({title_prefix}).")
        return

    fig_pnl = px.line(pnl_plot_data, x='timestamp', y='profit_and_loss', color='product',
                      title=f"{title_prefix} - Individual Product PNL & Total")
    fig_pnl.update_layout(hovermode="x unified", margin=dict(l=0,r=0,t=40,b=20), xaxis_title=None, yaxis_title="PNL", legend_title="Product", height=350)
    fig_pnl.update_traces(hovertemplate="Product: <b>%{fullData.name}</b><br>Timestamp: %{x}<br>PNL: %{y:.2f}<extra></extra>")

    total_pnl_trace_data = pnl_plot_data.groupby('timestamp')['profit_and_loss'].sum().reset_index()
    if not total_pnl_trace_data.empty:
        fig_pnl.add_scatter(x=total_pnl_trace_data['timestamp'], y=total_pnl_trace_data['profit_and_loss'],
                            mode='lines', name='Overall Total PNL',
                            line=dict(color='rgba(0,0,0,0.6)', width=4, dash='dot'),
                            hovertemplate="<b>Overall Total PNL</b><br>Timestamp: %{x}<br>Total PNL: %{y:.2f}<extra></extra>")
    st.plotly_chart(fig_pnl, use_container_width=True)

def display_position_chart(trades_df: pd.DataFrame, title_prefix: str):
    """Displays the position chart based on SUBMISSION trades."""
    st.markdown("##### Positions ('SUBMISSION' Trades)")
    if trades_df.empty:
        st.info(f"Position Chart: No trade history data available ({title_prefix}).")
        return

    player_trades = trades_df[(trades_df['buyer'] == "SUBMISSION") | (trades_df['seller'] == "SUBMISSION")].copy()
    if player_trades.empty:
        st.info(f"Position Chart: No 'SUBMISSION' trades found ({title_prefix}).")
        return
    try:
        player_trades['quantity'] = pd.to_numeric(player_trades['quantity'], errors='coerce').fillna(0).astype(int)
        player_trades['signed_quantity'] = player_trades.apply(lambda r: r['quantity'] if r['buyer']=='SUBMISSION' else -r['quantity'], axis=1)
        player_trades.sort_index(inplace=True)
        pos_frames = []
        products = sorted(player_trades['symbol'].unique())
        for product in products:
            prod_trades = player_trades[player_trades['symbol']==product].copy()
            prod_trades['position'] = prod_trades['signed_quantity'].cumsum()
            pos_frames.append(prod_trades)
        if pos_frames:
            all_positions = pd.concat(pos_frames).sort_index()
            all_positions['timestamp'] = all_positions.index
            fig_pos = px.line(all_positions, x='timestamp', y='position', color='symbol', title=f"{title_prefix} - Player Positions")
            fig_pos.update_layout(hovermode="x unified", margin=dict(l=0,r=0,t=30,b=20), xaxis_title=None, yaxis_title="Position", legend_title="Product", height=300)
            fig_pos.update_traces(hovertemplate="<b>%{fullData.name}</b><br>Pos: %{y}<extra></extra>")
            st.plotly_chart(fig_pos, use_container_width=True)
    except Exception as e:
        st.error(f"Position Chart Error ({title_prefix}): {e}")

def display_norm_price_chart(market_df: pd.DataFrame, title_prefix: str):
    """Displays the normalized price change chart."""
    st.markdown("##### Market Prices (Normalized Change)")
    if market_df.empty:
        st.warning(f"Norm Price Chart: No market data available ({title_prefix}).")
        return

    market_data_processed = market_df.copy()
    vwap_available = False
    if 'vwap' not in market_data_processed.columns or market_data_processed['vwap'].isna().all():
        try:
            print(f"Calculating VWAP for Norm Price Chart ({title_prefix})...")
            price_cols = [f'{p}_{i}' for p in ['bid_price', 'ask_price'] for i in range(1, 4)]
            vol_cols = [f'{p}_{i}' for p in ['bid_volume', 'ask_volume'] for i in range(1, 4)]
            cols_to_cvt = [c for c in price_cols + vol_cols if c in market_data_processed.columns]
            for col in cols_to_cvt:
                 if not pd.api.types.is_numeric_dtype(market_data_processed[col]):
                     market_data_processed[col] = pd.to_numeric(market_data_processed[col], errors='coerce')
            vwap_s = market_data_processed.apply(calculate_row_vwap, axis=1)
            market_data_processed['vwap'] = vwap_s
            market_data_processed['vwap'] = market_data_processed.groupby('product')['vwap'].ffill().bfill()
            vwap_available = not market_data_processed['vwap'].isna().all()
        except Exception as e:
            print(f"Error calculating VWAP for Norm Price Chart ({title_prefix}): {e}")
            market_data_processed['vwap'] = np.nan # Ensure column exists but is NaN
    else:
        vwap_available = True # VWAP already existed

    price_col_to_use = None
    y_axis_title = "Price Change ($)"
    if vwap_available:
        price_col_to_use = 'vwap'
        y_axis_title = f"{title_prefix} - VWAP Change ($)"
    elif 'mid_price' in market_data_processed.columns and not market_data_processed['mid_price'].isna().all():
        # Ensure Mid Price is numeric and filled
        if not pd.api.types.is_numeric_dtype(market_data_processed['mid_price']):
            market_data_processed['mid_price'] = pd.to_numeric(market_data_processed['mid_price'], errors='coerce')
        market_data_processed['mid_price'] = market_data_processed.groupby('product')['mid_price'].ffill().bfill()
        if not market_data_processed['mid_price'].isna().all():
            price_col_to_use = 'mid_price'
            y_axis_title = f"{title_prefix} - Mid Price Change ($)"

    if price_col_to_use and 'product' in market_data_processed.columns:
        price_data = market_data_processed[['product', price_col_to_use]].copy()
        price_data.index.name = 'timestamp'
        price_data.dropna(subset=[price_col_to_use, 'product'], inplace=True)
        if not price_data.empty:
            def normalize_group(g):
                first_idx = g[price_col_to_use].first_valid_index()
                if first_idx is None:
                    g['normalized_price'] = np.nan
                else:
                    g['normalized_price'] = g[price_col_to_use] - g.loc[first_idx, price_col_to_use]
                return g

            price_data_normalized = price_data.groupby('product', group_keys=False).apply(normalize_group)
            price_data_normalized_for_plot = price_data_normalized.reset_index()
            price_data_normalized_for_plot.dropna(subset=['normalized_price'], inplace=True)

            if not price_data_normalized_for_plot.empty:
                fig_price_norm = px.line(price_data_normalized_for_plot, x='timestamp', y='normalized_price', color='product')
                min_y = price_data_normalized_for_plot['normalized_price'].min()
                max_y = price_data_normalized_for_plot['normalized_price'].max()
                ymin = min_y - (abs(min_y) * 0.1 + 0.1) if pd.notna(min_y) else -1
                ymax = max_y + (abs(max_y) * 0.1 + 0.1) if pd.notna(max_y) else 1
                fig_price_norm.update_layout(
                    title=y_axis_title, hovermode="x unified", margin=dict(l=0,r=0,t=30,b=20),
                    xaxis_title=None, yaxis_title="Price Change ($)", legend_title="Product", height=300,
                    yaxis_zeroline=True, yaxis_zerolinecolor='Gray', yaxis_zerolinewidth=1, yaxis_range=[ymin, ymax]
                )
                price_data_orig_for_merge = price_data.reset_index()
                merged_data = pd.merge(
                    price_data_normalized_for_plot[['timestamp','product','normalized_price']],
                    price_data_orig_for_merge[['timestamp','product',price_col_to_use]],
                    on=['timestamp','product'], how='left'
                )
                merged_data.sort_values(['product','timestamp'], inplace=True)
                fig_price_norm.update_traces(
                    hovertemplate=(f"<b>%{{fullData.name}}</b><br>TS: %{{x}}<br>Chg: %{{y:+.2f}}<br>{price_col_to_use.upper()}: %{{customdata[0]:.2f}}<extra></extra>"),
                    customdata=merged_data[[price_col_to_use]].values
                )
                st.plotly_chart(fig_price_norm, use_container_width=True)
            else:
                st.warning(f"Norm Price Chart: No valid normalized price data ({title_prefix}).")
        else:
            st.warning(f"Norm Price Chart: No valid base price data ({title_prefix}).")
    else:
        st.warning(f"Norm Price Chart: Cannot find suitable price column (VWAP or Mid) ({title_prefix}).")


def display_fill_dist_chart(market_df: pd.DataFrame, trades_df: pd.DataFrame, title_prefix: str, is_permutation_run: bool = False):
    """Displays the fill distance chart."""
    st.divider()
    st.markdown("##### Fill Distance from VWAP")
    if market_df.empty:
        st.warning(f"Fill Dist: Market data needed ({title_prefix}).")
        return
    if trades_df.empty:
        st.warning(f"Fill Dist: Trade data needed ({title_prefix}).")
        return

    market_data_processed = market_df.copy()
    vwap_available = False
    if 'vwap' in market_data_processed.columns and not market_data_processed['vwap'].isna().all():
         vwap_available = True
    else: # Try to calculate VWAP if missing
         try:
            print(f"Calculating VWAP for Fill Distance Chart ({title_prefix})...")
            price_cols = [f'{p}_{i}' for p in ['bid_price', 'ask_price'] for i in range(1, 4)]
            vol_cols = [f'{p}_{i}' for p in ['bid_volume', 'ask_volume'] for i in range(1, 4)]
            cols_to_cvt = [c for c in price_cols + vol_cols if c in market_data_processed.columns]
            for col in cols_to_cvt:
                 if not pd.api.types.is_numeric_dtype(market_data_processed[col]):
                     market_data_processed[col] = pd.to_numeric(market_data_processed[col], errors='coerce')
            vwap_s = market_data_processed.apply(calculate_row_vwap, axis=1)
            market_data_processed['vwap'] = vwap_s
            market_data_processed['vwap'] = market_data_processed.groupby('product')['vwap'].ffill().bfill()
            vwap_available = not market_data_processed['vwap'].isna().all()
         except Exception as e:
            print(f"Error calculating VWAP for Fill Dist ({title_prefix}): {e}")
    if not vwap_available:
        st.warning(f"Fill Dist: VWAP unavailable ({title_prefix}).")
        return

    player_trades = trades_df[(trades_df['buyer'] == "SUBMISSION") | (trades_df['seller'] == "SUBMISSION")].copy()
    player_trades.index.name = 'timestamp'
    player_trades.reset_index(inplace=True)
    if player_trades.empty:
        st.info(f"Fill Dist: No SUBMISSION trades ({title_prefix}).")
        return

    def get_fill_type(row):
        if row['buyer'] == 'SUBMISSION': return row.get('seller', 'Unknown')
        elif row['seller'] == 'SUBMISSION': return row.get('buyer', 'Unknown')
        else: return 'Other'
    player_trades['fill_type'] = player_trades.apply(get_fill_type, axis=1)
    expected_fills = ['ExplicitBook', 'InferredBot'] if not is_permutation_run else ['ExplicitBook']
    player_trades = player_trades[player_trades['fill_type'].isin(expected_fills)]
    player_trades['quantity'] = pd.to_numeric(player_trades['quantity'], errors='coerce').fillna(0)
    if player_trades.empty:
        st.info(f"Fill Dist: No {expected_fills} fills found ({title_prefix}).")
        return

    # Merge with VWAP
    vwap_data_for_merge = market_data_processed[['product', 'vwap']].reset_index()
    merged_trades = pd.merge(player_trades, vwap_data_for_merge, left_on=['timestamp', 'symbol'], right_on=['timestamp', 'product'], how='left')
    if 'product' in merged_trades.columns:
        merged_trades.drop(columns=['product'], inplace=True)
    merged_trades.dropna(subset=['vwap', 'price'], inplace=True)
    if merged_trades.empty:
        st.warning(f"Fill Dist: No trades after VWAP merge ({title_prefix}).")
        return

    # Calculate/Bucket Distance & Aggregate
    merged_trades['distance'] = abs(merged_trades['price'] - merged_trades['vwap'])
    max_b = 5
    labels = {i: f"${i}-{i+1-0.01:.2f}" for i in range(max_b)}
    labels[max_b] = f"${max_b}+"
    merged_trades['dist_bucket'] = np.floor(merged_trades['distance']).astype(int).clip(upper=max_b)
    merged_trades['dist_label'] = merged_trades['dist_bucket'].map(labels)
    label_order = [labels[i] for i in range(max_b + 1)]

    overall_summary = merged_trades.groupby(['dist_label','fill_type'])['quantity'].sum().reset_index()
    overall_summary['display_group'] = 'Overall'
    product_summary = merged_trades.groupby(['dist_label','fill_type','symbol'])['quantity'].sum().reset_index()
    product_summary.rename(columns={'symbol':'display_group'}, inplace=True)
    combined_summary = pd.concat([overall_summary, product_summary], ignore_index=True)
    combined_summary.rename(columns={'quantity':'total_volume'}, inplace=True)
    prod_names = sorted(merged_trades['symbol'].unique())
    display_order = ['Overall'] + prod_names

    if combined_summary.empty:
        st.info(f"Fill Dist: No volume data after aggregation ({title_prefix}).")
        return

    color_map = {"ExplicitBook": px.colors.qualitative.Plotly[0], "InferredBot": px.colors.qualitative.Plotly[1]}
    fig_fill = px.bar(combined_summary, x='dist_label', y='total_volume', color='fill_type',
                      facet_col='display_group', barmode='group',
                      title=f"{title_prefix} - Fill Distance from VWAP",
                      labels={'dist_label':'Dist ($)', 'total_volume':'Volume', 'fill_type':'Source', 'display_group':'Group'},
                      category_orders={"dist_label": label_order, "display_group": display_order},
                      color_discrete_map=color_map)
    fig_fill.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=400)
    fig_fill.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(fig_fill, use_container_width=True)

# --- END: Refactored Charting Functions ---


# --- Session State Init & Clear/Rerun Functions ---
if 'log_viewer_mode' not in st.session_state:
    st.session_state.log_viewer_mode = False
if 'parsed_log_data' not in st.session_state:
    st.session_state.parsed_log_data = ([], pd.DataFrame(), pd.DataFrame(), "") # s_logs, m_df, t_df, raw_content
if 'last_loaded_log_file' not in st.session_state:
    st.session_state.last_loaded_log_file = None
if 'permutation_results' not in st.session_state:
    st.session_state.permutation_results = {'original_pnl': None, 'permuted_pnls': [], 'p_value': None}
if 'run_permutation' not in st.session_state:
    st.session_state.run_permutation = False
if 'show_output_log_state' not in st.session_state:
    st.session_state.show_output_log_state = False
if 'view_permutation_index' not in st.session_state:
    st.session_state.view_permutation_index = 0
if 'viewed_permutation_data' not in st.session_state:
    st.session_state.viewed_permutation_data = ([], pd.DataFrame(), pd.DataFrame(), "") # s_logs, m_df, t_df, raw_output
if 'permutation_rerun_args' not in st.session_state:
    st.session_state.permutation_rerun_args = {}

def clear_permutation_results():
    st.session_state.permutation_results = {'original_pnl': None, 'permuted_pnls': [], 'p_value': None}
    st.session_state.view_permutation_index = 0
    st.session_state.viewed_permutation_data = ([], pd.DataFrame(), pd.DataFrame(), "")
    if 'permutation_rerun_args' in st.session_state:
        del st.session_state.permutation_rerun_args

def clear_parsed_data():
    st.session_state.parsed_log_data = ([], pd.DataFrame(), pd.DataFrame(), "")
    st.session_state.last_loaded_log_file = None
    clear_permutation_results()

def rerun_and_view_permutation(perm_index_to_view: int, perm_args: dict):
    """Regenerates and reruns a specific permutation index."""
    st.session_state.viewed_permutation_data = ([], pd.DataFrame(), pd.DataFrame(), "") # Clear previous view
    if perm_index_to_view <= 0 or not perm_args or not perm_utils:
        st.warning("Cannot view permutation. Invalid index or missing data/utils.")
        return

    perm_progress = st.progress(0.0, text=f"Regenerating permutation {perm_index_to_view} data...")
    try:
        i_seed = perm_index_to_view - 1 # Adjust to 0-based index for seeding
        np.random.seed(i_seed) # Ensure deterministic regeneration
        permuted_log_changes = perm_utils.block_permutation(perm_args['log_vwap_changes'], perm_args['block_size'])
        reconstructed_log_vwap = perm_utils.reconstruct_log_vwap(perm_args['initial_log_vwap'], permuted_log_changes)
        reconstructed_vwap = np.exp(reconstructed_log_vwap)
        permuted_vwap_map = reconstructed_vwap.to_dict()
        valid_perm_timestamps = reconstructed_vwap.index.to_numpy()
        permuted_book_cache = perm_utils.generate_permuted_order_book_cache(valid_perm_timestamps, permuted_vwap_map, perm_args['relative_books'])
        perm_progress.progress(0.5, text=f"Running backtest for permutation {perm_index_to_view}...")

        view_backtester = run_backtest( # Rerun backtest with permuted data
             perm_args['trader_file'], perm_args['data_files'], perm_args['time_range'],
             perm_args['bot_behavior'], perm_args['ignore_limits'],
             order_depth_override=permuted_book_cache, disable_inferred=True
        )
        if view_backtester and hasattr(view_backtester, 'output') and view_backtester.output:
            print(f"Parsing output for viewed permutation {perm_index_to_view}")
            s_logs, m_df, t_df = parse_backtest_output(view_backtester.output)
            st.session_state.viewed_permutation_data = (s_logs, m_df, t_df, view_backtester.output)
            st.success(f"Loaded results for permutation {perm_index_to_view}.")
        else:
            st.error(f"Failed rerun for permutation {perm_index_to_view}.")
    except Exception as e:
        st.error(f"Error viewing permutation {perm_index_to_view}: {e}")
        traceback.print_exc()
    finally:
        perm_progress.empty()


# --- UI Layout ---
st.title("🌊 Prosperity Backtester & Permutation Test 📈")
leftcol, rightcol = st.columns([2, 5], gap="medium")

# --- Left Column (Configuration) ---
with leftcol:
    st.header("Configuration")
    is_viewer_mode = st.checkbox("View Log File Only", key="log_viewer_mode", on_change=clear_parsed_data)
    st.caption("Run simulation with trader." if not is_viewer_mode else "Parse and display a single log file.")
    st.divider()

    # Initialize config variables
    selected_trader_fname = None
    selected_data_fnames = []
    selected_log_viewer_file = None
    perm_blocksize = 1
    perm_n_iterations = 0
    time_range_values = (0, 199900) # Default time range
    bot_behavior = 'lte'
    ignore_limits_checkbox = False

    # --- Backtest Mode UI ---
    if not is_viewer_mode:
        st.subheader("1. Select Trader")
        try:
            trader_files = sorted([f for f in os.listdir(TRADER_DIR_ABS) if f.endswith(".py")])
        except Exception as e:
            st.error(f"Trader dir err: {e}")
            trader_files = []
        if not trader_files:
            st.warning(f"No traders in {TRADER_DIR_ABS}.")
        else:
            selected_trader_fname = st.selectbox("Trader:", trader_files, key="trader_select", label_visibility="collapsed")
        if selected_trader_fname:
            trader_code = load_file_content(os.path.join(TRADER_DIR_ABS, selected_trader_fname))
            with st.expander("View Code"):
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
            selected_data_fnames = st.multiselect("Logs (>=1, 1st is primary):", data_files, default=data_files[0] if data_files else None, key="data_select", label_visibility="collapsed")

        st.subheader("3. Settings")
        col_a, col_b = st.columns(2)
        with col_a:
            bot_behavior = st.selectbox("Bot Matching:", ["none", "eq", "lt", "lte"], index=3, key="bot_behavior", help="Rule vs inferred bots.")
        with col_b:
            ignore_limits_checkbox = st.checkbox("Ignore Limits", key="ignore_limits", help="Disable position limits.")

        # --- Time Range Slider (Backtest) ---
        min_time, max_time = 0, 199900
        if selected_data_fnames: # Determine range from first selected log
            try:
                 with open(os.path.join(DATA_DIR_ABS, selected_data_fnames[0]), 'r', encoding='utf-8') as f:
                     _, temp_mkt, _ = util._parse_data(f)
                 if not temp_mkt.empty and pd.api.types.is_numeric_dtype(temp_mkt.index):
                     min_time=int(temp_mkt.index.min())
                     max_time=int(temp_mkt.index.max())
            except Exception as e:
                print(f"Warn: Time range fail {selected_data_fnames[0]}: {e}")
        time_range_values = st.slider("Time Range:", min_value=min_time, max_value=max_time, value=(min_time, max_time), step=100, key="timerange_backtest")

        # --- Permutation Test UI (Backtest Only) ---
        st.divider()
        st.subheader("Permutation Testing")
        run_permutation_test = st.checkbox("Enable", key="run_permutation", value=False, on_change=clear_permutation_results)
        if run_permutation_test and perm_utils:
             total_ts = (time_range_values[1] - time_range_values[0]) // 100 + 1
             max_block = max(1, total_ts // 10)
             perm_blocksize = st.number_input("Block Size:", min_value=1, max_value=max_block, value=max(1, max_block // 10), step=10, key="perm_blocksize", help=f"Max sugg: {max_block}")
             perm_n_iterations = st.number_input("Permutations:", min_value=1, max_value=10000, value=99, step=10, key="perm_n_iterations")
        elif run_permutation_test and not perm_utils:
             st.error("Permutation testing enabled, but permutation_utils failed to import.")

    # --- Log Viewer Mode UI ---
    else:
         st.subheader("Select Log File")
         try:
             data_files = sorted([f for f in os.listdir(DATA_DIR_ABS) if f.endswith(".log")])
         except Exception as e:
             st.error(f"Data dir err: {e}")
             data_files = []
         if not data_files:
             st.warning(f"No logs found in {DATA_DIR_ABS}.")
         else:
             selected_log_viewer_file = st.selectbox("Log:", data_files, key="log_viewer_select", label_visibility="collapsed")
         if selected_log_viewer_file:
             selected_data_fnames = [selected_log_viewer_file] # Store single file name for consistency

         # --- Time Range Slider (Log Viewer) ---
         min_time, max_time = 0, 199900
         if selected_log_viewer_file: # Determine range from selected log
              try:
                   with open(os.path.join(DATA_DIR_ABS, selected_log_viewer_file), 'r', encoding='utf-8') as f:
                       _, temp_mkt, _ = util._parse_data(f)
                   if not temp_mkt.empty and pd.api.types.is_numeric_dtype(temp_mkt.index):
                       min_time=int(temp_mkt.index.min())
                       max_time=int(temp_mkt.index.max())
              except Exception as e:
                  print(f"Warn: Viewer time range fail {selected_log_viewer_file}: {e}")
         time_range_values = st.slider("Time Range Filter:", min_value=min_time, max_value=max_time, value=(min_time, max_time), step=100, key="timerange_viewer")

    # --- Common UI Elements ---
    st.checkbox("Show Output Log in Results", key="show_output_log_state", value=False)
    st.divider()
    run_perm_enabled = st.session_state.get('run_permutation', False) and not is_viewer_mode and perm_utils
    button_label = "📊 Load Log" if is_viewer_mode else ("🧪 Run & Permute" if run_perm_enabled else "🚀 Run Backtest")
    button_disabled = (is_viewer_mode and not selected_log_viewer_file) or \
                      (not is_viewer_mode and (not selected_trader_fname or not selected_data_fnames))
    run_button_pressed = st.button(button_label, use_container_width=True, type="primary", disabled=button_disabled)


# --- Main Execution Logic ---

backtester_instance = None # Holds the instance from the *original* backtest run
if run_button_pressed:
    clear_parsed_data() # Clear all previous results on new run

    if is_viewer_mode: # --- Log Viewer ---
        if selected_log_viewer_file:
            log_file_path = os.path.join(DATA_DIR_ABS, selected_log_viewer_file)
            st.session_state.last_loaded_log_file = selected_log_viewer_file
            print(f"--- Loading Log: {log_file_path} ---")
            try:
                with st.spinner(f"Parsing **{selected_log_viewer_file}**..."):
                    raw_content = load_file_content(log_file_path)
                    if raw_content.startswith("Error:"):
                        st.error(raw_content)
                    else:
                        s_logs, m_df, t_df = parse_backtest_output(raw_content)
                        # Apply time filter *after* parsing
                        m_df_f = m_df[(m_df.index >= time_range_values[0]) & (m_df.index <= time_range_values[1])]
                        t_df_f = t_df[(t_df.index >= time_range_values[0]) & (t_df.index <= time_range_values[1])]
                        s_logs_f = [log for log in s_logs if isinstance(log.get("timestamp"), int) and time_range_values[0] <= log["timestamp"] <= time_range_values[1]]
                        st.session_state.parsed_log_data = (s_logs_f, m_df_f, t_df_f, raw_content) # Store filtered data
                        st.success(f"Parsed {selected_log_viewer_file}.")
            except Exception as e:
                st.error(f"Failed load/parse: {e}")
                st.code(traceback.format_exc())
        else:
            st.warning("Select log file.")

    elif not is_viewer_mode and selected_trader_fname and selected_data_fnames: # --- Backtest ---
        run_permutations_now = st.session_state.get('run_permutation', False) and perm_utils

        # --- 1. Run Original Backtest ---
        st.info("Running original backtest...")
        with st.spinner(f"Running original: **{selected_trader_fname}**..."):
             backtester_instance = run_backtest(
                 selected_trader_fname, selected_data_fnames, time_range_values,
                 bot_behavior, ignore_limits_checkbox,
                 order_depth_override=None, disable_inferred=False
             )
        if not backtester_instance or not hasattr(backtester_instance, 'output') or not backtester_instance.output:
             st.error("Original backtest failed or produced no output.")
             st.stop() # Stop if original fails

        print("Parsing original backtest output...")
        s_logs_orig, m_df_orig, t_df_orig = parse_backtest_output(backtester_instance.output)
        st.session_state.parsed_log_data = (s_logs_orig, m_df_orig, t_df_orig, backtester_instance.output) # Store original run results

        # Extract Final PNL from original run
        final_pnl_original = np.nan
        if not m_df_orig.empty and 'profit_and_loss' in m_df_orig.columns:
            pnl_series_orig = pd.to_numeric(m_df_orig['profit_and_loss'], errors='coerce')
            last_timestamp = pnl_series_orig.index.max() if not pnl_series_orig.empty else None
            if last_timestamp is not None:
                # Sum PNL values for the last timestamp (as PNL is product-specific)
                final_pnl_original = pnl_series_orig.loc[last_timestamp].sum()
                if hasattr(final_pnl_original, 'item'):
                    final_pnl_original = final_pnl_original.item() # Ensure scalar
        st.session_state.permutation_results['original_pnl'] = final_pnl_original if pd.notna(final_pnl_original) else np.nan
        pnl_display_str = f"{final_pnl_original:.2f}" if pd.notna(final_pnl_original) else "N/A"
        st.success(f"Original backtest complete. Final PNL: {pnl_display_str}")

        # --- 2. Run Permutations (if enabled) ---
        if run_permutations_now:
            st.info(f"Starting {perm_n_iterations} permutations (Block: {perm_blocksize})...")
            perm_progress = st.progress(0.0, text="Preparing permutation data...")
            permuted_pnls_list = []
            log_vwap_changes, initial_log_vwap, relative_books_cache = None, None, None # Init vars

            # --- Prepare Base Data for Permutations ---
            try:
                with st.spinner("Preparing data for permutations..."):
                    print("Preparing base data for permutations...")
                    # Use market_data from the *original* backtester instance for consistency
                    # Ensure it's available after the original run
                    if not hasattr(backtester_instance, 'market_data') or backtester_instance.market_data.empty:
                         raise ValueError("Original backtester instance missing market_data needed for permutations.")
                    market_data_for_perm = backtester_instance.market_data.copy()
                    if 'product' not in market_data_for_perm.columns:
                        raise ValueError("'product' column missing.")

                    # Ensure VWAP exists or calculate it
                    if 'vwap' not in market_data_for_perm.columns or market_data_for_perm['vwap'].isna().all():
                        print("Recalculating VWAP for permutation base...")
                        price_cols = [f'{p}_{i}' for p in ['bid_price', 'ask_price'] for i in range(1, 4)]
                        vol_cols = [f'{p}_{i}' for p in ['bid_volume', 'ask_volume'] for i in range(1, 4)]
                        cols_to_cvt = [c for c in price_cols + vol_cols if c in market_data_for_perm.columns]
                        for col in cols_to_cvt:
                            if not pd.api.types.is_numeric_dtype(market_data_for_perm[col]):
                                market_data_for_perm[col] = pd.to_numeric(market_data_for_perm[col], errors='coerce')
                        vwap_s = market_data_for_perm.apply(calculate_row_vwap, axis=1)
                        market_data_for_perm['vwap'] = vwap_s
                        market_data_for_perm['vwap'] = market_data_for_perm.groupby('product')['vwap'].ffill().bfill()
                    if 'vwap' not in market_data_for_perm.columns or market_data_for_perm['vwap'].isna().any():
                        raise ValueError("VWAP calculation failed or resulted in NaNs.")

                    # Prepare VWAP series, log changes, initial value, and relative books
                    first_product = backtester_instance.products[0] # Use a consistent product
                    vwap_series_for_perm = market_data_for_perm[market_data_for_perm['product'] == first_product]['vwap'].copy()
                    # Ensure index alignment with original market data before processing
                    vwap_series_for_perm = vwap_series_for_perm.reindex(market_data_for_perm.index.unique()).ffill().bfill()
                    vwap_series_for_perm.dropna(inplace=True)
                    vwap_series_for_perm = vwap_series_for_perm[vwap_series_for_perm > 1e-6] # Drop zero/neg for log
                    if vwap_series_for_perm.empty:
                        raise ValueError(f"VWAP series empty/invalid for {first_product}.")

                    print("Calculating relative books...")
                    relative_books_cache = perm_utils.calculate_relative_order_books(market_data_for_perm, vwap_series_for_perm)
                    print("Calculating log VWAP changes...")
                    log_vwap_changes = perm_utils.get_log_vwap_changes(vwap_series_for_perm)
                    initial_log_vwap = np.log(vwap_series_for_perm.iloc[0])

                    # Store args needed for reruns
                    st.session_state.permutation_rerun_args = {
                        'trader_file': selected_trader_fname,
                        'data_files': selected_data_fnames[:1], # Only need first log for structure in rerun
                        'time_range': time_range_values,
                        'bot_behavior': bot_behavior, # Although unused in perm run, keep for signature
                        'ignore_limits': ignore_limits_checkbox,
                        'log_vwap_changes': log_vwap_changes, # Base data for permutation
                        'block_size': perm_blocksize,
                        'initial_log_vwap': initial_log_vwap,
                        'relative_books': relative_books_cache
                    }

            except Exception as prep_e:
                st.error(f"Error preparing data for permutations: {prep_e}")
                traceback.print_exc()
                st.stop()
            # --- End Prepare Data ---

            # --- Permutation Loop ---
            perm_progress.progress(0.0, text="Running permutations...")
            for i in range(perm_n_iterations):
                perm_index_for_seed = i # Use 0-based index for seeding
                print(f"\n--- Running Permutation {i+1}/{perm_n_iterations} (Seed: {perm_index_for_seed}) ---")
                perm_backtester = None # Ensure cleanup
                try:
                     np.random.seed(perm_index_for_seed) # Set seed
                     permuted_log_changes = perm_utils.block_permutation(log_vwap_changes, perm_blocksize)
                     reconstructed_log_vwap = perm_utils.reconstruct_log_vwap(initial_log_vwap, permuted_log_changes)
                     reconstructed_vwap = np.exp(reconstructed_log_vwap)
                     permuted_vwap_map = reconstructed_vwap.to_dict()
                     valid_perm_timestamps = reconstructed_vwap.index.to_numpy() # Timestamps for this perm run
                     permuted_book_cache = perm_utils.generate_permuted_order_book_cache(valid_perm_timestamps, permuted_vwap_map, relative_books_cache)

                     # Run backtest with permuted book, disable inferred book matching
                     perm_backtester = run_backtest(
                          selected_trader_fname, selected_data_fnames[:1], # Use only first log for structure
                          time_range_values, bot_behavior, ignore_limits_checkbox,
                          order_depth_override=permuted_book_cache, disable_inferred=True
                     )
                     # Extract PNL
                     if perm_backtester and hasattr(perm_backtester, 'output') and perm_backtester.output:
                         _, m_df_perm, _ = parse_backtest_output(perm_backtester.output)
                         final_pnl_perm = np.nan
                         if not m_df_perm.empty and 'profit_and_loss' in m_df_perm.columns:
                             pnl_series_perm = pd.to_numeric(m_df_perm['profit_and_loss'], errors='coerce')
                             last_ts_perm = pnl_series_perm.index.max() if not pnl_series_perm.empty else None
                             if last_ts_perm is not None:
                                  # Sum product PNLs at last step for this perm run
                                  final_pnl_perm = pnl_series_perm.loc[last_ts_perm].sum()
                                  if hasattr(final_pnl_perm, 'item'):
                                      final_pnl_perm = final_pnl_perm.item()
                         permuted_pnls_list.append(float(final_pnl_perm) if pd.notna(final_pnl_perm) else np.nan)
                         print(f"Permutation {i+1} Final PNL: {final_pnl_perm:.2f}")
                     else:
                         print(f"Warning: Permutation {i+1} failed or produced no output.")
                         permuted_pnls_list.append(np.nan)
                except Exception as loop_e:
                    print(f"ERROR in permutation loop {i+1}: {loop_e}")
                    traceback.print_exc()
                    permuted_pnls_list.append(np.nan)
                finally:
                     if perm_backtester: # Cleanup instance if it was created
                         del perm_backtester
                perm_progress.progress((i + 1) / perm_n_iterations, text=f"Running permutation {i+1}/{perm_n_iterations}")
            # --- End Loop ---

            # --- Calculate p-value ---
            valid_perm_pnls = [p for p in permuted_pnls_list if isinstance(p, (int, float)) and np.isfinite(p)]
            p_value = np.nan
            pnl_orig = st.session_state.permutation_results.get('original_pnl')
            if pd.isna(pnl_orig):
                print("Warning: Original PNL was NaN, cannot calculate p-value.")
            elif valid_perm_pnls:
                count_ge = sum(p >= pnl_orig for p in valid_perm_pnls)
                p_value = (count_ge + 1) / (len(valid_perm_pnls) + 1) # Avoid p=0
                print(f"Permutation p-value: {p_value:.4f} ({count_ge} >= {pnl_orig:.2f} out of {len(valid_perm_pnls)})")
            else:
                print("Warning: No valid permuted PNLs to calculate p-value.")
            st.session_state.permutation_results['permuted_pnls'] = valid_perm_pnls # Store only valid floats
            st.session_state.permutation_results['p_value'] = p_value
            perm_progress.empty()
            st.success("Permutation testing complete.")

    else:
        pass # Log viewer mode doesn't need action here


# --- Display Results ---
with rightcol:
    st.header("Results")
    display_data = False
    raw_output_content = ""
    source_description = ""
    s_logs, m_df, t_df = [], pd.DataFrame(), pd.DataFrame() # Default empty

    # Determine data source (original run data stored in parsed_log_data after run)
    if 'parsed_log_data' in st.session_state and isinstance(st.session_state.parsed_log_data, tuple) and len(st.session_state.parsed_log_data) == 4:
        s_logs_main, m_df_main, t_df_main, raw_content_main = st.session_state.parsed_log_data
        # Check if data actually exists (DataFrame might be empty if parsing failed or run was short)
        if m_df_main is not None and not m_df_main.empty:
            s_logs, m_df, t_df, raw_output_content = s_logs_main, m_df_main, t_df_main, raw_content_main
            display_data = True
            source_desc_key = st.session_state.last_loaded_log_file if is_viewer_mode else selected_trader_fname
            source_mode = "Log" if is_viewer_mode else "Original Backtest"
            source_description = f"Displaying {source_mode}: **{source_desc_key or 'N/A'}**"

    if not display_data and run_button_pressed: # If run was pressed but no data ended up in state
        st.warning("No data available to display. Check run status and potential errors above.")
    elif not display_data: # Initial state
        st.info("Configure settings and run backtest or load log using the controls on the left.")

    # --- Display Area ---
    if display_data:
        st.caption(source_description)
        perm_results = st.session_state.get('permutation_results', {})
        # Determine which tabs to show
        tabs_to_show = ["📊 Original Run Charts"] # Always show original
        # Show perm tab only if perm run AND has results list (even if empty)
        if not is_viewer_mode and perm_results and perm_results.get('permuted_pnls') is not None:
             tabs_to_show.append("🧪 Permutation Test")
        tabs_to_show.append("📜 Logs & Output") # Always show logs if data is displayed
        tab_objects = st.tabs(tabs_to_show)

        # --- Original Run Charts Tab ---
        with tab_objects[0]:
            st.subheader("Original Run / Loaded Log Charts")
            display_pnl_chart(m_df, "Original Run") # Use main data
            display_position_chart(t_df, "Original Run")
            display_norm_price_chart(m_df, "Original Run")
            display_fill_dist_chart(m_df, t_df, "Original Run", is_permutation_run=False) # Indicate not perm run

        # --- Permutation Test Tab ---
        perm_tab_index = -1
        if "🧪 Permutation Test" in tabs_to_show:
            perm_tab_index = tabs_to_show.index("🧪 Permutation Test")

        if perm_tab_index != -1:
            with tab_objects[perm_tab_index]:
                st.subheader("Permutation Test Results")
                orig_pnl_from_state = perm_results.get('original_pnl')
                orig_pnl_display = f"{orig_pnl_from_state:.2f}" if pd.notna(orig_pnl_from_state) else "N/A"
                p_val = perm_results.get('p_value')
                perm_pnls = perm_results.get('permuted_pnls', []) # This list only contains valid floats

                col1, col2 = st.columns(2)
                col1.metric("Original Final PNL", orig_pnl_display)
                col2.metric("p-value", f"{p_val:.4f}" if pd.notna(p_val) else "N/A")

                if perm_pnls: # Check if list of valid floats is not empty
                     st.markdown("###### PNL Distribution")
                     try:
                          bin_sz = max(1.0, (max(perm_pnls)-min(perm_pnls))/20) if len(perm_pnls)>1 else 1.0
                          fig_hist = ff.create_distplot([perm_pnls], ['Permuted PNLs'], bin_size=bin_sz, show_hist=True, show_rug=False)
                          fig_hist.update_layout(title="Distribution of Final PNLs", xaxis_title="Final PNL", yaxis_title="Density", height=400)
                          if pd.notna(orig_pnl_from_state):
                              fig_hist.add_vline(x=orig_pnl_from_state, line_width=2, line_dash="dash", line_color="red", annotation_text="Original PNL", annotation_position="top right")
                          st.plotly_chart(fig_hist, use_container_width=True)
                     except Exception as hist_e:
                          st.error(f"Could not create PNL distribution chart: {hist_e}")

                     # Show raw permuted PNLs (might include NaNs if runs failed)
                     raw_perm_pnls_from_state = st.session_state.permutation_results.get('permuted_pnls_raw', perm_pnls) # Prefer raw if available, else use filtered
                     with st.expander("Raw Permuted PNLs"):
                         st.dataframe(pd.Series(raw_perm_pnls_from_state, name="PNL"))
                else:
                     st.warning("No valid permutation PNLs were recorded to plot distribution.")

                # --- View Specific Permutation ---
                st.divider()
                st.subheader("View Specific Permutation Run")
                rerun_args = st.session_state.get('permutation_rerun_args')
                # Check perm_pnls (list of *valid* PNLs) to determine max_perms
                max_perms = len(perm_pnls) if perm_pnls else 0

                if max_perms == 0 or not rerun_args:
                    st.info("Run permutation test first or check for errors during runs.")
                else:
                    # Ensure view_index is within the valid range
                    current_view_index = st.session_state.get("view_perm_input", None)
                    if current_view_index is not None and (current_view_index < 1 or current_view_index > max_perms):
                         current_view_index = None # Reset if out of bounds

                    view_index = st.number_input(
                         f"Permutation Run to View (1-{max_perms})",
                         min_value=1, max_value=max_perms,
                         value=current_view_index, step=1,
                         key="view_perm_input"
                    )
                    if st.button("Load Selected Permutation Run", key="view_perm_button"):
                         if view_index is not None:
                             rerun_and_view_permutation(view_index, rerun_args)
                         else:
                             st.warning("Enter a permutation number.")

                    # Display if data loaded into viewed_permutation_data
                    if st.session_state.viewed_permutation_data and isinstance(st.session_state.viewed_permutation_data, tuple) and len(st.session_state.viewed_permutation_data) == 4 and not st.session_state.viewed_permutation_data[1].empty:
                         st.divider()
                         perm_view_idx = st.session_state.view_perm_input # Use the selected index
                         st.markdown(f"#### Charts for Permutation Run #{perm_view_idx}")
                         s_logs_k, m_df_k, t_df_k, _ = st.session_state.viewed_permutation_data
                         title = f"Permutation {perm_view_idx}"
                         # Call charting functions with permuted data
                         display_pnl_chart(m_df_k, title)
                         display_position_chart(t_df_k, title)
                         display_norm_price_chart(m_df_k, title)
                         display_fill_dist_chart(m_df_k, t_df_k, title, is_permutation_run=True) # Indicate it's a perm run

        # --- Logs & Output Tab ---
        logs_tab_index = tabs_to_show.index("📜 Logs & Output")
        with tab_objects[logs_tab_index]:
            st.subheader("Execution Logs")
            logs_to_display = s_logs # Default to original/loaded logs
            output_content_to_show = raw_output_content
            log_source_info = ""
            # Check if viewing a specific permutation run
            if st.session_state.viewed_permutation_data and isinstance(st.session_state.viewed_permutation_data, tuple) and len(st.session_state.viewed_permutation_data) == 4 and not st.session_state.viewed_permutation_data[1].empty:
                 logs_to_display = st.session_state.viewed_permutation_data[0]
                 output_content_to_show = st.session_state.viewed_permutation_data[3]
                 perm_view_idx_logs = st.session_state.get("view_perm_input", "N/A")
                 log_source_info = f"(Displaying logs for viewed permutation run #{perm_view_idx_logs})"
            st.caption(log_source_info)

            if logs_to_display:
                max_logs_to_show = 500
                display_logs_limited = logs_to_display[-max_logs_to_show:]
                if len(logs_to_display) > max_logs_to_show:
                    st.warning(f"Showing only last {max_logs_to_show} log entries.")
                # Display in reverse chronological order
                for i, log_entry in enumerate(reversed(display_logs_limited)):
                     ts = log_entry.get("timestamp", "N/A")
                     sbox = log_entry.get("sandboxLog", "")
                     lamb = log_entry.get("lambdaLog", "")
                     if not sbox.strip() and not lamb.strip(): # Skip empty entries
                         continue
                     with st.expander(f"Timestamp {ts}", expanded=(i==0)): # Expand latest
                         if lamb.strip():
                             st.text("Trader Output:")
                             st.code(lamb.strip(), language=None)
                         if sbox.strip():
                             st.text("Sandbox Messages:")
                             st.code(sbox.strip(), language=None)
            else:
                st.info("No execution logs available for this view.")

            st.divider()
            st.subheader("Raw Output & Download")
            if output_content_to_show:
                 st.text_area("Output Log Content:", output_content_to_show, height=400, key="output_log_area")
                 # Generate filename based on source
                 log_file_base = st.session_state.last_loaded_log_file if is_viewer_mode else (selected_data_fnames[0] if selected_data_fnames else 'multi_log')
                 trader_tag = "log_view" if is_viewer_mode else (selected_trader_fname.replace('.py','') if selected_trader_fname else "no_trader")
                 if st.session_state.viewed_permutation_data and isinstance(st.session_state.viewed_permutation_data, tuple) and len(st.session_state.viewed_permutation_data) == 4 and not st.session_state.viewed_permutation_data[1].empty:
                      perm_view_idx_fname = st.session_state.get("view_perm_input", "N/A")
                      trader_tag += f"_perm{perm_view_idx_fname}"
                 download_filename = f"output_{trader_tag}_{log_file_base.replace('.log','')}.log"
                 st.download_button(label="📥 Download Log", data=output_content_to_show, file_name=download_filename, mime="text/plain", use_container_width=True)
            else:
                 st.warning("No output content available.")

# --- END OF app.py ---