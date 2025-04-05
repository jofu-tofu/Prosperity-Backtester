# --- START OF util.py (Revised Parser) ---
# backtester/util.py
import importlib.util
import inspect
import json
import pandas as pd
import io
from typing import Any, Dict, List, Tuple, TextIO
import os
import traceback

def get_trader(full_filepath: str) -> Any:
    """Dynamically loads the Trader class from a Python file using its full path."""
    if not os.path.exists(full_filepath):
         raise FileNotFoundError(f"Trader file not found at: {full_filepath}")

    spec = importlib.util.spec_from_file_location("trader_module", full_filepath)
    if spec is None or spec.loader is None:
         raise ImportError(f"Could not load trader from {full_filepath}")
    trader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trader_module)

    for name, obj in inspect.getmembers(trader_module):
        if inspect.isclass(obj) and name == "Trader":
            print(f"Trader Class Found: {name}")
            return obj()
    raise AttributeError(f"No class named 'Trader' found in {full_filepath}")


def _parse_data(data_source: TextIO) -> Tuple[List[Dict], pd.DataFrame, pd.DataFrame]:
    """Parses the log file into sandbox logs, market data, and trade history."""
    lines = data_source.readlines()
    sandbox_logs = []
    market_data_lines = []
    trade_history_lines = []
    trade_history_start = -1
    activity_log_start = -1
    market_data_header_line = "" # Store the header

    # Find markers and header
    for i, line in enumerate(lines):
        if '"sandboxLog"' in line and '"lambdaLog"' in line :
            try:
                 start_idx = line.find('{')
                 end_idx = line.rfind('}')
                 if start_idx != -1 and end_idx != -1:
                     json_str = line[start_idx : end_idx + 1]
                     log_entry = json.loads(json_str)
                     sandbox_logs.append(log_entry)
                 else:
                      print(f"Warning: Could not parse JSON from sandbox log line: {line.strip()}")
            except json.JSONDecodeError:
                print(f"Warning: JSON parse fail: {line.strip()}")
                sandbox_logs.append({"sandboxLog": "ERROR PARSING", "lambdaLog": line.strip(), "timestamp": "ERROR"})

        elif "Activities log:" in line:
            if i + 1 < len(lines): # Check if there's a line after the marker
                activity_log_start = i + 2 # Data starts after header
                market_data_header_line = lines[i+1].strip() # Store the header line
            else:
                activity_log_start = -1 # No header/data found

        elif "Trade History:" in line:
             trade_history_start = i + 1
             if activity_log_start > 0: # Check if Activities log section had started
                market_data_lines = lines[activity_log_start:i]
                activity_log_start = -2 # Mark as processed

    # Handle case where Trade History section might be missing
    if activity_log_start > 0: # If Activities started but wasn't marked as processed
        market_data_lines = lines[activity_log_start:]

    # Process Market Data CSV
    market_data_df = pd.DataFrame()
    if market_data_lines and market_data_header_line:
         # Prepend the header to the data lines
         full_csv_content = market_data_header_line + "\n" + "".join(market_data_lines)
         market_csv_io = io.StringIO(full_csv_content)
         try:
             print("DEBUG util: Attempting to parse market data CSV...")
             # Let pandas infer dtypes initially, then convert timestamp
             market_data_df = pd.read_csv(market_csv_io, sep=";", index_col=False, low_memory=False)

             # --- Post-processing and Validation ---
             if 'timestamp' in market_data_df.columns:
                 # Convert timestamp to numeric, coercing errors, then drop NaNs from timestamp
                 market_data_df['timestamp'] = pd.to_numeric(market_data_df['timestamp'], errors='coerce')
                 market_data_df.dropna(subset=['timestamp'], inplace=True)
                 market_data_df['timestamp'] = market_data_df['timestamp'].astype(int)
                 market_data_df.set_index('timestamp', inplace=True)
                 print(f"DEBUG util: Parsed market data shape: {market_data_df.shape}")
                 # print(f"DEBUG util: Parsed market data columns: {market_data_df.columns.tolist()}")
                 # print(f"DEBUG util: Parsed market data head:\n{market_data_df.head().to_string()}")
             else:
                  print("ERROR util: 'timestamp' column not found after parsing market data CSV.")
                  market_data_df = pd.DataFrame() # Reset to empty if essential column missing

         except pd.errors.ParserError as pe:
              print(f"FATAL pandas.errors.ParserError parsing Activities log CSV: {pe}")
              print("--- CSV Header Received by Parser ---")
              print(market_data_header_line)
              print("--- First 10 lines of Data Received by Parser ---")
              print("".join(market_data_lines[:10]))
              print("--- End Failed CSV Content ---")
              market_data_df = pd.DataFrame() # Reset to empty on error
         except Exception as csv_e:
              print(f"ERROR parsing Activities log CSV: {csv_e}")
              traceback.print_exc() # Print full traceback
              market_data_df = pd.DataFrame()
    elif not market_data_header_line:
        print("Warning util: Could not find Activities log header line.")
    else: # Header found, but no data lines
        print("Warning util: Found Activities log header, but no data lines followed.")


    # Process Trade History
    trade_history_df = pd.DataFrame()
    if trade_history_start != -1:
        trade_history_str = "".join(lines[trade_history_start:])
        # Clean potential leading/trailing whitespace/newlines before parsing JSON
        trade_history_str = trade_history_str.strip()
        if not trade_history_str:
            print("Warning util: Trade History section found but is empty.")
        else:
            try:
                trade_history_json = json.loads(trade_history_str)
                if isinstance(trade_history_json, list): # Expecting a list of trade dicts
                    trade_history_df = pd.DataFrame(trade_history_json)
                    if 'timestamp' in trade_history_df.columns:
                        trade_history_df['timestamp'] = pd.to_numeric(trade_history_df['timestamp'], errors='coerce').fillna(0).astype(int)
                        trade_history_df.set_index('timestamp', inplace=True)
                    else:
                        print("Warning util: 'timestamp' column not found in trade history list.")
                        trade_history_df = pd.DataFrame()
                else:
                     print(f"Warning util: Trade History JSON was not a list (type: {type(trade_history_json)}).")
                     trade_history_df = pd.DataFrame()

            except json.JSONDecodeError as json_e:
                print(f"Error parsing Trade History JSON: {json_e}")
                print("--- JSON Content That Failed ---")
                print(trade_history_str[:1000]) # Print start of potentially bad JSON
                print("--- End Failed JSON Content ---")
            except Exception as trade_e:
                print(f"Error processing Trade History DataFrame: {trade_e}")

    return sandbox_logs, market_data_df, trade_history_df
# --- END OF util.py ---