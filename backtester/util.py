# backtester/util.py
import importlib.util
import inspect
import json
import pandas as pd
import io
from typing import Any, Dict, List, Tuple, TextIO
import os
# Function to load trader class dynamically
def get_trader(full_filepath: str) -> Any: # Changed parameter name for clarity
    """Dynamically loads the Trader class from a Python file using its full path."""
    if not os.path.exists(full_filepath):
         raise FileNotFoundError(f"Trader file not found at: {full_filepath}")

    spec = importlib.util.spec_from_file_location("trader_module", full_filepath) # Use full_filepath
    if spec is None or spec.loader is None:
         raise ImportError(f"Could not load trader from {full_filepath}")
    trader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trader_module)

    # Find the class inheriting from object (or a specific base class if provided)
    for name, obj in inspect.getmembers(trader_module):
        if inspect.isclass(obj) and name == "Trader": # Simple check, might need refinement
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

    for i, line in enumerate(lines):
        if '"sandboxLog"' in line and '"lambdaLog"' in line :
            try:
                 # Find the start and end of the JSON object reliably
                 start_idx = line.find('{')
                 end_idx = line.rfind('}')
                 if start_idx != -1 and end_idx != -1:
                     json_str = line[start_idx : end_idx + 1]
                     log_entry = json.loads(json_str)
                     sandbox_logs.append(log_entry)
                 else:
                      print(f"Warning: Could not parse JSON from sandbox log line: {line.strip()}")

            except json.JSONDecodeError:
                # Handle potential JSON errors if format is inconsistent
                print(f"Warning: Could not parse JSON from sandbox log line: {line.strip()}")
                sandbox_logs.append({"sandboxLog": "ERROR PARSING", "lambdaLog": line.strip(), "timestamp": "ERROR"}) # Placeholder

        elif "Activities log:" in line:
            activity_log_start = i + 1
        elif "Trade History:" in line:
             trade_history_start = i + 1
             # End market data processing
             if activity_log_start != -1:
                market_data_lines = lines[activity_log_start:i]

    if activity_log_start != -1 and trade_history_start == -1: # Handle case where Trade History section is missing
        market_data_lines = lines[activity_log_start:]

    if trade_history_start != -1:
        trade_history_str = "".join(lines[trade_history_start:])
        try:
            trade_history_json = json.loads(trade_history_str)
        except json.JSONDecodeError:
            print("Error parsing Trade History JSON.")
            trade_history_json = []
    else:
        trade_history_json = []

    # Process Market Data CSV
    if not market_data_lines:
         print("Warning: No Market Data found in the log.")
         market_data_df = pd.DataFrame()
    else:
         market_csv_io = io.StringIO("".join(market_data_lines))
         # Make sure timestamps are parsed correctly - sometimes they are floats
         market_data_df = pd.read_csv(market_csv_io, sep=";", index_col="timestamp", converters={'timestamp': int})
         market_data_df.index = market_data_df.index.astype(int) # Ensure integer index


    # Process Trade History
    if not trade_history_json:
         print("Warning: No Trade History found in the log.")
         trade_history_df = pd.DataFrame()
    else:
         trade_history_df = pd.DataFrame(trade_history_json)
         if 'timestamp' in trade_history_df.columns:
             trade_history_df['timestamp'] = pd.to_numeric(trade_history_df['timestamp'], errors='coerce').fillna(0).astype(int)
             trade_history_df = trade_history_df.set_index('timestamp')
         else:
              print("Warning: 'timestamp' column not found in trade history.")
              trade_history_df = pd.DataFrame() # Create empty if timestamp missing


    return sandbox_logs, market_data_df, trade_history_df