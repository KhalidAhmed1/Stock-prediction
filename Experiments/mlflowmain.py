import os
import sys
import mlflow

# Fix import path
sys.path.insert(0, os.path.dirname(__file__))
from helpers import *

# MLflow tracking URI
MLFLOW_DIR = os.path.join(os.path.dirname(__file__), "mlruns")
os.makedirs(MLFLOW_DIR, exist_ok=True)
mlflow.set_tracking_uri("file:///" + MLFLOW_DIR.replace("\\", "/"))

# Stock Configuration
STOCKS = [
    {"ticker": "META",    "name": "Meta Platforms"},
    {"ticker": "MSFT",    "name": "Microsoft"},
    {"ticker": "GC=F",    "name": "Gold Futures"},
    {"ticker": "JUFU.TO", "name": "Jushi Holdings (TSX)"},
    {"ticker": "SWEDY",   "name": "Schneider Electric ADR"},
    {"ticker": "HPQ",     "name": "HP Inc."},
    {"ticker": "NVDA",    "name": "NVIDIA"},
]

# Hyperparameters
WINDOW_SIZE   = 60
TEST_SIZE     = 0.1
HIDDEN_SIZE   = 32
NUM_LAYERS    = 1
BATCH_SIZE    = 32
EPOCHS        = 20
LEARNING_RATE = 4e-3
PERIOD        = "5y"


def start_experiments_for_all_stocks(stocks):
    """
    Run hyperparameter optimization experiments for a list of stocks.

    This function iterates over each stock in the provided list and runs
    an Optuna optimization process using a helper function. Results are
    tracked using MLflow.

    Args:
        stocks (list of dict): List of stock configurations, where each dictionary contains:
            - 'ticker' (str): Stock ticker symbol
            - 'name' (str): Human-readable stock name

    Returns:
        None

    Behavior:
        - Executes experiments sequentially for each stock
        - Logs successful experiment completion
        - Catches and logs exceptions without stopping the loop

    Example:
        start_experiments_for_all_stocks(STOCKS)
    """
    for stock in stocks:
        try:
            study, prepared = run_optuna_for_stock(
                stock_ticker=stock['ticker'],
                stock_name=stock['name'],
                device='cpu',
                n_trials=5,
                period=PERIOD
            )
            print(f"Experiment on {stock['ticker']} has finished!!!")
        except Exception as e:
            print(f"Experiment on {stock['ticker']} FAILED: {e}")
            continue


if __name__ == "__main__":
    print("The function is working")
    start_experiments_for_all_stocks(STOCKS)