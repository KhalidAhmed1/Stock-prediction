import os
import sys
import mlflow
import mlflow.pytorch

# Add Experiments to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "Experiments"))

MLFLOW_DIR = os.path.join(BASE_DIR, "Experiments", "mlruns")
mlflow.set_tracking_uri("file:///" + MLFLOW_DIR.replace("\\", "/"))

from Architecture import LSTMModel
from helpers import prepare_dataset_once


STOCKS = [
    {"ticker": "META",    "name": "Meta Platforms"},
    {"ticker": "MSFT",    "name": "Microsoft"},
    {"ticker": "GC=F",    "name": "Gold Futures"},
    {"ticker": "JUFU.TO", "name": "Jushi Holdings (TSX)"},
    {"ticker": "SWEDY",   "name": "Schneider Electric ADR"},
    {"ticker": "HPQ",     "name": "HP Inc."},
    {"ticker": "NVDA",    "name": "NVIDIA"},
]

# Cache loaded models and data
loaded_models = {}
loaded_data = {}


def get_experiment_name(ticker, name):
    return f"LSTM_Stock_Forecasting/{ticker}_{name.replace(' ', '_')}"


def load_model_for_stock(ticker, name):
    """Load the best model from MLflow for a given stock."""
    if ticker in loaded_models:
        return loaded_models[ticker]

    experiment_name = get_experiment_name(ticker, name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"No experiment found: {experiment_name}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.run_type = 'optuna_parent'",
        order_by=["metrics.best_val_loss ASC"],
        max_results=1
    )

    if runs.empty:
        raise ValueError(f"No parent run found for {ticker}")

    run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{run_id}/best_model"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    loaded_models[ticker] = model
    print(f"[{ticker}] Model loaded from run {run_id}")

    return model


def load_data_for_stock(ticker, period="5y"):
    """Load and prepare data for a stock. Cached."""
    if ticker in loaded_data:
        return loaded_data[ticker]

    prepared = prepare_dataset_once(
        stock_ticker=ticker,
        period=period,
        window_size=60,
        test_size=0.1
    )

    loaded_data[ticker] = prepared
    print(f"[{ticker}] Data prepared and cached.")

    return prepared


def load_all_models():
    """Preload all models and data at startup."""
    for stock in STOCKS:
        try:
            load_model_for_stock(stock["ticker"], stock["name"])
            load_data_for_stock(stock["ticker"])
        except Exception as e:
            print(f"[{stock['ticker']}] Failed to load: {e}")