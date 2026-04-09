import os
import sys
import mlflow
import torch
import numpy as np

# Point to your mlruns folder
MLFLOW_DIR = os.path.join(os.path.dirname(__file__), "Experiments", "mlruns")
mlflow.set_tracking_uri("file:///" + MLFLOW_DIR.replace("\\", "/"))

# Add Experiments to path so Architecture.py can be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Experiments"))
from Architecture import LSTMModel


def get_best_model_for_stock(stock_ticker, stock_name):
    """
    Find the parent run for a stock, then load the best_model artifact.
    """
    experiment_name = f"LSTM_Stock_Forecasting/{stock_ticker}_{stock_name.replace(' ', '_')}"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"No experiment found: {experiment_name}")

    # Search for parent run
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.run_type = 'optuna_parent'",
        order_by=["metrics.best_val_loss ASC"],
        max_results=1
    )

    if runs.empty:
        raise ValueError(f"No parent run found for {stock_ticker}")

    best_run_id = runs.iloc[0]["run_id"]
    print(f"[{stock_ticker}] Best parent run ID: {best_run_id}")
    print(f"[{stock_ticker}] Best val loss: {runs.iloc[0]['metrics.best_val_loss']}")

    # Load the best model
    model_uri = f"runs:/{best_run_id}/best_model"
    model = mlflow.pytorch.load_model(model_uri)

    return model, best_run_id


def get_best_params_for_stock(stock_ticker, stock_name):
    """
    Get the best hyperparameters for a stock.
    """
    experiment_name = f"LSTM_Stock_Forecasting/{stock_ticker}_{stock_name.replace(' ', '_')}"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"No experiment found: {experiment_name}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.run_type = 'optuna_parent'",
        max_results=1
    )

    if runs.empty:
        raise ValueError(f"No parent run found for {stock_ticker}")

    row = runs.iloc[0]

    params = {
        "hidden_size": int(row["params.best_hidden_size"]),
        "num_layers": int(row["params.best_num_layers"]),
        "lr": float(row["params.best_lr"]),
        "batch_size": int(row["params.best_batch_size"]),
    }

    metrics = {
        "best_val_loss": row["metrics.best_val_loss"],
        "best_model_mse": row["metrics.best_model_mse"],
        "best_model_rmse": row["metrics.best_model_rmse"],
        "best_model_mae": row["metrics.best_model_mae"],
        "best_model_r2": row["metrics.best_model_r2"],
    }

    return params, metrics


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":

    STOCKS = [
        {"ticker": "META",    "name": "Meta Platforms"},
        {"ticker": "MSFT",    "name": "Microsoft"},
        {"ticker": "GC=F",    "name": "Gold Futures"},
        {"ticker": "JUFU.TO", "name": "Jushi Holdings (TSX)"},
        {"ticker": "SWEDY",   "name": "Schneider Electric ADR"},
        {"ticker": "HPQ",     "name": "HP Inc."},
        {"ticker": "NVDA",    "name": "NVIDIA"},
    ]

    for stock in STOCKS:
        ticker = stock["ticker"]
        name = stock["name"]

        try:
            # Load best model
            model, run_id = get_best_model_for_stock(ticker, name)
            print(f"[{ticker}] Model loaded successfully!")

            # Get best params and metrics
            params, metrics = get_best_params_for_stock(ticker, name)
            print(f"[{ticker}] Best params: {params}")
            print(f"[{ticker}] Best metrics: {metrics}")

            # Model is ready to use for inference
            model.eval()

            # Example: dummy input (batch=1, seq_len=60, features=9)
            dummy_input = torch.randn(1, 60, 9)
            with torch.no_grad():
                prediction = model(dummy_input)
            print(f"[{ticker}] Sample prediction shape: {prediction.shape}")
            print(f"[{ticker}] Sample prediction value: {prediction.item():.6f}")
            print()

        except Exception as e:
            print(f"[{ticker}] FAILED: {e}")
            print()