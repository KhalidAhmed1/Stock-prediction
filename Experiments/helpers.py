import os
import sys
import mlflow
import mlflow.pytorch
import optuna
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(__file__))
from Architecture import LSTMModel


def create_stock_experiment(stock_ticker, stock_name=None, base_name="LSTM_Stock_Forecasting"):
    if stock_name:
        experiment_name = f"{base_name}/{stock_ticker}_{stock_name.replace(' ', '_')}"
    else:
        experiment_name = f"{base_name}/{stock_ticker}"

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = mlflow.create_experiment(experiment_name)
        print(f"Created experiment: {experiment_name}")
    else:
        exp_id = exp.experiment_id
        print(f"Using existing experiment: {experiment_name}")

    mlflow.set_experiment(experiment_name)
    return experiment_name, exp_id


def prepare_dataset_once(stock_ticker, period="20y", window_size=60, test_size=0.1):
    df = yf.download(stock_ticker, period=period, auto_adjust=False, progress=False)

    if df.empty:
        raise ValueError(f"No data found for {stock_ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()

    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['return'] = df['Close'].pct_change()
    df['ma20'] = df['Close'].rolling(20).mean()
    df['ma50'] = df['Close'].rolling(50).mean()
    df['volatility'] = df['return'].rolling(20).std()
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    FEATURES = ['Close', 'Day_of_Week', 'Month', 'Volume', 'return', 'ma20', 'ma50', 'volatility', 'rsi']
    df = df[FEATURES].dropna().copy()

    if len(df) <= window_size:
        raise ValueError(f"Not enough data for {stock_ticker} after preprocessing")

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    scaled_features = feature_scaler.fit_transform(df[FEATURES])
    scaled_target = target_scaler.fit_transform(df[['Close']])

    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(scaled_features[i - window_size:i])
        y.append(scaled_target[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    split_idx = int(len(X) * (1 - test_size))

    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]

    prepared_data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "target_scaler": target_scaler,
        "input_size": X.shape[2],
        "window_size": window_size
    }

    return prepared_data


def make_loaders_from_prepared(prepared_data, batch_size):
    train_dataset = TensorDataset(
        torch.tensor(prepared_data["X_train"], dtype=torch.float32),
        torch.tensor(prepared_data["y_train"], dtype=torch.float32)
    )

    val_dataset = TensorDataset(
        torch.tensor(prepared_data["X_val"], dtype=torch.float32),
        torch.tensor(prepared_data["y_val"], dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_predictions(model, data_loader, target_scaler, device):
    model.eval()

    preds = []
    actuals = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            output = model(batch_X).cpu().numpy()
            actual = batch_y.cpu().numpy()

            preds.append(output)
            actuals.append(actual)

    preds = np.vstack(preds)
    actuals = np.vstack(actuals)

    preds = target_scaler.inverse_transform(preds)
    actuals = target_scaler.inverse_transform(actuals)

    return actuals.flatten(), preds.flatten()


def save_loss_curve_matplotlib(train_losses, val_losses, stock_ticker, trial_number):
    filename = f"loss_curve_{stock_ticker}_trial_{trial_number}.png"

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"Loss Curve - {stock_ticker} - Trial {trial_number}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

    return filename


def objective(trial, prepared_data, stock_ticker, stock_name, device):
    params = {
        "hidden_size": trial.suggest_int("hidden_size", 16, 128),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "epochs": 10,
        "window_size": 60
    }

    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.set_tag("stock_ticker", stock_ticker)
        mlflow.set_tag("stock_name", stock_name)
        mlflow.set_tag("run_type", "optuna_trial")

        train_loader, val_loader = make_loaders_from_prepared(
            prepared_data=prepared_data,
            batch_size=params["batch_size"]
        )

        model = LSTMModel(
            prepared_data["input_size"],
            params["hidden_size"],
            params["num_layers"],
            1
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

        train_losses = []
        val_losses = []

        print("------------------------------------------------------------------------------")
        print(f"Start Trial {trial.number} for {stock_ticker}")
        print("------------------------------------------------------------------------------")

        for epoch in range(params["epochs"]):
            model.train()
            total_train_loss = 0.0

            train_pbar = tqdm(
                train_loader,
                desc=f"Trial {trial.number} | Epoch {epoch + 1}/{params['epochs']}",
                leave=False
            )

            for batch_idx, (batch_X, batch_y) in enumerate(train_pbar):
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

                running_avg_loss = total_train_loss / (batch_idx + 1)

                train_pbar.set_postfix({
                    "batch_loss": f"{loss.item():.6f}",
                    "train_avg": f"{running_avg_loss:.6f}"
                })

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            total_val_loss = 0.0

            with torch.no_grad():
                for val_X, val_y in val_loader:
                    val_X = val_X.to(device)
                    val_y = val_y.to(device)

                    val_output = model(val_X)
                    val_loss = criterion(val_output, val_y)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            print(f"Trial {trial.number} | Epoch {epoch + 1}/{params['epochs']} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        actuals, preds = get_predictions(
            model=model,
            data_loader=val_loader,
            target_scaler=prepared_data["target_scaler"],
            device=device
        )

        mse = mean_squared_error(actuals, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, preds)
        r2 = r2_score(actuals, preds)

        mlflow.log_metrics({
            "final_mse": mse,
            "final_rmse": rmse,
            "final_mae": mae,
            "final_r2": r2
        })

        mlflow.pytorch.log_model(model, artifact_path="model")

        loss_file = save_loss_curve_matplotlib(
            train_losses=train_losses,
            val_losses=val_losses,
            stock_ticker=stock_ticker,
            trial_number=trial.number
        )

        mlflow.log_artifact(loss_file)

        if os.path.exists(loss_file):
            os.remove(loss_file)

        return avg_val_loss


def train_best_model(prepared_data, stock_ticker, stock_name, best_params, device):
    train_loader, val_loader = make_loaders_from_prepared(
        prepared_data=prepared_data,
        batch_size=best_params["batch_size"]
    )

    model = LSTMModel(
        prepared_data["input_size"],
        best_params["hidden_size"],
        best_params["num_layers"],
        1
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])

    train_losses = []
    val_losses = []

    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X = val_X.to(device)
                val_y = val_y.to(device)

                val_output = model(val_X)
                val_loss = criterion(val_output, val_y)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

    actuals, preds = get_predictions(
        model=model,
        data_loader=val_loader,
        target_scaler=prepared_data["target_scaler"],
        device=device
    )

    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)

    return model, train_losses, val_losses, mse, rmse, mae, r2


def run_optuna_for_stock(stock_ticker, stock_name, device, n_trials=20, period="5y"):
    create_stock_experiment(stock_ticker, stock_name)

    prepared_data = prepare_dataset_once(
        stock_ticker=stock_ticker,
        period=period,
        window_size=60,
        test_size=0.1
    )

    with mlflow.start_run(run_name=f"{stock_ticker}_optuna_parent"):
        mlflow.set_tag("stock_ticker", stock_ticker)
        mlflow.set_tag("stock_name", stock_name)
        mlflow.set_tag("run_type", "optuna_parent")

        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("window_size", 60)
        mlflow.log_param("period", period)

        study = optuna.create_study(
            direction="minimize",
            study_name=f"study_{stock_ticker}",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )

        study.optimize(
            lambda trial: objective(
                trial=trial,
                prepared_data=prepared_data,
                stock_ticker=stock_ticker,
                stock_name=stock_name,
                device=device
            ),
            n_trials=n_trials
        )

        mlflow.log_metric("best_val_loss", study.best_value)

        for k, v in study.best_params.items():
            mlflow.log_param(f"best_{k}", v)

        mlflow.set_tag("best_trial_number", study.best_trial.number)

        best_model, train_losses, val_losses, mse, rmse, mae, r2 = train_best_model(
            prepared_data=prepared_data,
            stock_ticker=stock_ticker,
            stock_name=stock_name,
            best_params=study.best_params,
            device=device
        )

        mlflow.pytorch.log_model(best_model, artifact_path="best_model")

        mlflow.log_metrics({
            "best_model_mse": mse,
            "best_model_rmse": rmse,
            "best_model_mae": mae,
            "best_model_r2": r2
        })

        loss_file = save_loss_curve_matplotlib(
            train_losses=train_losses,
            val_losses=val_losses,
            stock_ticker=stock_ticker,
            trial_number="best_model"
        )

        mlflow.log_artifact(loss_file)

        if os.path.exists(loss_file):
            os.remove(loss_file)

    return study, prepared_data