import torch
import numpy as np
import pandas as pd
from datetime import timedelta


def predict_n_days_ahead(model, prepared_data, n_days=5, device='cpu'):
    try:
        n_days = int(n_days)
    except (TypeError, ValueError):
        return None, None, "WARNING: Forecast horizon must be a number."

    if n_days > 40:
        return None, None, "WARNING: Model is only valid up to 20 days ahead. Please reduce your prediction horizon."

    target_scaler = prepared_data["target_scaler"]
    X_val = prepared_data["X_val"]
    last_window = X_val[-1].copy()

    model.eval()
    predictions = []

    current_window = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).to(device)

    for day in range(n_days):
        with torch.no_grad():
            pred = model(current_window)

        pred_value = pred.cpu().numpy().flatten()[0]
        predictions.append(pred_value)

        new_row = current_window[0, -1, :].cpu().numpy().copy()
        new_row[0] = pred_value

        new_window = current_window[0, 1:, :].cpu().numpy()
        new_row = new_row.reshape(1, -1)
        new_window = np.vstack([new_window, new_row])

        current_window = torch.tensor(new_window, dtype=torch.float32).unsqueeze(0).to(device)

    predictions = np.array(predictions).reshape(-1, 1)
    predictions_real = target_scaler.inverse_transform(predictions).flatten()

    # Generate future dates
    last_date_raw = prepared_data["dates"][-1]
    last_date = pd.to_datetime(last_date_raw, errors="coerce")
    if pd.isna(last_date):
        return None, None, f"WARNING: Invalid last date in prepared data ({last_date_raw!r})."

    start_date = last_date + timedelta(days=1)
    future_dates = pd.bdate_range(start=start_date, periods=n_days).strftime('%Y-%m-%d').tolist()

    return predictions_real.tolist(), future_dates, None


def get_validation_actuals_and_preds(model, prepared_data, device='cpu'):
    target_scaler = prepared_data["target_scaler"]
    X_val = prepared_data["X_val"]
    y_val = prepared_data["y_val"]

    model.eval()

    X_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()

    actuals = target_scaler.inverse_transform(y_val).flatten().tolist()
    preds = target_scaler.inverse_transform(preds).flatten().tolist()

    # Get validation dates
    dates = prepared_data["dates"][-(len(actuals)):].tolist()
    dates = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in dates]

    return actuals, preds, dates