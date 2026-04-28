import os
import sys
import torch

# Add Experiments to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "Experiments"))

from Architecture import LSTMModel
from helpers import prepare_dataset_once


# Map saved model files to stock info
MODEL_MAPPING = {
    "META": {"name": "Meta Platforms", "file": "META_lstm.pt"},
    "MSFT": {"name": "Microsoft", "file": "MSFT_lstm.pt"},
    "GC=F": {"name": "Gold Futures", "file": "GCF_lstm.pt"},
    "HPQ": {"name": "HP Inc.", "file": "HPQ_lstm.pt"},
    "NVDA": {"name": "NVIDIA", "file": "NVDA_lstm.pt"},
}

STOCKS = [
    {"ticker": "META",    "name": "Meta Platforms"},
    {"ticker": "MSFT",    "name": "Microsoft"},
    {"ticker": "GC=F",    "name": "Gold Futures"},
    {"ticker": "HPQ",     "name": "HP Inc."},
    {"ticker": "NVDA",    "name": "NVIDIA"},
]

# Cache loaded models and data
loaded_models = {}
loaded_data = {}

SAVED_MODELS_DIR = os.path.join(BASE_DIR, "Data Exploration", "saved_models")


def _looks_like_state_dict(obj):
    return isinstance(obj, dict) and any(
        isinstance(v, torch.Tensor) for v in obj.values()
    )


def _normalize_state_dict_keys(state_dict):
    """Strip common wrapper prefixes added by training frameworks."""
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model.", "network."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        normalized[new_key] = value
    return normalized


def _extract_state_dict(loaded_obj):
    """Find a valid tensor state_dict from common checkpoint formats."""
    if _looks_like_state_dict(loaded_obj):
        return loaded_obj

    if isinstance(loaded_obj, dict):
        # Common checkpoint keys used by custom training loops.
        for key in (
            "state_dict",
            "model_state_dict",
            "net_state_dict",
            "weights",
            "model_weights",
            "model",
        ):
            if key in loaded_obj and _looks_like_state_dict(loaded_obj[key]):
                return loaded_obj[key]

        # Some checkpoints nest the state dict one level deeper.
        for value in loaded_obj.values():
            if isinstance(value, dict):
                nested = _extract_state_dict(value)
                if nested is not None:
                    return nested

    return None


def load_model_for_stock(ticker, name):
    """Load model from saved .pt file for a given stock."""
    if ticker in loaded_models:
        return loaded_models[ticker]

    if ticker not in MODEL_MAPPING:
        raise ValueError(f"No saved model found for ticker {ticker}")

    model_file = MODEL_MAPPING[ticker]["file"]
    model_path = os.path.join(SAVED_MODELS_DIR, model_file)

    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")

    try:
        # Saved files may be either a full model or many checkpoint dict variants.
        loaded_obj = torch.load(model_path, map_location="cpu")

        if isinstance(loaded_obj, torch.nn.Module):
            model = loaded_obj
        else:
            state_dict = _extract_state_dict(loaded_obj)

            if not isinstance(state_dict, dict):
                raise ValueError(f"Unsupported model format in {model_file}")

            state_dict = _normalize_state_dict_keys(state_dict)

            required_keys = ("lstm.weight_ih_l0", "lstm.weight_hh_l0", "fc.weight")
            if not all(k in state_dict for k in required_keys):
                available = list(state_dict.keys())[:10]
                raise ValueError(
                    f"Unsupported state_dict layout in {model_file}. "
                    f"Missing LSTM keys. Sample keys: {available}"
                )

            # Infer architecture directly from saved tensor shapes.
            input_size = state_dict["lstm.weight_ih_l0"].shape[1]
            hidden_size = state_dict["lstm.weight_hh_l0"].shape[1]
            num_layers = len([k for k in state_dict.keys() if k.startswith("lstm.weight_ih_l")])
            output_size = state_dict["fc.weight"].shape[0]

            model = LSTMModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size
            )
            model.load_state_dict(state_dict)

        model.eval()
        loaded_models[ticker] = model
        print(f"[{ticker}] Model loaded from {model_file}")
        return model
    except Exception as e:
        raise ValueError(f"Failed to load model for {ticker}: {e}")


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