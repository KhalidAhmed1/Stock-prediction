import os
import sys
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Fix paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "Experiments"))
sys.path.insert(0, os.path.dirname(__file__))

from model_loader import STOCKS, load_model_for_stock, load_data_for_stock, load_all_models
from predictor import predict_n_days_ahead, get_validation_actuals_and_preds

app = FastAPI(title="Stock LSTM Predictor")

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


@app.on_event("startup")
async def startup():
    print("Loading all models...")
    load_all_models()
    print("All models loaded.")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"stocks": STOCKS}
    )


@app.get("/api/stocks")
async def get_stocks():
    return JSONResponse(content={"stocks": STOCKS})


@app.get("/api/visualize/{ticker}")
async def visualize(ticker: str):
    stock = next((s for s in STOCKS if s["ticker"] == ticker), None)
    if stock is None:
        return JSONResponse(content={"error": f"Stock {ticker} not found"}, status_code=404)

    try:
        model = load_model_for_stock(ticker, stock["name"])
        data = load_data_for_stock(ticker)
        actuals, preds, dates = get_validation_actuals_and_preds(model, data)

        return JSONResponse(content={
            "ticker": ticker,
            "name": stock["name"],
            "actuals": actuals,
            "predictions": preds,
            "dates": dates
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/predict/{ticker}/{n_days}")
async def predict_ahead(ticker: str, n_days: int):
    stock = next((s for s in STOCKS if s["ticker"] == ticker), None)
    if stock is None:
        return JSONResponse(content={"error": f"Stock {ticker} not found"}, status_code=404)

    if n_days > 40:
        return JSONResponse(content={
            "error": "WARNING: The model is only valid up to 40 days ahead. Please reduce your prediction horizon.",
            "warning": True
        }, status_code=400)

    if n_days < 1:
        return JSONResponse(content={"error": "Minimum 1 day"}, status_code=400)

    try:
        model = load_model_for_stock(ticker, stock["name"])
        data = load_data_for_stock(ticker)

        predictions, future_dates, warning = predict_n_days_ahead(model, data, n_days=n_days)

        if warning:
            return JSONResponse(content={"error": warning, "warning": True}, status_code=400)

        actuals, val_preds, val_dates = get_validation_actuals_and_preds(model, data)
        last_actual = actuals[-1]

        return JSONResponse(content={
            "ticker": ticker,
            "name": stock["name"],
            "n_days": n_days,
            "last_actual_price": last_actual,
            "future_predictions": predictions,
            "future_dates": future_dates,
            "val_actuals": actuals,
            "val_predictions": val_preds,
            "val_dates": val_dates
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)