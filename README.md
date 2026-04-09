

# 📈 Stock Price Prediction with LSTM & MLflow

An end-to-end machine learning pipeline for predicting stock prices using **LSTM** neural networks. Features automated hyperparameter optimization with **Optuna**, experiment tracking with **MLflow**, and an interactive **FastAPI** web dashboard for real-time visualization and multi-day forecasting.

---

## 📂 Project Structure

```text
stockproject/
├── Experiments/
│   ├── Architecture.py
│   ├── helpers.py
│   ├── mlflowmain.py
│   ├── mlruns/
│   └── __init__.py
├── Deployment/
│   ├── app.py
│   ├── model_loader.py
│   ├── predictor.py
│   ├── templates/
│   │   └── index.html
│   └── static/
├── Data Exploration/
│   ├── stock_visualization.ipynb
│   └── multi_stock_lstm.ipynb
├── main.py
├── requirements.txt
└── README.md
```


```

---

## 🚀 Getting Started

### 1. Clone the Repository

```powershell
git clone https://github.com/YOUR_USERNAME/stockproject.git
cd stockproject
```

### 2. Create Virtual Environment (Python 3.11 required)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 4. Verify Installation

```powershell
python -c "import torch; print(torch.__version__)"
python -c "import mlflow; print(mlflow.__version__)"
python -c "import optuna; print(optuna.__version__)"
```

---

## 🧪 Training & Experiments

### How It Works

1. Stock data is downloaded **once** per ticker using `yfinance`
2. Features are engineered: RSI, moving averages, volatility, returns
3. Data is scaled and windowed with a **60-day lookback**
4. Optuna searches over `hidden_size`, `num_layers`, `learning_rate`, `batch_size`
5. Each trial is logged as a **nested MLflow run** with metrics, params, loss curves, and models
6. After all trials, the **best model is retrained and saved** in the parent MLflow run

### Supported Stocks

| Ticker | Name           |
| ------ | -------------- |
| META   | Meta Platforms |
| MSFT   | Microsoft      |
| GC=F   | Gold Futures   |
| HPQ    | HP Inc.        |
| NVDA   | NVIDIA         |

### Run Experiments

```powershell
cd Experiments
python mlflowmain.py
```

### View Results in MLflow UI

Open a separate terminal:

```powershell
cd Experiments
mlflow ui --backend-store-uri ./mlruns
```

Then open: **http://127.0.0.1:5000**

---

## 🏗️ Model Architecture

```
Input (9 features × 60 timesteps)
    ↓
LSTM Layer(s) — tuned by Optuna
    ↓
Fully Connected Layer → 1 output (predicted closing price)
```

### Input Features

| Feature     | Description                    |
| ----------- | ------------------------------ |
| Close       | Closing price                  |
| Day_of_Week | Day of week (0–4)             |
| Month       | Month (1–12)                  |
| Volume      | Trading volume                 |
| return      | Daily return                   |
| ma20        | 20-day moving average          |
| ma50        | 50-day moving average          |
| volatility  | 20-day rolling std deviation   |
| rsi         | 14-day Relative Strength Index |

### Hyperparameters Tuned

| Parameter     | Range              |
| ------------- | ------------------ |
| hidden_size   | 16 – 128          |
| num_layers    | 1 – 3             |
| learning_rate | 1e-4 – 1e-2 (log) |
| batch_size    | 16, 32, 64         |

---

## 💻 Deployment & Web App

### Start the Server

```powershell
cd Deployment
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open: **http://127.0.0.1:8000**

### Features

- **Stock Selection** — Dropdown to pick any trained stock
- **Visualize** — Plot validation actual vs predicted with real dates
- **Predict Ahead** — Forecast 1 to 40 days ahead
- **Slider** — Quick adjust forecast days
- **Stats Panel** — Last price, forecast days, predicted price
- **Warnings** — Beyond 30 days shows reliability warning, max 40 days
- **Clear** — Reset the chart

### API Endpoints

| Endpoint                          | Description                        |
| --------------------------------- | ---------------------------------- |
| `GET /`                         | Dashboard page                     |
| `GET /api/stocks`               | List available stocks              |
| `GET /api/visualize/{ticker}`   | Validation actuals and predictions |
| `GET /api/predict/{ticker}/{n}` | N-day future forecast              |

---

## 📊 Data Exploration

```powershell
cd "Data Exploration"
jupyter notebook
```

Contains notebooks for data visualization, feature engineering, and preliminary LSTM testing.

---

## 📦 Key Dependencies

| Package      | Version |
| ------------ | ------- |
| Python       | 3.11    |
| PyTorch      | 2.1.0   |
| MLflow       | 2.9.2   |
| Optuna       | 3.6.1   |
| NumPy        | 1.26.4  |
| Pandas       | 2.0.3   |
| scikit-learn | 1.3.2   |
| yfinance     | 0.2.36+ |
| FastAPI      | latest  |
| uvicorn      | latest  |
| setuptools   | <81     |

---

## ⚠️ Notes

- **Python 3.11 is required** — PyTorch 2.1.0 does not support 3.12+
- **setuptools must be < 81** — MLflow 2.9.2 needs `pkg_resources`
- **Some tickers may fail** to download due to Yahoo Finance availability
- **Predictions beyond 30 days** become less reliable due to error accumulation
- If PyTorch gives DLL errors, install [VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

```

```
