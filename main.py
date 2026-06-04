from __future__ import annotations
import os
import json
import random
import warnings
from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler
from sklearn.multioutput import MultiOutputRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")

# ─── GLOBAL DETERMINISM ───────────────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Hyperparameters
LOOKBACK      = 30   # days of history fed into LSTM as a sequence
HORIZON_SHORT = 30   # days ahead for short forecast
HORIZON_LONG  = 90   # days ahead for long forecast

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

app = FastAPI(title="Sales Forecasting API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://fuse-eg.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── SCHEMA ───────────────────────────────────────────────────────────────────

class SalesItem(BaseModel):
    order_date: str = Field(..., description="YYYY-MM-DD")
    revenue: float

# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────

def build_daily_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Resample to daily, add calendar + lag + rolling features."""
    df = raw.copy().resample("D").sum().fillna(0)

    df["dow"]       = df.index.dayofweek
    df["dom"]       = df.index.day
    df["month"]     = df.index.month
    df["quarter"]   = df.index.quarter
    df["is_weekend"]= (df["dow"] >= 5).astype(int)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"] / 7)

    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"]   = df["revenue"].shift(lag)
        df[f"trend_{lag}"] = df["revenue"] - df[f"lag_{lag}"]

    for w in [7, 14, 30]:
        df[f"rmean_{w}"] = df["revenue"].rolling(w).mean()
        df[f"rstd_{w}"]  = df["revenue"].rolling(w).std().fillna(0)
        df[f"rmin_{w}"]  = df["revenue"].rolling(w).min()
        df[f"rmax_{w}"]  = df["revenue"].rolling(w).max()

    df["pct_change"]    = df["revenue"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    df["is_zero"]       = (df["revenue"] == 0).astype(int)

    return df.dropna()


# ─── DATASET BUILDER — DIRECT MULTI-STEP ──────────────────────────────────────
# Instead of recursive 1-step prediction (error compounds), we build
# (X=lookback_window → y=next_horizon_days) samples.  Both LSTM and GBR
# are trained this way so neither compounds errors over 90 steps.

def make_multistep_dataset(series: np.ndarray, feature_matrix: np.ndarray,
                           lookback: int, horizon: int):
    """
    Returns:
        X_seq  : (N, lookback, n_features)  — for LSTM
        X_flat : (N, lookback * n_features) — for GBR
        Y      : (N, horizon)               — targets
    """
    X_seq, X_flat, Y = [], [], []
    for i in range(lookback, len(series) - horizon + 1):
        window = feature_matrix[i - lookback : i]   # (lookback, F)
        target = series[i : i + horizon]             # (horizon,)
        X_seq.append(window)
        X_flat.append(window.flatten())
        Y.append(target)
    return np.array(X_seq), np.array(X_flat), np.array(Y)


# ─── MODEL BUILDERS ───────────────────────────────────────────────────────────

def build_lstm(lookback: int, n_features: int, horizon: int) -> Sequential:
    tf.random.set_seed(SEED)
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(lookback, n_features)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        Dense(64, activation="relu"),
        Dense(horizon),          # predict all horizon steps at once
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="huber")
    return model


def build_gbr(horizon: int) -> MultiOutputRegressor:
    """
    MultiOutputRegressor wraps GBR to predict all horizon days at once,
    one tree per day.  GBR with robust params is extremely stable.
    """
    base = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        loss="huber",
        random_state=SEED,
    )
    return MultiOutputRegressor(base, n_jobs=-1)


# ─── TRAIN + PREDICT ──────────────────────────────────────────────────────────

def train_and_forecast(daily_df: pd.DataFrame, horizon: int):
    feature_cols = [c for c in daily_df.columns if c != "revenue"]
    feat_arr     = daily_df[feature_cols].values.astype(np.float32)
    rev_arr      = daily_df["revenue"].values.astype(np.float32)

    # Scale features with RobustScaler (handles outliers in revenue data)
    feat_scaler = RobustScaler()
    feat_scaled = feat_scaler.fit_transform(feat_arr)

    # Scale target independently
    rev_scaler  = RobustScaler()
    rev_scaled  = rev_scaler.fit_transform(rev_arr.reshape(-1, 1)).reshape(-1)

    X_seq, X_flat, Y_raw = make_multistep_dataset(
        rev_arr, feat_scaled, LOOKBACK, horizon
    )
    # Y in original scale for GBR (tree-based, no need to scale target)
    # Y scaled for LSTM
    Y_scaled = rev_scaler.transform(Y_raw.reshape(-1, 1)).reshape(Y_raw.shape)

    if len(X_seq) < 10:
        raise ValueError("Not enough data for multi-step training.")

    split = max(1, int(len(X_seq) * 0.8))
    Xseq_tr,  Xseq_te  = X_seq[:split],  X_seq[split:]
    Xflat_tr, Xflat_te = X_flat[:split], X_flat[split:]
    Ysc_tr,   Ysc_te   = Y_scaled[:split], Y_scaled[split:]
    Yraw_tr,  Yraw_te  = Y_raw[:split],  Y_raw[split:]

    # ── LSTM ──
    lstm = build_lstm(LOOKBACK, feat_scaled.shape[1], horizon)
    lstm.fit(
        Xseq_tr, Ysc_tr,
        validation_split=0.15,
        epochs=150,
        batch_size=32,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=15,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=7, min_lr=1e-5),
        ],
        verbose=0,
    )
    lstm_pred_te = rev_scaler.inverse_transform(
        lstm.predict(Xseq_te, verbose=0)
    )  # (n_test, horizon)

    # ── GBR ──
    gbr = build_gbr(horizon)
    gbr.fit(Xflat_tr, Yraw_tr)
    gbr_pred_te = gbr.predict(Xflat_te)   # (n_test, horizon)

    # ── Optimal blend on test set ──
    best_w, best_mae = _optimise_blend(lstm_pred_te, gbr_pred_te, Yraw_te)

    # ── Generate actual future forecast using the LAST window ──
    last_seq  = feat_scaled[-LOOKBACK:].reshape(1, LOOKBACK, feat_scaled.shape[1])
    last_flat = feat_scaled[-LOOKBACK:].flatten().reshape(1, -1)

    lstm_future = rev_scaler.inverse_transform(
        lstm.predict(last_seq, verbose=0)
    ).reshape(-1)
    gbr_future  = gbr.predict(last_flat).reshape(-1)

    future_vals = np.maximum(
        0.0, best_w * lstm_future + (1 - best_w) * gbr_future
    )

    # Build dated predictions
    last_date = daily_df.index.max()
    forecast  = [
        {"date": str((last_date + timedelta(days=i + 1)).date()),
         "predicted_sales": float(v)}
        for i, v in enumerate(future_vals)
    ]

    return forecast, best_mae


def _optimise_blend(lstm_p: np.ndarray, gbr_p: np.ndarray,
                    y_true: np.ndarray) -> tuple[float, float]:
    """Grid-search blend weight on flattened test predictions."""
    lstm_f = lstm_p.flatten()
    gbr_f  = gbr_p.flatten()
    y_f    = y_true.flatten()
    best_w, best_mae = 0.5, float("inf")
    for w in np.linspace(0, 1, 41):
        mae = mean_absolute_error(y_f, w * lstm_f + (1 - w) * gbr_f)
        if mae < best_mae:
            best_mae, best_w = mae, w
    return best_w, best_mae


# ─── LLM RISK FACTORS ────────────────────────────────────────────────────────

_RISK_PROMPT = (
    "You are a senior revenue analyst. Given the JSON payload about future sales "
    "trajectory, return ONLY a valid JSON array of contextual risk factor slugs "
    "(operational, competitor, or external). No prose, no markdown fences.\n"
    'Example: ["supply_chain_disruption", "competitor_price_war"]'
)

def get_risk_factors(context: dict) -> list[str]:
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                     "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": _RISK_PROMPT},
                    {"role": "user",   "content": json.dumps(context)},
                ],
                "temperature": 0.2,
            },
            timeout=15,
        )
        if r.status_code != 200:
            return ["general_market_volatility"]
        raw = r.json()["choices"][0]["message"]["content"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return ["macroeconomic_uncertainty"]


# ─── ENDPOINT ─────────────────────────────────────────────────────────────────

@app.post("/api/v1/forecast-recommendations")
async def generate_recommendations(payload: List[SalesItem]):
    df = pd.DataFrame([i.dict() for i in payload])
    if df.empty or "order_date" not in df.columns or "revenue" not in df.columns:
        raise HTTPException(400, "Missing required fields: order_date, revenue")

    df["order_date"] = pd.to_datetime(df["order_date"])
    df = df.sort_values("order_date").reset_index(drop=True)
    raw_sales = df.groupby("order_date")["revenue"].sum().to_frame()

    daily_df = build_daily_features(raw_sales)

    # Minimum data: lookback + horizon + enough training samples
    min_rows = LOOKBACK + HORIZON_LONG + 10
    if len(daily_df) < min_rows:
        raise HTTPException(
            422,
            f"Insufficient data: need at least {min_rows} days of history "
            f"(after feature warm-up). Got {len(daily_df)}."
        )

    try:
        forecast_90, best_mae = train_and_forecast(daily_df, HORIZON_LONG)
    except ValueError as e:
        raise HTTPException(422, str(e))

    p30 = forecast_90[:HORIZON_SHORT]
    p90 = forecast_90[:HORIZON_LONG]

    rev30 = sum(p["predicted_sales"] for p in p30)
    rev90 = sum(p["predicted_sales"] for p in p90)

    # MAE-based confidence bounds
    buf30 = best_mae * HORIZON_SHORT
    buf90 = best_mae * HORIZON_LONG

    peak_days = [
        p["date"]
        for p in sorted(p90, key=lambda x: x["predicted_sales"], reverse=True)[:3]
    ]

    hist_90    = float(raw_sales["revenue"].iloc[-HORIZON_LONG:].sum())
    growth_yoy = round((rev90 - hist_90) / hist_90, 2) if hist_90 > 0 else 0.0

    risk_factors = get_risk_factors({
        "projected_30_revenue":   rev30,
        "projected_90_revenue":   rev90,
        "peak_days_identified":   peak_days,
        "calculated_growth_rate": growth_yoy,
    })

    return {
        "next_30_days": {
            "expected_revenue": round(rev30, 2),
            "lower_bound":      round(max(0, rev30 - buf30), 2),
            "upper_bound":      round(rev30 + buf30, 2),
        },
        "next_90_days": {
            "expected_revenue": round(rev90, 2),
            "lower_bound":      round(max(0, rev90 - buf90), 2),
            "upper_bound":      round(rev90 + buf90, 2),
        },
        "peak_days":       peak_days,
        "risk_factors":    risk_factors,
        "growth_rate_yoy": growth_yoy,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)