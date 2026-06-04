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
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

warnings.filterwarnings("ignore")

# ─── DETERMINISM — fixes the inconsistency problem ───────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

app = FastAPI(title="Sales Forecasting & Recommendation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://fuse-eg.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── SCHEMAS ─────────────────────────────────────────────────────────────────

class SalesItem(BaseModel):
    order_date: str = Field(..., description="Date format: YYYY-MM-DD or ISO string")
    revenue: float = Field(..., description="Revenue amount generated")

# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().resample("D").sum().fillna(0)

    df["day_of_week"]  = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["month"]        = df.index.month
    df["quarter"]      = df.index.quarter
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"]   = df["revenue"].shift(lag)
        df[f"trend_{lag}"] = df["revenue"] - df[f"lag_{lag}"]

    for w in [7, 14, 30]:
        df[f"roll_mean_{w}"] = df["revenue"].rolling(w).mean()
        df[f"roll_std_{w}"]  = df["revenue"].rolling(w).std()
        df[f"roll_min_{w}"]  = df["revenue"].rolling(w).min()
        df[f"roll_max_{w}"]  = df["revenue"].rolling(w).max()

    df["is_zero_sales"] = (df["revenue"] == 0).astype(int)
    df["pct_change"]    = df["revenue"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)

    return df.dropna()

# ─── DATA PREPARATION ────────────────────────────────────────────────────────

def prepare_data(raw: pd.DataFrame):
    data = create_features(raw)
    if len(data) < 35:
        return None

    feature_cols = [c for c in data.columns if c != "revenue"]
    split = int(len(data) * 0.7)
    train, test = data.iloc[:split], data.iloc[split:]

    return (
        train[feature_cols], test[feature_cols],
        train["revenue"],    test["revenue"],
        test, feature_cols,
    )

def scale_data(X_train, X_test, y_train):
    sx, sy = MinMaxScaler(), MinMaxScaler()
    Xtr = sx.fit_transform(X_train)
    Xte = sx.transform(X_test)
    ytr = sy.fit_transform(y_train.values.reshape(-1, 1))

    # LSTM expects (samples, timesteps=1, features)
    return (
        sx, sy,
        Xtr, Xte, ytr,
        Xtr.reshape(Xtr.shape[0], 1, Xtr.shape[1]),
        Xte.reshape(Xte.shape[0], 1, Xte.shape[1]),
    )

# ─── MODEL TRAINING ──────────────────────────────────────────────────────────

def build_lstm(input_dim: int) -> Sequential:
    """Fixed, proven architecture — no random search needed."""
    tf.random.set_seed(SEED)
    model = Sequential([
        LSTM(64, activation="tanh", return_sequences=False,
             input_shape=(1, input_dim)),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=5e-3), loss="mse")
    return model


def train_lstm(X_tr_lstm, y_tr_sc, X_te_lstm, sy) -> np.ndarray:
    model = build_lstm(X_tr_lstm.shape[2])
    model.fit(
        X_tr_lstm, y_tr_sc,
        validation_split=0.15,
        epochs=60,                          # enough to converge
        batch_size=16,
        callbacks=[EarlyStopping(monitor="val_loss", patience=8,
                                 restore_best_weights=True)],
        verbose=0,
    )
    return sy.inverse_transform(model.predict(X_te_lstm, verbose=0)).reshape(-1), model


def train_gbr(X_tr, y_tr, X_te) -> np.ndarray:
    """Fixed hyper-params that are consistently good for daily revenue data."""
    gbr = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=SEED,
    )
    gbr.fit(X_tr, y_tr)
    return gbr.predict(X_te), gbr

# ─── ENSEMBLE WEIGHT OPTIMISATION ────────────────────────────────────────────

def optimise_ensemble(lstm_p, gbr_p, y_test):
    best_w, best_mae = 0.5, float("inf")
    for w in np.linspace(0, 1, 41):          # 41 steps = 0.025 resolution
        mae = mean_absolute_error(y_test, w * lstm_p + (1 - w) * gbr_p)
        if mae < best_mae:
            best_mae, best_w = mae, w
    return best_w, best_mae

# ─── RECURSIVE FUTURE FORECAST ───────────────────────────────────────────────

def future_forecast(raw_sales, feature_cols, sx, sy, lstm_model, gbr_model,
                    weight, days=90) -> list[dict]:
    current = raw_sales.copy()
    preds   = []
    last    = current.index.max()

    for i in range(1, days + 1):
        next_dt = last + timedelta(days=i)
        tmp     = create_features(current)
        if tmp.empty:
            break

        row        = tmp.iloc[[-1]][feature_cols]
        row_sc     = sx.transform(row)
        row_lstm   = row_sc.reshape(1, 1, row_sc.shape[1])

        lstm_p = sy.inverse_transform(
            lstm_model.predict(row_lstm, verbose=0)
        ).reshape(-1)[0]
        gbr_p  = gbr_model.predict(row_sc)[0]

        val = max(0.0, weight * lstm_p + (1 - weight) * gbr_p)
        preds.append({"date": str(next_dt.date()), "predicted_sales": val})
        current.loc[next_dt] = [val]

    return preds

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
        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return ["macroeconomic_uncertainty"]

# ─── ENDPOINT ────────────────────────────────────────────────────────────────

@app.post("/api/v1/forecast-recommendations")
async def generate_recommendations(payload: List[SalesItem]):
    df = pd.DataFrame([i.dict() for i in payload])
    if df.empty or "order_date" not in df.columns or "revenue" not in df.columns:
        raise HTTPException(400, "Missing required fields: order_date, revenue")

    df["order_date"] = pd.to_datetime(df["order_date"])
    df = df.sort_values("order_date").reset_index(drop=True)
    raw_sales = df.groupby("order_date")["revenue"].sum().to_frame()

    prep = prepare_data(raw_sales)
    if prep is None:
        raise HTTPException(
            422,
            "Insufficient data: need at least 35 days after rolling-window warm-up.",
        )

    X_tr, X_te, y_tr, y_te, _, feature_cols = prep
    sx, sy, Xtr_sc, Xte_sc, ytr_sc, Xtr_lstm, Xte_lstm = scale_data(X_tr, X_te, y_tr)

    lstm_pred, lstm_model = train_lstm(Xtr_lstm, ytr_sc, Xte_lstm, sy)
    gbr_pred,  gbr_model  = train_gbr(Xtr_sc, y_tr, Xte_sc)

    best_w, best_mae = optimise_ensemble(lstm_pred, gbr_pred, y_te)

    forecast = future_forecast(
        raw_sales, feature_cols, sx, sy,
        lstm_model, gbr_model, best_w, days=90,
    )

    p30, p90 = forecast[:30], forecast[:90]
    rev30 = sum(p["predicted_sales"] for p in p30)
    rev90 = sum(p["predicted_sales"] for p in p90)

    buf30, buf90 = best_mae * 30, best_mae * 90

    peak_days = [
        p["date"]
        for p in sorted(p90, key=lambda x: x["predicted_sales"], reverse=True)[:3]
    ]

    hist_90    = float(raw_sales["revenue"].iloc[-90:].sum())
    growth_yoy = round((rev90 - hist_90) / hist_90, 2) if hist_90 > 0 else 0.0

    risk_factors = get_risk_factors({
        "projected_30_revenue":  rev30,
        "projected_90_revenue":  rev90,
        "peak_days_identified":  peak_days,
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
        "peak_days":      peak_days,
        "risk_factors":   risk_factors,
        "growth_rate_yoy": growth_yoy,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)