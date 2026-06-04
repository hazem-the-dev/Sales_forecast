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

from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# ─── DETERMINISM ──────────────────────────────────────────────────────────────
SEED     = 42
LOOKBACK = 30
HORIZON  = 90

random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

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

def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy().resample("D").sum().fillna(0)

    df["dow"]        = df.index.dayofweek
    df["dom"]        = df.index.day
    df["month"]      = df.index.month
    df["quarter"]    = df.index.quarter
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]    = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["dow"] / 7)

    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"]   = df["revenue"].shift(lag)
        df[f"trend_{lag}"] = df["revenue"] - df[f"lag_{lag}"]

    for w in [7, 14, 30]:
        df[f"rmean_{w}"] = df["revenue"].rolling(w).mean()
        df[f"rstd_{w}"]  = df["revenue"].rolling(w).std().fillna(0)
        df[f"rmin_{w}"]  = df["revenue"].rolling(w).min()
        df[f"rmax_{w}"]  = df["revenue"].rolling(w).max()

    df["pct_change"] = df["revenue"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    df["is_zero"]    = (df["revenue"] == 0).astype(int)

    return df.dropna()

# ─── MULTI-STEP DATASET ───────────────────────────────────────────────────────
# Each sample: X = flattened lookback window, Y = next `horizon` revenue values.
# No recursive prediction — both models see the full future horizon at once.

def make_dataset(rev: np.ndarray, feats: np.ndarray, lookback: int, horizon: int):
    X, Y = [], []
    for i in range(lookback, len(rev) - horizon + 1):
        X.append(feats[i - lookback : i].flatten())
        Y.append(rev[i : i + horizon])
    return np.array(X), np.array(Y)

# ─── TRAIN & FORECAST ─────────────────────────────────────────────────────────

def train_and_forecast(daily_df: pd.DataFrame):
    feat_cols = [c for c in daily_df.columns if c != "revenue"]
    feat_arr  = daily_df[feat_cols].values.astype(np.float64)
    rev_arr   = daily_df["revenue"].values.astype(np.float64)

    scaler   = RobustScaler()
    feat_sc  = scaler.fit_transform(feat_arr)

    X, Y = make_dataset(rev_arr, feat_sc, LOOKBACK, HORIZON)

    split     = max(1, int(len(X) * 0.8))
    X_tr, X_te = X[:split], X[split:]
    Y_tr, Y_te = Y[:split], Y[split:]

    # Model 1 — GBR: accurate, stable, deterministic
    gbr = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5,
            loss="huber", random_state=SEED,
        ),
        n_jobs=-1,
    )
    gbr.fit(X_tr, Y_tr)

    # Model 2 — HistGBR: faster than GBR, handles larger datasets well
    hgbr = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_iter=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=5, random_state=SEED,
        ),
        n_jobs=-1,
    )
    hgbr.fit(X_tr, Y_tr)

    # Blend weight optimised on test set
    gbr_te  = gbr.predict(X_te)
    hgbr_te = hgbr.predict(X_te)
    best_w, best_mae = _optimise_blend(gbr_te, hgbr_te, Y_te)

    # Final forecast from the last lookback window
    last_window = feat_sc[-LOOKBACK:].flatten().reshape(1, -1)
    gbr_fut  = gbr.predict(last_window).reshape(-1)
    hgbr_fut = hgbr.predict(last_window).reshape(-1)

    future_vals = np.maximum(
        0.0, best_w * gbr_fut + (1 - best_w) * hgbr_fut
    )

    last_date = daily_df.index.max()
    forecast  = [
        {"date": str((last_date + timedelta(days=i + 1)).date()),
         "predicted_sales": float(v)}
        for i, v in enumerate(future_vals)
    ]

    return forecast, best_mae


def _optimise_blend(p1: np.ndarray, p2: np.ndarray, y: np.ndarray):
    p1f, p2f, yf = p1.flatten(), p2.flatten(), y.flatten()
    best_w, best_mae = 0.5, float("inf")
    for w in np.linspace(0, 1, 41):
        mae = mean_absolute_error(yf, w * p1f + (1 - w) * p2f)
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

    daily_df = build_features(raw_sales)

    min_rows = LOOKBACK + HORIZON + 10
    if len(daily_df) < min_rows:
        raise HTTPException(
            422,
            f"Need at least {min_rows} days of history after warm-up. Got {len(daily_df)}."
        )

    forecast_90, best_mae = train_and_forecast(daily_df)

    p30 = forecast_90[:30]
    p90 = forecast_90[:90]
    rev30 = sum(p["predicted_sales"] for p in p30)
    rev90 = sum(p["predicted_sales"] for p in p90)

    peak_days = [
        p["date"]
        for p in sorted(p90, key=lambda x: x["predicted_sales"], reverse=True)[:3]
    ]

    hist_90    = float(raw_sales["revenue"].iloc[-90:].sum())
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
            "lower_bound":      round(max(0, rev30 - best_mae * 30), 2),
            "upper_bound":      round(rev30 + best_mae * 30, 2),
        },
        "next_90_days": {
            "expected_revenue": round(rev90, 2),
            "lower_bound":      round(max(0, rev90 - best_mae * 90), 2),
            "upper_bound":      round(rev90 + best_mae * 90, 2),
        },
        "peak_days":       peak_days,
        "risk_factors":    risk_factors,
        "growth_rate_yoy": growth_yoy,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)