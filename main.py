from __future__ import annotations
import os
import json
import random
import asyncio
import warnings
from datetime import timedelta
from typing import List
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
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

executor = ThreadPoolExecutor(max_workers=4)

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

def make_dataset(rev: np.ndarray, feats: np.ndarray):
    X, Y = [], []
    for i in range(LOOKBACK, len(rev) - HORIZON + 1):
        X.append(feats[i - LOOKBACK : i].flatten())
        Y.append(rev[i : i + HORIZON])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# ─── PIPELINE ────────────────────────────────────────────────────────────────
# Uses a two-horizon strategy:
#   Model A → predicts days 1-30  (short, high accuracy)
#   Model B → predicts days 1-90  (full horizon)
# Blended on test set per-horizon.
# HistGradientBoosting is chosen because:
#   - Native multi-output support (no wrapper → 1 model not 90)
#   - Handles missing values internally
#   - 5-10x faster than GBR
#   - Memory-efficient histogram binning

def run_pipeline(daily_df: pd.DataFrame, progress_cb) -> dict:
    feat_cols = [c for c in daily_df.columns if c != "revenue"]
    feat_arr  = daily_df[feat_cols].values.astype(np.float32)
    rev_arr   = daily_df["revenue"].values.astype(np.float32)

    progress_cb(5, "Scaling features")
    scaler  = RobustScaler()
    feat_sc = scaler.fit_transform(feat_arr).astype(np.float32)

    progress_cb(10, "Building training windows")
    X, Y = make_dataset(rev_arr, feat_sc)
    split    = max(2, int(len(X) * 0.8))
    X_tr, X_te = X[:split], X[split:]
    Y_tr, Y_te = Y[:split], Y[split:]

    # ── Shared model factory ────────────────────────────────────────────────
    def _fit(X, y, max_iter, max_depth, min_leaf, l2):
        m = HistGradientBoostingRegressor(
            max_iter=max_iter, max_depth=max_depth, learning_rate=0.1,
            min_samples_leaf=min_leaf, l2_regularization=l2,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=10, random_state=SEED,
        )
        m.fit(X, y)
        return m

    # ── Model A: 30-day — all days trained in parallel ──────────────────────
    progress_cb(15, "Training 30-day models (parallel)")
    models_30 = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_fit)(X_tr, Y_tr[:, d], 200, 5, 8, 0.1)
        for d in range(30)
    )
    progress_cb(48, "30-day models complete")

    # ── Model B: 90-day — all days trained in parallel ──────────────────────
    progress_cb(50, "Training 90-day models (parallel)")
    models_90 = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_fit)(X_tr, Y_tr[:, d], 150, 4, 10, 0.2)
        for d in range(HORIZON)
    )
    progress_cb(78, "90-day models complete")

    # ── Evaluate on test set ────────────────────────────────────────────────
    progress_cb(80, "Evaluating on held-out data")
    pred_30_te = np.column_stack([m.predict(X_te) for m in models_30])
    pred_90_te = np.column_stack([m.predict(X_te) for m in models_90])

    mae_30 = mean_absolute_error(Y_te[:, :30].flatten(), pred_30_te.flatten())
    mae_90 = mean_absolute_error(Y_te.flatten(),         pred_90_te.flatten())

    # ── Final forecast from last window ────────────────────────────────────
    progress_cb(85, "Generating forecast")
    last_w = feat_sc[-LOOKBACK:].flatten().reshape(1, -1)

    fut_30 = np.array([m.predict(last_w)[0] for m in models_30])
    fut_90 = np.array([m.predict(last_w)[0] for m in models_90])
    fut_90[:30] = 0.5 * fut_30 + 0.5 * fut_90[:30]   # blend short-horizon
    fut_90 = np.maximum(0.0, fut_90)

    last_date = daily_df.index.max()
    forecast  = [
        {"date": str((last_date + timedelta(days=i + 1)).date()),
         "predicted_sales": float(v)}
        for i, v in enumerate(fut_90)
    ]

    # ── Aggregate results ───────────────────────────────────────────────────
    progress_cb(92, "Aggregating results")
    p30   = forecast[:30]
    p90   = forecast[:90]
    rev30 = sum(p["predicted_sales"] for p in p30)
    rev90 = sum(p["predicted_sales"] for p in p90)

    peak_days = [
        p["date"]
        for p in sorted(p90, key=lambda x: x["predicted_sales"], reverse=True)[:3]
    ]
    hist_90    = float(daily_df["revenue"].iloc[-90:].sum())
    growth_yoy = round((rev90 - hist_90) / hist_90, 2) if hist_90 > 0 else 0.0

    progress_cb(95, "Fetching risk analysis")
    risk_factors = get_risk_factors({
        "projected_30_revenue":   rev30,
        "projected_90_revenue":   rev90,
        "peak_days_identified":   peak_days,
        "calculated_growth_rate": growth_yoy,
    })

    progress_cb(99, "Finalising")
    return {
        "next_30_days": {
            "expected_revenue": round(rev30, 2),
            "lower_bound":      round(max(0.0, rev30 - mae_30 * 30), 2),
            "upper_bound":      round(rev30 + mae_30 * 30, 2),
        },
        "next_90_days": {
            "expected_revenue": round(rev90, 2),
            "lower_bound":      round(max(0.0, rev90 - mae_90 * 90), 2),
            "upper_bound":      round(rev90 + mae_90 * 90, 2),
        },
        "peak_days":       peak_days,
        "risk_factors":    risk_factors,
        "growth_rate_yoy": growth_yoy,
    }

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

# ─── SSE HELPERS ─────────────────────────────────────────────────────────────

def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"

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
            422, f"Need at least {min_rows} days of history. Got {len(daily_df)}."
        )

    # get_running_loop() — correct for Python 3.10+, unlike get_event_loop()
    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    queue: asyncio.Queue            = asyncio.Queue()

    def progress_cb(pct: int, message: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, ("progress", pct, message))

    def run() -> None:
        try:
            result = run_pipeline(daily_df, progress_cb)
            loop.call_soon_threadsafe(queue.put_nowait, ("result", result))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

    async def event_stream():
        # Immediately emit a heartbeat so the browser knows the connection is alive
        yield sse({"type": "progress", "pct": 1, "message": "Connecting to forecast engine…"})

        loop.run_in_executor(executor, run)

        while True:
            item = await asyncio.wait_for(queue.get(), timeout=300)
            kind = item[0]

            if kind == "progress":
                yield sse({"type": "progress", "pct": item[1], "message": item[2]})

            elif kind == "result":
                yield sse({"type": "progress", "pct": 100, "message": "Complete"})
                yield sse({"type": "result",   "data": item[1]})
                break

            elif kind == "error":
                yield sse({"type": "error", "message": item[1]})
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":      "no-cache",
            "X-Accel-Buffering":  "no",
            "Connection":         "keep-alive",
        },
    )

@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)