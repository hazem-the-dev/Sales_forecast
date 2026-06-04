from __future__ import annotations
import os
import json
import warnings
from datetime import datetime, timedelta
from typing import List
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ML / DL Imports (Tuning libraries removed)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam 
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")

app = FastAPI(title="Sales Forecasting & Recommendation API (Lightweight)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# ─── PYDANTIC SCHEMAS FOR JSON VALIDATION ────────────────────────────────────

class SalesItem(BaseModel):
    order_date: str = Field(..., description="Date format: YYYY-MM-DD or ISO string")
    revenue: float = Field(..., description="Revenue amount generated")

class ForecastRequest(BaseModel):
    business_id: str = "My Store"
    force_refresh: bool = True
    sales_data: List[SalesItem]

# ─── CORE ML PIPELINE FUNCTIONS ──────────────────────────────────────────────

def create_features_dynamic(df, freq='D'):
    df = df.copy()
    df_resampled = df.resample(freq).sum().fillna(0)

    df_resampled['day_of_week'] = df_resampled.index.dayofweek
    df_resampled['day_of_month'] = df_resampled.index.day
    df_resampled['week_of_year'] = df_resampled.index.isocalendar().week
    df_resampled['month'] = df_resampled.index.month
    df_resampled['quarter'] = df_resampled.index.quarter
    df_resampled['is_weekend'] = (df_resampled['day_of_week'] >= 5).astype(int)

    df_resampled['month_sin'] = np.sin(2 * np.pi * df_resampled['month'] / 12)
    df_resampled['month_cos'] = np.cos(2 * np.pi * df_resampled['month'] / 12)
    df_resampled['dow_sin'] = np.sin(2 * np.pi * df_resampled['day_of_week'] / 7)
    df_resampled['dow_cos'] = np.cos(2 * np.pi * df_resampled['day_of_week'] / 7)

    lags, windows = [1, 7, 14, 30], [7, 14, 30]

    for lag in lags:
        df_resampled[f'lag_{lag}'] = df_resampled['revenue'].shift(lag)
        df_resampled[f'trend_{lag}'] = df_resampled['revenue'] - df_resampled[f'lag_{lag}']

    for w in windows:
        df_resampled[f'roll_mean_{w}'] = df_resampled['revenue'].rolling(w).mean()
        df_resampled[f'roll_std_{w}'] = df_resampled['revenue'].rolling(w).std()
        df_resampled[f'roll_min_{w}'] = df_resampled['revenue'].rolling(w).min()
        df_resampled[f'roll_max_{w}'] = df_resampled['revenue'].rolling(w).max()

    df_resampled['is_zero_sales'] = (df_resampled['revenue'] == 0).astype(int)
    df_resampled['pct_change'] = df_resampled['revenue'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)

    return df_resampled.dropna()

def prepare_data(raw_data):
    data = create_features_dynamic(raw_data, freq='D')
    if len(data) < 35: 
        return None

    target_col = "revenue"
    feature_cols = data.drop(columns=[target_col]).columns.tolist()

    split_idx = int(len(data) * 0.7)
    train, test = data.iloc[:split_idx], data.iloc[split_idx:]

    return train[feature_cols], test[feature_cols], train[target_col], test[target_col], test, feature_cols

def scale_data(X_train, X_test, y_train):
    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    return scaler_x, scaler_y, X_train_scaled, X_test_scaled, y_train_scaled, X_train_lstm, X_test_lstm

def optimize_ensemble(lstm_pred, gbr_pred, y_test):
    best_weight, best_mae = 0.5, float('inf')
    lstm_arr, gbr_arr, y_test_arr = np.array(lstm_pred), np.array(gbr_pred), np.array(y_test)

    for w in np.linspace(0, 1, 21):
        mae = mean_absolute_error(y_test_arr, w * lstm_arr + (1 - w) * gbr_arr)
        if mae < best_mae:
            best_mae, best_weight = mae, w

    return best_weight, best_mae

# ─── FUTURE FORECAST ENGINE ──────────────────────────────────────────────────

def generate_future_forecast(raw_sales, feature_cols, scaler_x, scaler_y, lstm_model, gbr_model, ensemble_weight, days_to_forecast=90):
    current_data = raw_sales.copy()
    future_predictions = []
    last_date = current_data.index.max()
    
    for i in range(1, days_to_forecast + 1):
        next_date = last_date + timedelta(days=i)
        
        temp_df = create_features_dynamic(current_data, freq='D')
        if temp_df.empty: break
            
        last_row = temp_df.iloc[[-1]][feature_cols]
        last_row_scaled = scaler_x.transform(last_row)
        last_row_lstm = last_row_scaled.reshape((last_row_scaled.shape[0], 1, last_row_scaled.shape[1]))
        
        lstm_p_scaled = lstm_model.predict(last_row_lstm, verbose=0)
        lstm_p = scaler_y.inverse_transform(lstm_p_scaled).reshape(-1)[0]
        gbr_p = gbr_model.predict(last_row_scaled)[0]
        
        combined_pred = max(0.0, (ensemble_weight * lstm_p) + ((1 - ensemble_weight) * gbr_p))
        future_predictions.append({"date": str(next_date.date()), "predicted_sales": combined_pred})
        
        current_data.loc[next_date] = [combined_pred]
        
    return future_predictions

# ─── LLM CLIENT FUNCTION ─────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
PROMPT_RISK_FACTORS = """You are a senior revenue analyst looking at business operational sales trends. Based on the JSON payload regarding future trajectory shifts, return a list of contextual risk factors (such as operational, competitors or external). 
Return ONLY a valid JSON array of strings. No conversational text outside the JSON block.
Output format:
["risk_factor_slug_1", "risk_factor_slug_2"]"""

def _call_llm_for_risks(context: dict) -> list[str]:
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "system", "content": PROMPT_RISK_FACTORS}, {"role": "user", "content": json.dumps(context)}], "temperature": 0.2},
            timeout=15,
        )
        if response.status_code != 200:
            return ["general_market_volatility"]
        
        raw = response.json()["choices"][0]["message"]["content"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
            raw = raw.strip()
        return json.loads(raw)
    except Exception:
        return ["macroeconomic_uncertainty"]

# ─── API ENDPOINT ROUTE ──────────────────────────────────────────────────────

@app.post("/api/v1/forecast-recommendations")
async def generate_recommendations(payload: ForecastRequest):
    incoming_data = [item.dict() for item in payload.sales_data]
    df = pd.DataFrame(incoming_data)
    
    if df.empty or 'order_date' not in df.columns or 'revenue' not in df.columns:
        raise HTTPException(status_code=400, detail="Missing required layout fields inside 'sales_data'")

    df['order_date'] = pd.to_datetime(df['order_date'])
    df = df.sort_values('order_date').reset_index(drop=True)
    raw_sales = df.groupby('order_date')['revenue'].sum().to_frame()

    prep_res = prepare_data(raw_sales)
    if prep_res is None:
        raise HTTPException(status_code=422, detail="Insufficient data rows to process historical validation and 30-day lookbacks.")

    X_train, X_test, y_train, y_test, test_df, feature_cols = prep_res
    scaler_x, scaler_y, X_train_sc, X_test_sc, y_train_sc, X_train_lstm, X_test_lstm = scale_data(X_train, X_test, y_train)
    
    # --- LIGHTWEIGHT LSTM (Static Architecture) ---
    best_lstm_model = Sequential([
        LSTM(32, activation='tanh', return_sequences=False, input_shape=(1, X_train_lstm.shape[2])),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    best_lstm_model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    best_lstm_model.fit(X_train_lstm, y_train_sc, epochs=10, batch_size=16, verbose=0)
    
    lstm_pred_scaled = best_lstm_model.predict(X_test_lstm, verbose=0)
    lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled).reshape(-1)
    
    # --- LIGHTWEIGHT GBR (No CV Search) ---
    best_gbr_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    best_gbr_model.fit(X_train_sc, y_train)
    gbr_pred = best_gbr_model.predict(X_test_sc)

    # Hybrid weight optimization
    best_weight, best_mae = optimize_ensemble(lstm_pred, gbr_pred, y_test)

    # Generate Forecasts
    future_forecast = generate_future_forecast(
        raw_sales=raw_sales, feature_cols=feature_cols, scaler_x=scaler_x, 
        scaler_y=scaler_y, lstm_model=best_lstm_model, gbr_model=best_gbr_model, 
        ensemble_weight=best_weight, days_to_forecast=90
    )

    pred_30_days = future_forecast[:30]
    pred_90_days = future_forecast[:90]
    
    revenue_30 = sum(p["predicted_sales"] for p in pred_30_days)
    revenue_90 = sum(p["predicted_sales"] for p in pred_90_days)
    
    buffer_30 = best_mae * 30
    buffer_90 = best_mae * 90

    sorted_peaks = sorted(pred_90_days, key=lambda x: x["predicted_sales"], reverse=True)
    peak_days = [p["date"] for p in sorted_peaks[:3]]

    historical_sum = float(raw_sales['revenue'].iloc[-90:].sum())
    growth_rate_yoy = round((revenue_90 - historical_sum) / historical_sum, 2) if historical_sum > 0 else 0.0

    llm_context = {
        "projected_30_revenue": revenue_30,
        "projected_90_revenue": revenue_90,
        "peak_days_identified": peak_days,
        "calculated_growth_rate": growth_rate_yoy
    }
    risk_factors = _call_llm_for_risks(llm_context)

    return {
        "next_30_days": {
            "expected_revenue": round(revenue_30, 2),
            "lower_bound": round(max(0, revenue_30 - buffer_30), 2),
            "upper_bound": round(revenue_30 + buffer_30, 2)
        },
        "next_90_days": {
            "expected_revenue": round(revenue_90, 2),
            "lower_bound": round(max(0, revenue_90 - buffer_90), 2),
            "upper_bound": round(revenue_90 + buffer_90, 2)
        },
        "peak_days": peak_days,
        "risk_factors": risk_factors,
        "growth_rate_yoy": growth_rate_yoy
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)