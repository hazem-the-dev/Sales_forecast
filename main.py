from __future__ import annotations
import os
import json
import random
import asyncio
import warnings
from datetime import timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
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

from sqlalchemy import create_engine, text
from openai import OpenAI
import logging
import traceback


logging.basicConfig(level=logging.DEBUG)
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
DATABASE_URL  = os.environ.get("DATABASE_URL", "")

app = FastAPI(title="Sales Forecasting & FUSE Chat API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://fuse-eg.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=4)

# ==============================================================================
# 1. ORIGINAL FORECASTING PIPELINE (unchanged)
# ==============================================================================

class SalesItem(BaseModel):
    order_date: str = Field(..., description="YYYY-MM-DD")
    revenue: float

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

def make_dataset(rev: np.ndarray, feats: np.ndarray):
    X, Y = [], []
    for i in range(LOOKBACK, len(rev) - HORIZON + 1):
        X.append(feats[i - LOOKBACK : i].flatten())
        Y.append(rev[i : i + HORIZON])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

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
    def _fit(X, y, max_iter, max_depth, min_leaf, l2):
        m = HistGradientBoostingRegressor(
            max_iter=max_iter, max_depth=max_depth, learning_rate=0.1,
            min_samples_leaf=min_leaf, l2_regularization=l2,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=10, random_state=SEED,
        )
        m.fit(X, y)
        return m
    progress_cb(15, "Training 30-day models (parallel)")
    models_30 = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_fit)(X_tr, Y_tr[:, d], 200, 5, 8, 0.1) for d in range(30)
    )
    progress_cb(48, "30-day models complete")
    progress_cb(50, "Training 90-day models (parallel)")
    models_90 = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_fit)(X_tr, Y_tr[:, d], 150, 4, 10, 0.2) for d in range(HORIZON)
    )
    progress_cb(78, "90-day models complete")
    progress_cb(80, "Evaluating on held-out data")
    pred_30_te = np.column_stack([m.predict(X_te) for m in models_30])
    pred_90_te = np.column_stack([m.predict(X_te) for m in models_90])
    mae_30 = mean_absolute_error(Y_te[:, :30].flatten(), pred_30_te.flatten())
    mae_90 = mean_absolute_error(Y_te.flatten(),         pred_90_te.flatten())
    progress_cb(85, "Generating forecast")
    last_w = feat_sc[-LOOKBACK:].flatten().reshape(1, -1)
    fut_30 = np.array([m.predict(last_w)[0] for m in models_30])
    fut_90 = np.array([m.predict(last_w)[0] for m in models_90])
    fut_90[:30] = 0.5 * fut_30 + 0.5 * fut_90[:30]
    fut_90 = np.maximum(0.0, fut_90)
    last_date = daily_df.index.max()
    forecast  = [
        {"date": str((last_date + timedelta(days=i + 1)).date()), "predicted_sales": float(v)}
        for i, v in enumerate(fut_90)
    ]
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
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
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

def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"

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
        raise HTTPException(422, f"Need at least {min_rows} days of history. Got {len(daily_df)}.")
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
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )

@app.get("/health")
async def health():
    return {"status": "ok"}


# ==============================================================================
# 2. FUSE BUSINESS ADVISOR CHAT — Tool-calling approach (no bulk data load)
# ==============================================================================

fuse_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
db_engine = None

@app.on_event("startup")
def on_startup():
    global db_engine
    if DATABASE_URL:
        db_engine = create_engine(DATABASE_URL)
        print("[FUSE] Database engine initialized.")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    businessId: str
    message: str
    history: List[ChatMessage] = []


# ─── TOOL DEFINITIONS (sent to the LLM) ───────────────────────────────────────

CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_revenue_summary",
            "description": (
                "Returns total revenue, profit, order count, and average margin "
                "grouped by year and month. Use this for general performance questions, "
                "trends, or year-on-year comparisons."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_performance",
            "description": (
                "Returns revenue, profit, units sold, and margin for each product. "
                "Use this for product-level questions: best sellers, worst performers, "
                "margin leaders, pricing analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "How many products to return (default 20, max 100).",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_monthly_trend",
            "description": (
                "Returns month-by-month revenue and order count for a specific year, "
                "or the last N months. Use for seasonality, monthly deep-dives, or "
                "recent trend questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "description": "Filter to a specific year (e.g. 2024). Omit for last 12 months.",
                    },
                    "last_n_months": {
                        "type": "integer",
                        "description": "Return the last N months of data (default 12).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_products",
            "description": (
                "Returns the top N products ranked by a chosen metric: revenue, profit, "
                "margin, or units_sold. Use for 'best product' or ranking questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["revenue", "profit", "margin", "units_sold"],
                        "description": "The metric to rank by.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "How many top products to return (default 5).",
                    },
                },
                "required": ["metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_orders_summary",
            "description": (
                "Returns a summary of orders in the last N days: total revenue, order count, "
                "average order value. Use for 'recently', 'this week', 'last 30 days' questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Number of recent days to summarise (default 30).",
                    }
                },
                "required": [],
            },
        },
    },
]


# ─── TOOL EXECUTOR ────────────────────────────────────────────────────────────

def execute_tool(tool_name: str, args: dict, business_id: str) -> str:
    """Run the requested SQL tool and return a plain-text result string."""
    if not db_engine:
        return "Database not available."

    try:
        with db_engine.connect() as conn:

            # Base subquery reused across all tools
            base = """
                SELECT
                    o.created_at                                    AS order_date,
                    oi.product_id,
                    p.name                                          AS product_name,
                    oi.unit_price,
                    oi.quantity,
                    COALESCE(oi.item_discount, 0)                   AS item_discount,
                    COALESCE(p.cost, 0)                             AS cost
                FROM "order" o
                JOIN order_item oi ON o.id = oi.order_id
                JOIN product    p  ON p.id = oi.product_id
                WHERE o.business_id = :biz_id
                  AND o.status NOT IN ('cancelled', 'refunded')
            """

            if tool_name == "get_revenue_summary":
                q = text(f"""
                    WITH base AS ({base})
                    SELECT
                        EXTRACT(YEAR  FROM order_date)::int  AS year,
                        TO_CHAR(order_date, 'YYYY-MM')       AS month,
                        COUNT(*)                              AS orders,
                        SUM((unit_price - item_discount) * quantity)               AS revenue,
                        SUM(((unit_price - item_discount) - cost) * quantity)      AS profit
                    FROM base
                    GROUP BY 1, 2
                    ORDER BY 2
                """)
                df = pd.read_sql(q, conn, params={"biz_id": business_id})
                if df.empty:
                    return "No order data found."
                df["margin"] = (df["profit"] / df["revenue"].replace(0, np.nan)).fillna(0)
                yearly = df.groupby("year").agg(
                    revenue=("revenue","sum"), profit=("profit","sum"),
                    orders=("orders","sum"), margin=("margin","mean")
                ).reset_index()
                yearly["yoy_growth"] = yearly["revenue"].pct_change() * 100
                lines = ["=== Revenue Summary by Year ==="]
                for _, r in yearly.iterrows():
                    g = f" | YoY Growth: {r['yoy_growth']:+.1f}%" if pd.notna(r["yoy_growth"]) else ""
                    lines.append(
                        f"  {int(r['year'])}: Revenue={r['revenue']:,.0f} EGP | "
                        f"Profit={r['profit']:,.0f} EGP | Margin={r['margin']:.1%} | "
                        f"Orders={int(r['orders'])}{g}"
                    )
                lines.append("\n=== Last 12 Months ===")
                for _, r in df.tail(12).iterrows():
                    lines.append(f"  {r['month']}: Revenue={r['revenue']:,.0f} EGP | Orders={int(r['orders'])}")
                return "\n".join(lines)

            elif tool_name == "get_product_performance":
                limit = min(int(args.get("limit", 20)), 100)
                q = text(f"""
                    WITH base AS ({base})
                    SELECT
                        product_name,
                        COUNT(*)                                                   AS orders,
                        SUM(quantity)                                              AS units_sold,
                        SUM((unit_price - item_discount) * quantity)               AS revenue,
                        SUM(((unit_price - item_discount) - cost) * quantity)      AS profit,
                        AVG(unit_price)                                            AS avg_price
                    FROM base
                    GROUP BY product_name
                    ORDER BY revenue DESC
                    LIMIT :lim
                """)
                df = pd.read_sql(q, conn, params={"biz_id": business_id, "lim": limit})
                if df.empty:
                    return "No product data found."
                df["margin"] = (df["profit"] / df["revenue"].replace(0, np.nan)).fillna(0)
                lines = [f"=== Product Performance (top {limit} by revenue) ==="]
                for _, r in df.iterrows():
                    lines.append(
                        f"  {r['product_name']}: Revenue={r['revenue']:,.0f} EGP | "
                        f"Profit={r['profit']:,.0f} EGP | Margin={r['margin']:.1%} | "
                        f"Units={int(r['units_sold'])} | Avg Price={r['avg_price']:,.0f} EGP"
                    )
                return "\n".join(lines)

            elif tool_name == "get_monthly_trend":
                year = args.get("year")
                last_n = int(args.get("last_n_months", 12))
                if year:
                    q = text(f"""
                        WITH base AS ({base})
                        SELECT
                            TO_CHAR(order_date, 'YYYY-MM') AS month,
                            COUNT(*)                        AS orders,
                            SUM((unit_price - item_discount) * quantity) AS revenue
                        FROM base
                        WHERE EXTRACT(YEAR FROM order_date) = :yr
                        GROUP BY 1 ORDER BY 1
                    """)
                    df = pd.read_sql(q, conn, params={"biz_id": business_id, "yr": year})
                else:
                    q = text(f"""
                        WITH base AS ({base})
                        SELECT
                            TO_CHAR(order_date, 'YYYY-MM') AS month,
                            COUNT(*)                        AS orders,
                            SUM((unit_price - item_discount) * quantity) AS revenue
                        FROM base
                        GROUP BY 1 ORDER BY 1
                        LIMIT :n
                    """)
                    # get all then tail
                    q2 = text(f"""
                        WITH base AS ({base})
                        SELECT
                            TO_CHAR(order_date, 'YYYY-MM') AS month,
                            COUNT(*)                        AS orders,
                            SUM((unit_price - item_discount) * quantity) AS revenue
                        FROM base
                        GROUP BY 1 ORDER BY 1
                    """)
                    df = pd.read_sql(q2, conn, params={"biz_id": business_id}).tail(last_n)
                if df.empty:
                    return "No monthly data found."
                lines = [f"=== Monthly Trend ==="]
                for _, r in df.iterrows():
                    lines.append(f"  {r['month']}: Revenue={r['revenue']:,.0f} EGP | Orders={int(r['orders'])}")
                return "\n".join(lines)

            elif tool_name == "get_top_products":
                metric = args.get("metric", "revenue")
                limit  = int(args.get("limit", 5))
                metric_map = {
                    "revenue":    "SUM((unit_price - item_discount) * quantity)",
                    "profit":     "SUM(((unit_price - item_discount) - cost) * quantity)",
                    "units_sold": "SUM(quantity)",
                    "margin":     "SUM(((unit_price - item_discount) - cost) * quantity) / NULLIF(SUM((unit_price - item_discount) * quantity), 0)",
                }
                order_expr = metric_map.get(metric, metric_map["revenue"])
                q = text(f"""
                    WITH base AS ({base})
                    SELECT
                        product_name,
                        SUM((unit_price - item_discount) * quantity)               AS revenue,
                        SUM(((unit_price - item_discount) - cost) * quantity)      AS profit,
                        SUM(quantity)                                              AS units_sold,
                        SUM(((unit_price - item_discount) - cost) * quantity)
                            / NULLIF(SUM((unit_price - item_discount) * quantity), 0) AS margin
                    FROM base
                    GROUP BY product_name
                    ORDER BY {order_expr} DESC
                    LIMIT :lim
                """)
                df = pd.read_sql(q, conn, params={"biz_id": business_id, "lim": limit})
                if df.empty:
                    return "No product data found."
                lines = [f"=== Top {limit} Products by {metric.replace('_',' ').title()} ==="]
                for i, (_, r) in enumerate(df.iterrows(), 1):
                    lines.append(
                        f"  #{i} {r['product_name']}: Revenue={r['revenue']:,.0f} EGP | "
                        f"Profit={r['profit']:,.0f} EGP | Margin={r['margin']:.1%} | "
                        f"Units={int(r['units_sold'])}"
                    )
                return "\n".join(lines)

            elif tool_name == "get_recent_orders_summary":
                days = int(args.get("days", 30))
                q = text(f"""
                    WITH base AS ({base})
                    SELECT
                        COUNT(DISTINCT order_date::date)                           AS active_days,
                        COUNT(*)                                                   AS total_items,
                        SUM((unit_price - item_discount) * quantity)               AS revenue,
                        SUM(((unit_price - item_discount) - cost) * quantity)      AS profit,
                        AVG((unit_price - item_discount) * quantity)               AS avg_item_value
                    FROM base
                    WHERE order_date >= NOW() - INTERVAL ':days days'
                """)
                # SQLAlchemy doesn't interpolate inside strings, use explicit cast
                q2 = text(f"""
                    WITH base AS (
                        SELECT
                            o.created_at                                    AS order_date,
                            oi.unit_price,
                            oi.quantity,
                            COALESCE(oi.item_discount, 0)                   AS item_discount,
                            COALESCE(p.cost, 0)                             AS cost
                        FROM "order" o
                        JOIN order_item oi ON o.id = oi.order_id
                        JOIN product    p  ON p.id = oi.product_id
                        WHERE o.business_id = :biz_id
                          AND o.status NOT IN ('cancelled', 'refunded')
                          AND o.created_at >= NOW() - CAST(:days_val || ' days' AS INTERVAL)
                    )
                    SELECT
                        COUNT(*)                                                   AS total_items,
                        SUM((unit_price - item_discount) * quantity)               AS revenue,
                        SUM(((unit_price - item_discount) - cost) * quantity)      AS profit,
                        AVG((unit_price - item_discount) * quantity)               AS avg_item_value
                    FROM base
                """)
                df = pd.read_sql(q2, conn, params={"biz_id": business_id, "days_val": str(days)})
                if df.empty or df["revenue"].iloc[0] is None:
                    return f"No orders found in the last {days} days."
                r = df.iloc[0]
                margin = (r["profit"] / r["revenue"]) if r["revenue"] else 0
                return (
                    f"=== Last {days} Days Summary ===\n"
                    f"  Revenue:         {r['revenue']:,.0f} EGP\n"
                    f"  Profit:          {r['profit']:,.0f} EGP\n"
                    f"  Margin:          {margin:.1%}\n"
                    f"  Items Sold:      {int(r['total_items'])}\n"
                    f"  Avg Item Value:  {r['avg_item_value']:,.0f} EGP"
                )

            else:
                return f"Unknown tool: {tool_name}"

    except Exception as e:
        traceback.print_exc()
        return f"Tool error ({tool_name}): {str(e)}"


# ─── CHAT ENDPOINT ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are FUSE AI — a senior business advisor with MBA-level expertise 
in strategy, finance, pricing, operations, and growth. You have access to the business's 
live database through a set of tools. 

IMPORTANT RULES:
- Always call a tool before answering data-related questions. Never guess numbers.
- Call only the tool(s) needed to answer the specific question — don't over-fetch.
- After receiving tool results, give a sharp, insight-driven answer:
  1. Lead with the key number
  2. Diagnose what it means
  3. Give one concrete action
- Be concise. Avoid filler. Talk like a trusted advisor, not a chatbot."""


@app.post("/api/v1/chat")
async def chat_with_fuse(req: ChatRequest):
    if not req.message or not req.businessId:
        raise HTTPException(status_code=400, detail="businessId and message are required.")

    formatted_history = [{"role": m.role, "content": m.content} for m in req.history]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *formatted_history,
        {"role": "user", "content": req.message},
    ]

    loop = asyncio.get_running_loop()

    # ── Agentic tool-call loop (max 3 rounds to stay within timeout) ──────────
    MAX_ROUNDS = 3
    for round_num in range(MAX_ROUNDS):
        try:
            response = await loop.run_in_executor(
                None,
                lambda: fuse_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    tools=CHAT_TOOLS,
                    tool_choice="auto",
                    temperature=0.15,
                    max_tokens=1024,
                ),
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")

        choice = response.choices[0]

        # ── Model wants to call tools ─────────────────────────────────────────
        if choice.finish_reason == "tool_calls":
            tool_calls = choice.message.tool_calls

            # Append assistant message with tool_calls
            messages.append({
                "role": "assistant",
                "content": choice.message.content or "",
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })

            # Execute each tool and append results
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                logging.debug(f"[FUSE] Tool call: {tc.function.name}({args})")
                result_text = await loop.run_in_executor(
                    None, execute_tool, tc.function.name, args, req.businessId
                )
                logging.debug(f"[FUSE] Tool result ({tc.function.name}): {result_text[:200]}")

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result_text,
                })

            # Loop back → model will now generate a final answer
            continue

        # ── Model produced a final text answer ───────────────────────────────
        reply = choice.message.content or "I couldn't generate a response."
        return {"reply": reply}

    # Fallback if we exhausted rounds without a final answer
    return {"reply": "I reached the maximum number of tool calls. Please try rephrasing your question."}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)