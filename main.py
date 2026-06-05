from __future__ import annotations
import os
import json
import random
import asyncio
import warnings
from datetime import timedelta
from typing import List, Dict, Any
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

# ─── NEW IMPORTS FOR CHAT ROUTE ───────────────────────────────────────────────
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from openai import OpenAI

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
DATABASE_URL = os.environ.get("DATABASE_URL", "") # Added for Chat DB connection

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
# 1. YOUR EXACT ORIGINAL FORECASTING PIPELINE & ROUTE
# ==============================================================================

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
        delayed(_fit)(X_tr, Y_tr[:, d], 200, 5, 8, 0.1)
        for d in range(30)
    )
    progress_cb(48, "30-day models complete")

    progress_cb(50, "Training 90-day models (parallel)")
    models_90 = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_fit)(X_tr, Y_tr[:, d], 150, 4, 10, 0.2)
        for d in range(HORIZON)
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
    fut_90[:30] = 0.5 * fut_30 + 0.5 * fut_90[:30]   # blend short-horizon
    fut_90 = np.maximum(0.0, fut_90)

    last_date = daily_df.index.max()
    forecast  = [
        {"date": str((last_date + timedelta(days=i + 1)).date()),
         "predicted_sales": float(v)}
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

def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"

# ─── YOUR ORIGINAL FORECAST ENDPOINT ──────────────────────────────────────────

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
        headers={
            "Cache-Control":      "no-cache",
            "X-Accel-Buffering":  "no",
            "Connection":         "keep-alive",
        },
    )

@app.get("/health")
async def health():
    return {"status": "ok"}


# ==============================================================================
# 2. NEW FUSE BUSINESS ADVISOR CHAT (Appended safely)
# ==============================================================================

fuse_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
embedder = None
db_engine = None

@app.on_event("startup")
def on_startup():
    global embedder, db_engine
    print("[FUSE] Initializing embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("[FUSE] Model loaded.")
    
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

@dataclass
class VectorChunk:
    id: str
    document: str
    embedding: np.ndarray

@dataclass
class FuseState:
    rows: List[Dict[str, Any]]
    chunks: List[VectorChunk]
    data_summary: str
    yearly_stats: List[Dict[str, Any]]

_cache: Dict[str, FuseState] = {}
_inflight: Dict[str, asyncio.Task] = {}

def load_data_from_db(business_id: str) -> List[Dict[str, Any]]:
    if not db_engine:
        raise Exception("DATABASE_URL is not set.")

    query = text("""
        SELECT 
            o."createdAt" as order_date, 
            oi."productId" as product_id, 
            p."name" as product_name, 
            oi."unitPrice" as unit_price, 
            oi."quantity" as quantity, 
            oi."itemDiscount" as item_discount, 
            p."cost" as cost
        FROM "order" o
        INNER JOIN "orderItem" oi ON o.id = oi."orderId"
        INNER JOIN "product" p ON p.id = oi."productId"
        WHERE o."businessId" = :biz_id 
          AND o."status" NOT IN ('cancelled', 'refunded')
    """)

    with db_engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"biz_id": business_id})

    if df.empty:
        return []

    df = df.dropna(subset=['order_date', 'product_id', 'unit_price'])
    df['price'] = df['unit_price'].astype(float).fillna(0)
    df['quantity'] = df['quantity'].astype(float).fillna(1)
    df['discount'] = df['item_discount'].astype(float).fillna(0)
    df['cost'] = df['cost'].astype(float).fillna(0)
    
    df['revenue'] = (df['price'] - df['discount']) * df['quantity']
    df['profit'] = df['revenue'] - (df['cost'] * df['quantity'])
    df['profit_margin'] = np.where(df['revenue'] > 0, df['profit'] / df['revenue'], 0)
    
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['year'] = df['order_date'].dt.year
    df['month'] = df['order_date'].dt.strftime('%Y-%m')

    return df.to_dict('records')

def build_aggregate_documents(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    df = pd.DataFrame(rows)
    docs = []
    if df.empty: return docs

    monthly = df.groupby('month').agg(
        rev=('revenue', 'sum'), prof=('profit', 'sum'), orders=('revenue', 'count'),
        marg=('profit_margin', 'mean'), qty=('quantity', 'sum')
    ).reset_index()
    for _, r in monthly.iterrows():
        docs.append({"id": f"monthly_{r['month']}", "doc": 
            f"Month {r['month']}: Revenue={r['rev']:,.0f} EGP, Profit={r['prof']:,.0f} EGP, "
            f"Orders={r['orders']}, Margin={r['marg']:.1%}, Units Sold={r['qty']:,.0f}"})

    prod = df.groupby(['product_id', 'product_name']).agg(
        rev=('revenue', 'sum'), prof=('profit', 'sum'), orders=('revenue', 'count'),
        marg=('profit_margin', 'mean'), qty=('quantity', 'sum'), price=('price', 'mean')
    ).reset_index()
    for _, r in prod.iterrows():
        docs.append({"id": f"product_{r['product_id']}", "doc": 
            f"Product '{r['product_name']}': Total Revenue={r['rev']:,.0f} EGP, "
            f"Total Profit={r['prof']:,.0f} EGP, Orders={r['orders']}, Avg Margin={r['marg']:.1%}, "
            f"Units Sold={r['qty']:,.0f}, Avg Price={r['price']:,.0f} EGP"})

    yearly = df.groupby('year').agg(
        rev=('revenue', 'sum'), prof=('profit', 'sum'), orders=('revenue', 'count'),
        marg=('profit_margin', 'mean'), qty=('quantity', 'sum')
    ).reset_index()
    for _, r in yearly.iterrows():
        docs.append({"id": f"year_{r['year']}", "doc": 
            f"Year {r['year']}: Revenue={r['rev']:,.0f} EGP, Profit={r['prof']:,.0f} EGP, "
            f"Orders={r['orders']}, Avg Margin={r['marg']:.1%}, Units Sold={r['qty']:,.0f}"})

    return docs

def build_vector_chunks(rows: List[Dict[str, Any]]) -> List[VectorChunk]:
    docs = build_aggregate_documents(rows)
    if not docs: return []
    texts = [d["doc"] for d in docs]
    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return [VectorChunk(id=docs[i]["id"], document=docs[i]["doc"], embedding=embeddings[i]) for i in range(len(docs))]

def cosine_sim(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    return float(np.dot(query_emb, doc_emb))

def query_chunks(query: str, chunks: List[VectorChunk], top_k: int = 12) -> List[str]:
    if not chunks: return []
    query_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scored = [(c.document, cosine_sim(query_emb, c.embedding)) for c in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored[:top_k]]

def build_data_summary(rows: List[Dict[str, Any]]) -> str:
    if not rows: return "No order data available for this business yet."
    df = pd.DataFrame(rows)
    
    yearly = df.groupby("year").agg(
        rev=("revenue", "sum"), prof=("profit", "sum"), 
        orders=("revenue", "count"), marg=("profit_margin", "mean")
    ).reset_index().sort_values("year")
    yearly["rev_growth"] = yearly["rev"].pct_change() * 100
    
    yearly_str = ""
    for _, r in yearly.iterrows():
        g = f" (YoY: {r['rev_growth']:+.1f}%)" if pd.notna(r["rev_growth"]) else " (base year)"
        yearly_str += f"  {int(r['year'])}: Revenue={r['rev']:>14,.0f} EGP | Profit={r['prof']:>12,.0f} EGP | Margin={r['marg']:.1%} | Orders={r['orders']}{g}\n"

    monthly_trend = df.groupby("month")["revenue"].sum().sort_index().tail(12)
    monthly_str = "\n".join(f"  {m}: {v:,.0f} EGP" for m, v in monthly_trend.items())

    p_rev = df.groupby("product_name")["revenue"].sum().nlargest(5)
    p_pro = df.groupby("product_name")["profit"].sum().nlargest(5)
    p_mar = df.groupby("product_name")["profit_margin"].mean().nlargest(5)

    return f"""
════════════════════════════════════════════════
FUSE BUSINESS INTELLIGENCE
════════════════════════════════════════════════
▸ PORTFOLIO SNAPSHOT
  Total Orders  : {len(df):,}
  Total Revenue : {df['revenue'].sum():,.0f} EGP
  Total Profit  : {df['profit'].sum():,.0f} EGP
  Avg Order Size: {df['price'].mean():,.0f} EGP
  Avg Margin    : {df['profit_margin'].mean():.1%}

▸ YEAR-ON-YEAR PERFORMANCE
{yearly_str}
▸ LAST 12 MONTHS — MONTHLY REVENUE
{monthly_str}

▸ TOP PRODUCTS — REVENUE
{chr(10).join(f"  {k}: {v:,.0f} EGP" for k, v in p_rev.items())}

▸ TOP PRODUCTS — PROFIT
{chr(10).join(f"  {k}: {v:,.0f} EGP" for k, v in p_pro.items())}

▸ TOP PRODUCTS — MARGIN
{chr(10).join(f"  {k}: {v:.1%}" for k, v in p_mar.items())}
════════════════════════════════════════════════
"""

async def _init_fuse_sync(business_id: str) -> FuseState:
    loop = asyncio.get_running_loop()
    rows = await loop.run_in_executor(None, load_data_from_db, business_id)
    data_summary = await loop.run_in_executor(None, build_data_summary, rows)
    
    yearly_stats = []
    if rows:
        df = pd.DataFrame(rows)
        y_df = df.groupby("year").agg(rev=("revenue", "sum"), prof=("profit", "sum"), marg=("profit_margin", "mean")).reset_index()
        y_df["growth"] = y_df["rev"].pct_change() * 100
        for _, r in y_df.iterrows():
            yearly_stats.append({
                "year": int(r['year']), "revenue": r['rev'], "profit": r['prof'], 
                "margin": r['marg'], "growth": r['growth'] if pd.notna(r['growth']) else None
            })

    chunks = await loop.run_in_executor(None, build_vector_chunks, rows)
    return FuseState(rows=rows, chunks=chunks, data_summary=data_summary, yearly_stats=yearly_stats)

async def init_fuse(business_id: str) -> FuseState:
    if business_id in _cache: return _cache[business_id]
    if business_id in _inflight: return await _inflight[business_id]
        
    task = asyncio.create_task(_init_fuse_sync(business_id))
    _inflight[business_id] = task
    try:
        state = await task
        _cache[business_id] = state
        return state
    finally:
        _inflight.pop(business_id, None)

@app.post("/api/v1/chat")
async def chat_with_fuse(req: ChatRequest):
    if not req.message or not req.businessId:
        raise HTTPException(status_code=400, detail="businessId and message are required.")

    try:
        state = await init_fuse(req.businessId)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database fetch failed: {str(e)}")

    retrieved_docs = query_chunks(req.message, state.chunks, top_k=12)
    
    yearly_anchor_parts = []
    for r in state.yearly_stats:
        growth_str = f"{r['growth']:+.1f}%" if r['growth'] is not None else "N/A"
        yearly_anchor_parts.append(
            f"Year {r['year']}: Revenue={r['revenue']:,.0f} EGP, Profit={r['profit']:,.0f} EGP, "
            f"Margin={r['margin']:.1%}, YoY Growth={growth_str}"
        )
    context = f"[Yearly Anchors]\n{chr(10).join(yearly_anchor_parts)}\n\n[Retrieved Context]\n{chr(10).join(retrieved_docs)}"

    system_prompt = f"""
You are FUSE AI — a senior business advisor embedded inside this company.
You have an MBA-level grasp of strategy, finance, pricing, operations, and growth.

════════════════════════════════════════════════
BUSINESS INTELLIGENCE
════════════════════════════════════════════════
{state.data_summary}
════════════════════════════════════════════════

━━━ THE CONSULTANT STANDARD ━━━
1. ANCHOR IN DATA FIRST. Open with the most relevant hard number.
2. DIAGNOSE WHAT THE DATA IS TELLING YOU. 
3. APPLY BUSINESS EXPERTISE. Layer in the "so what".
4. FOR FUTURE QUESTIONS: Extrapolate from the trend.
5. CLOSE WITH ONE SHARP ACTION.
"""

    formatted_history = [{"role": m.role, "content": m.content} for m in req.history]
    messages = [
        {"role": "system", "content": system_prompt},
        *formatted_history,
        {"role": "user", "content": f"Question: {req.message}\n\nRelevant Context:\n{context}"}
    ]

    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: fuse_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.15,
            max_tokens=1024
        ))
        
        reply = response.choices[0].message.content
        return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

@app.post("/api/v1/cache/clear")
def clear_cache(businessId: str):
    _cache.pop(businessId, None)
    return {"status": "cleared", "businessId": businessId}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)