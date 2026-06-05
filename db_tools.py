from sqlalchemy import text

def run_sql_query(conn, query: str, params: dict):
    """
    SAFE SQL EXECUTOR (read-only)
    """

    lowered = query.strip().lower()

    # 🚨 safety guardrails
    if not lowered.startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    if "delete" in lowered or "update" in lowered or "insert" in lowered:
        raise ValueError("Write operations are not allowed.")

    result = conn.execute(text(query), params)
    rows = result.fetchall()

    return [dict(row._mapping) for row in rows]

def get_monthly_sales(conn, business_id: str):
    query = """
    SELECT 
        DATE_TRUNC('month', created_at) AS month,
        SUM(unit_price * quantity - item_discount) AS revenue,
        SUM((unit_price - cost) * quantity) AS profit
    FROM order_item oi
    JOIN "order" o ON o.id = oi.order_id
    JOIN product p ON p.id = oi.product_id
    WHERE o.business_id = :business_id
    GROUP BY month
    ORDER BY month;
    """

    return run_sql_query(conn, query, {"business_id": business_id})

def get_recent_orders(conn, business_id: str, limit: int = 20):
    query = """
    SELECT 
        o.created_at,
        SUM(oi.unit_price * oi.quantity) as total
    FROM "order" o
    JOIN order_item oi ON o.id = oi.order_id
    WHERE o.business_id = :business_id
    GROUP BY o.id, o.created_at
    ORDER BY o.created_at DESC
    LIMIT :limit
    """

    return run_sql_query(conn, query, {
        "business_id": business_id,
        "limit": limit
    })