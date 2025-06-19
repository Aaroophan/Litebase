from fastapi import FastAPI, HTTPException, Depends, Query, Request
from pydantic import BaseModel
import sqlite3
import os
from typing import Dict, Any, List, Optional

app = FastAPI()

# Configuration
DATABASE_FILE = "default.db"
API_KEY = "DEMO_KEY"

# Pydantic models
class InsertRequest(BaseModel):
    data: Dict[str, Any]

class RowResponse(BaseModel):
    id: int
    data: Dict[str, Any]
    created_at: str

class QueryResponse(BaseModel):
    data: List[RowResponse]
    next_offset: Optional[int] = None

# Database connection
def get_db():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn

# Fixed Authentication
def api_key_auth(request: Request, api_key: str = Query(...)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# Helper functions
def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cursor = conn.cursor()
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cursor.fetchone() is not None

def create_table(conn: sqlite3.Connection, table_name: str, data: dict):
    cursor = conn.cursor()
    columns = ["id INTEGER PRIMARY KEY AUTOINCREMENT", "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"]
    
    for key, value in data.items():
        if isinstance(value, int):
            col_type = "INTEGER"
        elif isinstance(value, float):
            col_type = "REAL"
        else:
            col_type = "TEXT"
        columns.append(f"{key} {col_type}")
    
    create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
    cursor.execute(create_sql)
    conn.commit()

# API Endpoints
@app.post("/tables/{table_name}", response_model=RowResponse, status_code=201)
def insert_row(
    table_name: str, 
    request: InsertRequest,
    authorized: bool = Depends(api_key_auth)
):
    conn = get_db()
    
    # Create table if doesn't exist
    if not table_exists(conn, table_name):
        create_table(conn, table_name, request.data)
    
    # Insert data
    cursor = conn.cursor()
    columns = ', '.join(request.data.keys())
    placeholders = ', '.join(['?'] * len(request.data))
    values = list(request.data.values())
    
    cursor.execute(
        f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})",
        values
    )
    row_id = cursor.lastrowid
    
    # Fetch created row
    cursor.execute(
        f"SELECT *, datetime(created_at) as created_at FROM {table_name} WHERE id = ?",
        (row_id,)
    )
    row = cursor.fetchone()
    conn.commit()
    conn.close()
    
    return {
        "id": row["id"],
        "data": dict(row),
        "created_at": row["created_at"]
    }

@app.get("/tables/{table_name}", response_model=QueryResponse)
def query_rows(
    table_name: str,
    request: Request,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    authorized: bool = Depends(api_key_auth)
):
    conn = get_db()
    
    # Validate table exists
    if not table_exists(conn, table_name):
        raise HTTPException(status_code=404, detail="Table not found")
    
    # Extract filters from query parameters
    filters = {}
    reserved_params = ["limit", "offset", "api_key"]
    for key, value in request.query_params.items():
        if key not in reserved_params:
            filters[key] = value
    
    # Build query
    where_clause = ""
    params = []
    for col, value in filters.items():
        where_clause += f" AND {col} = ?" if where_clause else f" WHERE {col} = ?"
        params.append(value)
    
    # Fetch data
    cursor = conn.cursor()
    base_query = f"SELECT *, datetime(created_at) as created_at FROM {table_name}"
    cursor.execute(
        f"{base_query}{where_clause} LIMIT ? OFFSET ?",
        params + [limit + 1, offset]
    )
    rows = cursor.fetchall()
    conn.close()
    
    # Check for next page
    next_offset = None
    if len(rows) > limit:
        rows = rows[:-1]
        next_offset = offset + limit
    
    return {
        "data": [
            {
                "id": row["id"],
                "data": dict(row),
                "created_at": row["created_at"]
            } for row in rows
        ],
        "next_offset": next_offset
    }

@app.patch("/tables/{table_name}/{row_id}", response_model=RowResponse)
def update_row(
    table_name: str,
    row_id: int,
    request: InsertRequest,
    authorized: bool = Depends(api_key_auth)
):
    conn = get_db()
    
    # Validate table exists
    if not table_exists(conn, table_name):
        raise HTTPException(status_code=404, detail="Table not found")
    
    # Build update statement
    set_clause = ', '.join([f"{key} = ?" for key in request.data.keys()])
    values = list(request.data.values())
    
    cursor = conn.cursor()
    cursor.execute(
        f"UPDATE {table_name} SET {set_clause} WHERE id = ?",
        values + [row_id]
    )
    
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Row not found")
    
    # Fetch updated row
    cursor.execute(
        f"SELECT *, datetime(created_at) as created_at FROM {table_name} WHERE id = ?",
        (row_id,)
    )
    row = cursor.fetchone()
    conn.commit()
    conn.close()
    
    return {
        "id": row["id"],
        "data": dict(row),
        "created_at": row["created_at"]
    }

@app.delete("/tables/{table_name}/{row_id}", status_code=204)
def delete_row(
    table_name: str,
    row_id: int,
    authorized: bool = Depends(api_key_auth)
):
    conn = get_db()
    
    # Validate table exists
    if not table_exists(conn, table_name):
        raise HTTPException(status_code=404, detail="Table not found")
    
    cursor = conn.cursor()
    cursor.execute(
        f"DELETE FROM {table_name} WHERE id = ?",
        (row_id,)
    )
    
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Row not found")
    
    conn.commit()
    conn.close()
    return {}