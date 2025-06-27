import uuid
import sqlite3
import os
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from contextlib import contextmanager
from fastapi import FastAPI, HTTPException, Depends, Query, Header, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, field_validator
from jose import jwt, JWTError
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Litebase API",
    description="Instant, isolated SQLite databases with REST APIs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Configuration =====
ADMIN_TOKEN = os.getenv("Aaroophan", "Password")  # Admin secret
PROJECT_DB_DIR = "project_dbs"  # Database storage
MAX_DB_SIZE = 100 * 1024 * 1024  # 100MB limit per DB
MAX_TABLE_NAME_LENGTH = 255  # Table name limit
MAX_COLUMN_NAME_LENGTH = 255  # Column name limit

os.makedirs(PROJECT_DB_DIR, exist_ok=True)

# Generate server RSA key pair
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
    backend=default_backend()
)
public_key = private_key.public_key()

# ===== Enhanced Data Models =====
class ProjectCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=512)

class ProjectResponse(BaseModel):
    project_id: str
    api_token: str
    name: str
    created_at: str

class InsertRequest(BaseModel):
    data: Dict[str, Any] = Field(..., min_items=1)
    
    @field_validator('data')
    def validate_data_keys(cls, v):
        reserved_words = {'id', 'created_at', 'owner_id', 'updated_at'}
        for key in v.keys():
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):
                raise ValueError(f"Invalid column name: {key}. Must start with letter/underscore and contain only alphanumeric characters and underscores.")
            if len(key) > MAX_COLUMN_NAME_LENGTH:
                raise ValueError(f"Column name too long: {key}. Maximum length is {MAX_COLUMN_NAME_LENGTH}.")
            if key.lower() in reserved_words:
                raise ValueError(f"Reserved column name: {key}")
        return v

class UpdateRequest(BaseModel):
    data: Dict[str, Any] = Field(..., min_items=1)
    
    @field_validator('data')
    def validate_data_keys(cls, v):
        reserved_words = {'id', 'created_at', 'owner_id'}
        for key in v.keys():
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):
                raise ValueError(f"Invalid column name: {key}")
            if len(key) > MAX_COLUMN_NAME_LENGTH:
                raise ValueError(f"Column name too long: {key}")
            if key.lower() in reserved_words:
                raise ValueError(f"Reserved column name: {key}")
        return v

class RowResponse(BaseModel):
    id: int
    data: Dict[str, Any]
    created_at: str
    updated_at: Optional[str] = None
    owner_id: str

class QueryResponse(BaseModel):
    data: List[RowResponse]
    total_count: Optional[int] = None
    next_offset: Optional[int] = None
    has_more: bool = False

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str

# ===== Enhanced Helper Functions =====
def validate_table_name(table_name: str) -> str:
    """Validate and sanitize table name"""
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
        raise HTTPException(400, "Invalid table name. Must start with letter/underscore and contain only alphanumeric characters and underscores.")
    if len(table_name) > MAX_TABLE_NAME_LENGTH:
        raise HTTPException(400, f"Table name too long. Maximum length is {MAX_TABLE_NAME_LENGTH}.")
    return table_name

@contextmanager
def get_db_connection(project_id: str):
    """Context manager for database connections with proper cleanup"""
    db_path = os.path.join(PROJECT_DB_DIR, f"{project_id}.db")
    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Enable foreign keys and WAL mode for better performance
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        yield conn
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error for project {project_id}: {e}")
        raise HTTPException(500, "Database operation failed")
    finally:
        if conn:
            conn.close()

def check_db_size(project_id: str):
    """Check if database size exceeds limit"""
    db_path = os.path.join(PROJECT_DB_DIR, f"{project_id}.db")
    if os.path.exists(db_path) and os.path.getsize(db_path) > MAX_DB_SIZE:
        raise HTTPException(413, "Database size limit exceeded")

def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    return cursor.fetchone() is not None

def get_table_columns(conn: sqlite3.Connection, table_name: str) -> Dict[str, str]:
    """Get existing table columns and their types"""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = {}
    for row in cursor.fetchall():
        columns[row[1]] = row[2]  # name: type
    return columns

def infer_sqlite_type(value: Any) -> str:
    """Infer SQLite type from Python value"""
    if isinstance(value, bool):
        return "BOOLEAN"
    elif isinstance(value, int):
        return "INTEGER"
    elif isinstance(value, float):
        return "REAL"
    elif isinstance(value, (dict, list)):
        return "JSON"  # We'll store as TEXT but validate JSON
    else:
        return "TEXT"

def create_table(conn: sqlite3.Connection, table_name: str, data: dict, project_id: str):
    cursor = conn.cursor()
    columns = [
        "id INTEGER PRIMARY KEY AUTOINCREMENT",
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        f"owner_id TEXT NOT NULL DEFAULT '{project_id}'"
    ]
    
    for key, value in data.items():
        col_type = infer_sqlite_type(value)
        columns.append(f'"{key}" {col_type}')
    
    create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(columns)})'
    cursor.execute(create_sql)
    
    # Create index on owner_id for performance
    cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_owner ON "{table_name}"(owner_id)')
    conn.commit()

def add_missing_columns(conn: sqlite3.Connection, table_name: str, new_data: dict):
    """Add any missing columns to existing table"""
    existing_columns = get_table_columns(conn, table_name)
    cursor = conn.cursor()
    
    for key, value in new_data.items():
        if key not in existing_columns:
            col_type = infer_sqlite_type(value)
            try:
                cursor.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{key}" {col_type}')
            except sqlite3.OperationalError as e:
                logger.warning(f"Failed to add column {key} to {table_name}: {e}")
    
    conn.commit()

def verify_project_token(token: str) -> str:
    try:
        # Remove 'Bearer ' prefix if present
        if token.startswith('Bearer '):
            token = token[7:]
        
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            options={"verify_aud": False}
        )
        return payload["project_id"]
    except JWTError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )

def build_where_clause(filters: Dict[str, str], project_id: str) -> tuple[str, List[Any]]:
    conditions = ["owner_id = ?"]
    params = [project_id]
    
    for key, value in filters.items():
        if "__" in key:
            col, operator = key.split("__", 1)
            # Validate column name
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', col):
                raise HTTPException(400, f"Invalid column name in filter: {col}")
            
            if operator == "gt":
                conditions.append(f'"{col}" > ?')
                params.append(value)
            elif operator == "lt":
                conditions.append(f'"{col}" < ?')
                params.append(value)
            elif operator == "gte":
                conditions.append(f'"{col}" >= ?')
                params.append(value)
            elif operator == "lte":
                conditions.append(f'"{col}" <= ?')
                params.append(value)
            elif operator == "ne":
                conditions.append(f'"{col}" != ?')
                params.append(value)
            elif operator == "in":
                items = [item.strip() for item in value.split(",")]
                placeholders = ",".join("?" * len(items))
                conditions.append(f'"{col}" IN ({placeholders})')
                params.extend(items)
            elif operator == "contains":
                conditions.append(f'"{col}" LIKE ?')
                params.append(f"%{value}%")
            elif operator == "startswith":
                conditions.append(f'"{col}" LIKE ?')
                params.append(f"{value}%")
            elif operator == "endswith":
                conditions.append(f'"{col}" LIKE ?')
                params.append(f"%{value}")
            else:
                raise HTTPException(400, f"Invalid operator: {operator}")
        else:
            # Validate column name
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):
                raise HTTPException(400, f"Invalid column name in filter: {key}")
            conditions.append(f'"{key}" = ?')
            params.append(value)
    
    return " AND ".join(conditions), params

def build_order_clause(sort: str) -> str:
    if not sort:
        return ""
    
    order_parts = []
    for field in sort.split(","):
        field = field.strip()
        if field.startswith("-"):
            col_name = field[1:]
            direction = "DESC"
        else:
            col_name = field
            direction = "ASC"
        
        # Validate column name
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', col_name):
            raise HTTPException(400, f"Invalid column name in sort: {col_name}")
        
        order_parts.append(f'"{col_name}" {direction}')
    
    return "ORDER BY " + ", ".join(order_parts)

# ===== Exception Handlers =====
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# ===== Admin Endpoints =====
@app.post("/admin/projects", response_model=ProjectResponse, status_code=201)
def create_project(
    request: ProjectCreateRequest,
    admin_token: str = Header(..., alias="X-Admin-Token")
):
    if admin_token != ADMIN_TOKEN:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid admin token")
    
    project_id = f"proj_{uuid.uuid4().hex[:10]}"
    created_at = datetime.utcnow()
    
    api_token = jwt.encode(
        {
            "project_id": project_id,
            "name": request.name,
            "created_at": created_at.isoformat(),
            "exp": created_at + timedelta(days=365)
        },
        private_key,
        algorithm="RS256"
    )
    
    # Create empty database file
    db_path = os.path.join(PROJECT_DB_DIR, f"{project_id}.db")
    with sqlite3.connect(db_path) as conn:
        # Initialize with metadata table
        conn.execute("""
            CREATE TABLE _litebase_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(
            "INSERT INTO _litebase_metadata (key, value) VALUES (?, ?)",
            ("project_name", request.name)
        )
        if request.description:
            conn.execute(
                "INSERT INTO _litebase_metadata (key, value) VALUES (?, ?)",
                ("description", request.description)
            )
        conn.commit()
    
    logger.info(f"Created project {project_id} with name '{request.name}'")
    
    return {
        "project_id": project_id,
        "api_token": api_token,
        "name": request.name,
        "created_at": created_at.isoformat()
    }

# ===== Project Endpoints =====
@app.post("/{project_id}/tables/{table_name}", response_model=RowResponse, status_code=201)
def insert_row(
    project_id: str,
    table_name: str,
    request: InsertRequest,
    authorization: str = Header(..., alias="Authorization")
):
    table_name = validate_table_name(table_name)
    token_project_id = verify_project_token(authorization)
    if token_project_id != project_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Token project mismatch")
    
    check_db_size(project_id)
    
    with get_db_connection(project_id) as conn:
        # Create table if doesn't exist
        if not table_exists(conn, table_name):
            create_table(conn, table_name, request.data, project_id)
        else:
            # Add any missing columns
            add_missing_columns(conn, table_name, request.data)
        
        # Prepare data for insertion
        data_with_owner = request.data.copy()
        data_with_owner["owner_id"] = project_id
        
        cursor = conn.cursor()
        columns = ', '.join([f'"{k}"' for k in data_with_owner.keys()])
        placeholders = ', '.join(['?'] * len(data_with_owner))
        values = list(data_with_owner.values())
        
        cursor.execute(
            f'INSERT INTO "{table_name}" ({columns}) VALUES ({placeholders})',
            values
        )
        row_id = cursor.lastrowid
        
        # Fetch created row
        cursor.execute(
            f'SELECT *, datetime(created_at) as created_at, datetime(updated_at) as updated_at FROM "{table_name}" WHERE id = ?',
            (row_id,)
        )
        row = cursor.fetchone()
        conn.commit()
        
        return {
            "id": row["id"],
            "data": dict(row),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "owner_id": row["owner_id"]
        }

@app.get("/{project_id}/tables/{table_name}", response_model=QueryResponse)
def query_rows(
    project_id: str,
    table_name: str,
    filter: Optional[str] = Query(None, description="Filter conditions (e.g., price__gt=100,status=active)"),
    sort: Optional[str] = Query(None, description="Sort fields (e.g., -created_at,price)"),
    fields: Optional[str] = Query(None, description="Select specific fields (e.g., id,name,price)"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    count: bool = Query(False, description="Include total count"),
    authorization: str = Header(..., alias="Authorization")
):
    table_name = validate_table_name(table_name)
    token_project_id = verify_project_token(authorization)
    if token_project_id != project_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Token project mismatch")
    
    with get_db_connection(project_id) as conn:
        # Validate table exists
        if not table_exists(conn, table_name):
            raise HTTPException(404, "Table not found")
        
        # Parse filter string into dictionary
        filters = {}
        if filter:
            for condition in filter.split(","):
                condition = condition.strip()
                if "=" in condition:
                    key, value = condition.split("=", 1)
                    filters[key.strip()] = value.strip()
        
        # Build WHERE clause with security enforcement
        where_clause, where_params = build_where_clause(filters, project_id)
        where_clause = f"WHERE {where_clause}" if where_clause else ""
        
        # Build ORDER BY clause
        order_clause = build_order_clause(sort) if sort else ""
        
        # Build SELECT clause
        if fields:
            field_list = [f.strip() for f in fields.split(",")]
            # Validate field names and ensure required fields are included
            validated_fields = ["id", "created_at", "updated_at", "owner_id"]
            for field in field_list:
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', field):
                    if field not in validated_fields:
                        validated_fields.append(field)
            select_clause = ', '.join([f'"{f}"' for f in validated_fields])
        else:
            select_clause = "*"
        
        cursor = conn.cursor()
        
        # Get total count if requested
        total_count = None
        if count:
            count_query = f'SELECT COUNT(*) FROM "{table_name}" {where_clause}'
            cursor.execute(count_query, where_params)
            total_count = cursor.fetchone()[0]
        
        # Build main query
        base_query = (
            f'SELECT {select_clause}, datetime(created_at) as created_at, '
            f'datetime(updated_at) as updated_at FROM "{table_name}" '
            f'{where_clause} {order_clause}'
        )
        
        # Execute query with pagination
        cursor.execute(
            f"{base_query} LIMIT ? OFFSET ?",
            where_params + [limit + 1, offset]
        )
        rows = cursor.fetchall()
        
        # Check for next page
        has_more = len(rows) > limit
        next_offset = None
        if has_more:
            rows = rows[:-1]
            next_offset = offset + limit
        
        return {
            "data": [
                {
                    "id": row["id"],
                    "data": dict(row),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "owner_id": row["owner_id"]
                } for row in rows
            ],
            "total_count": total_count,
            "next_offset": next_offset,
            "has_more": has_more
        }

@app.patch("/{project_id}/tables/{table_name}/{row_id}", response_model=RowResponse)
def update_row(
    project_id: str,
    table_name: str,
    row_id: int,
    request: UpdateRequest,
    authorization: str = Header(..., alias="Authorization")
):
    table_name = validate_table_name(table_name)
    token_project_id = verify_project_token(authorization)
    if token_project_id != project_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Token project mismatch")
    
    check_db_size(project_id)
    
    with get_db_connection(project_id) as conn:
        # Validate table exists
        if not table_exists(conn, table_name):
            raise HTTPException(404, "Table not found")
        
        # Add any missing columns
        add_missing_columns(conn, table_name, request.data)
        
        # Build update statement
        set_clauses = [f'"{key}" = ?' for key in request.data.keys()]
        set_clauses.append('updated_at = CURRENT_TIMESTAMP')
        set_clause = ', '.join(set_clauses)
        values = list(request.data.values())
        
        cursor = conn.cursor()
        cursor.execute(
            f'UPDATE "{table_name}" SET {set_clause} WHERE id = ? AND owner_id = ?',
            values + [row_id, project_id]
        )
        
        if cursor.rowcount == 0:
            raise HTTPException(404, "Row not found or access denied")
        
        # Fetch updated row
        cursor.execute(
            f'SELECT *, datetime(created_at) as created_at, datetime(updated_at) as updated_at FROM "{table_name}" WHERE id = ?',
            (row_id,)
        )
        row = cursor.fetchone()
        conn.commit()
        
        return {
            "id": row["id"],
            "data": dict(row),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "owner_id": row["owner_id"]
        }

@app.delete("/{project_id}/tables/{table_name}/{row_id}", status_code=204)
def delete_row(
    project_id: str,
    table_name: str,
    row_id: int,
    authorization: str = Header(..., alias="Authorization")
):
    table_name = validate_table_name(table_name)
    token_project_id = verify_project_token(authorization)
    if token_project_id != project_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Token project mismatch")
    
    with get_db_connection(project_id) as conn:
        # Validate table exists
        if not table_exists(conn, table_name):
            raise HTTPException(404, "Table not found")
        
        cursor = conn.cursor()
        cursor.execute(
            f'DELETE FROM "{table_name}" WHERE id = ? AND owner_id = ?',
            (row_id, project_id)
        )
        
        if cursor.rowcount == 0:
            raise HTTPException(404, "Row not found or access denied")
        
        conn.commit()

# ===== Health and Info Endpoints =====
@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/{project_id}/info")
def project_info(
    project_id: str,
    authorization: str = Header(..., alias="Authorization")
):
    token_project_id = verify_project_token(authorization)
    if token_project_id != project_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Token project mismatch")
    
    db_path = os.path.join(PROJECT_DB_DIR, f"{project_id}.db")
    if not os.path.exists(db_path):
        raise HTTPException(404, "Project not found")
    
    with get_db_connection(project_id) as conn:
        cursor = conn.cursor()
        
        # Get project metadata
        metadata = {}
        try:
            cursor.execute("SELECT key, value FROM _litebase_metadata")
            for row in cursor.fetchall():
                metadata[row[0]] = row[1]
        except sqlite3.OperationalError:
            pass  # Metadata table doesn't exist in older projects
        
        # Get table information
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name != '_litebase_metadata'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        # Get database size
        db_size = os.path.getsize(db_path)
    
    return {
        "project_id": project_id,
        "metadata": metadata,
        "tables": tables,
        "database_size_bytes": db_size,
        "max_database_size_bytes": MAX_DB_SIZE
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="http://127.0.0.1", port=8000)