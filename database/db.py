"""
Database Management Module
Provides thread-safe connection pooling, initialization, and CRUD operations.
"""

import sqlite3
import os
from flask import g

DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'schema.sql')


def get_db(db_path=None):
    """
    Returns a database connection for current request context or standalone script.
    """
    path = db_path or DATABASE_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # If running inside Flask context
    try:
        if 'db' not in g:
            g.db = sqlite3.connect(path)
            g.db.row_factory = sqlite3.Row
        return g.db
    except RuntimeError:
        # Outside Flask context (scripts, migrations)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return conn


def close_db(e=None):
    """Closes the current request's database connection."""
    try:
        db = g.pop('db', None)
        if db is not None:
            db.close()
    except RuntimeError:
        pass


def init_db(db_path=None):
    """Initializes tables from schema.sql."""
    conn = get_db(db_path)
    with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)
    conn.commit()
    print("[+] Database initialized successfully.")
    return conn


def query_db(query, args=(), one=False, db_path=None):
    """Convenience helper for parameterized SQL execution."""
    conn = get_db(db_path)
    cur = conn.cursor()
    cur.execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv


def execute_db(query, args=(), db_path=None):
    """Convenience helper for INSERT/UPDATE/DELETE queries."""
    conn = get_db(db_path)
    cur = conn.cursor()
    cur.execute(query, args)
    conn.commit()
    last_id = cur.lastrowid
    cur.close()
    return last_id


if __name__ == "__main__":
    init_db()
