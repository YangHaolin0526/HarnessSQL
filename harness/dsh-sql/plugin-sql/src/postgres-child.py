#!/usr/bin/env python3
import json, os, sys
import psycopg2

req = json.load(sys.stdin)
try:
    conn = psycopg2.connect(host=os.getenv('PGHOST', 'postgresql'), port=int(os.getenv('PGPORT', '5432')),
        user=os.getenv('PGUSER', 'root'), password=os.getenv('PGPASSWORD', '123123'),
        dbname=os.getenv('PGDATABASE') or req.get('database'), connect_timeout=10)
    conn.autocommit = False
    with conn.cursor() as cur:
        cur.execute(f"SET LOCAL statement_timeout = {int(req.get('timeoutMs', 60000))}")
        mode = req.get('mode', 'query')
        if mode == 'list':
            cur.execute("SELECT table_type, table_schema, table_name FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog','information_schema') ORDER BY table_schema, table_name")
            result = {'ok': True, 'columns': [d.name for d in cur.description], 'rows': cur.fetchall()}
        elif mode == 'schema':
            tables = req.get('tables') or []
            cur.execute("SELECT table_schema,table_name,column_name,data_type,is_nullable,column_default FROM information_schema.columns WHERE table_schema||'.'||table_name=ANY(%s) OR table_name=ANY(%s) ORDER BY table_schema,table_name,ordinal_position", (tables, tables))
            result = {'ok': True, 'columns': [d.name for d in cur.description], 'rows': cur.fetchall()}
        else:
            outputs = []
            for statement in req.get('statements') or [req.get('sql')]:
                cur.execute(statement)
                if cur.description:
                    cap = int(req.get('maxRows', 50)); rows = cur.fetchmany(cap + 1)
                    outputs.append({'columns': [d.name for d in cur.description], 'rows': rows[:cap], 'truncated': len(rows) > cap, 'status': cur.statusmessage})
                else:
                    outputs.append({'columns': [], 'rows': [], 'truncated': False, 'status': cur.statusmessage})
            result = {'ok': True, 'outputs': outputs}
    conn.rollback(); conn.close()
except Exception as exc:
    result = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
print(json.dumps(result, default=str))
