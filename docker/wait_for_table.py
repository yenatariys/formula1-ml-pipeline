import time
from sqlalchemy import create_engine, text
import sys

DB_URL = "postgresql+psycopg2://admin:admin123@f1_postgres:5432/f1_data"
engine = create_engine(DB_URL, pool_pre_ping=True)

timeout = 120  # detik
interval = 3
deadline = time.time() + timeout

while time.time() < deadline:
    try:
        with engine.connect() as conn:
            exists = conn.execute(text(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema='public' AND table_name='f1_results')"
            )).scalar()
            if exists:
                print("f1_results found, continuing.")
                sys.exit(0)
    except Exception as e:
        print("DB not ready:", e)
    print("Waiting for f1_results table...")
    time.sleep(interval)

print("Timed out waiting for f1_results table.", file=sys.stderr)
sys.exit(1)