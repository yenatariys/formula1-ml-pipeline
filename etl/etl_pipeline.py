from extract_data import extract_data
from load_data import load_to_postgres
import time
import psycopg2

# tunggu Postgres siap
while True:
    try:
        conn = psycopg2.connect(
            dbname="f1_data",
            user="admin",
            password="admin123",
            host="f1_postgres",
            port=5432
        )
        conn.close()
        print("Postgres ready!")
        break
    except psycopg2.OperationalError:
        print("Waiting for Postgres...")
        time.sleep(2)

def main():
    df = extract_data(data_dir="data/")  # folder CSV
    load_to_postgres(df)

if __name__ == "__main__":
    main()