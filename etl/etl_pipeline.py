# etl_pipeline.py
from extract_data import extract_data
from transform_data import transform_data
import time
import psycopg2

# -----------------------------------------------------------
# Tunggu Postgres siap
# -----------------------------------------------------------
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
        print("Postgres is ready!")
        break
    except psycopg2.OperationalError:
        print("Waiting for Postgres...")
        time.sleep(2)

# -----------------------------------------------------------
# Main ETL function
# -----------------------------------------------------------
def main():
    print("Starting ETL pipeline...")

    # 1️⃣ Extract
    df = extract_data(data_dir="data/")  # pandas DataFrame dari CSV
    print(f"Extracted {len(df)} records")

    # 2️⃣ Transform (Spark)
    sdf = transform_data(df)  # Spark DataFrame, sekaligus tersimpan di Postgres
    print("Transformation complete and saved to Postgres")

    # 3️⃣ (Opsional) jika mau cek hasil di pandas
    # df_transformed = sdf.toPandas()
    # print(df_transformed.head())

    print("ETL pipeline finished successfully!")

# -----------------------------------------------------------
if __name__ == "__main__":
    main()