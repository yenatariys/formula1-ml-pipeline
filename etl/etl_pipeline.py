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
    df = extract_data(data_dir="data/")  # extract pandas DF dari CSV
    
    # transform pake Spark
    sdf = transform_data(df)
    
    # convert Spark DF ke pandas DF sebelum load
    df_transformed = sdf.toPandas()
    
    # load ke Postgres dengan nama tabel hasil transformasi
    load_to_postgres(df_transformed, "f1_results_transformed")

    print("ETL pipeline finished successfully!")

# -----------------------------------------------------------
if __name__ == "__main__":
    main()