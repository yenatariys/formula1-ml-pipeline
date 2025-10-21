from extract_data import extract_data
from transform_data import transform_data
from load_data import load_data_to_postgres

def main():
    print("Starting ETL pipeline...")
    df = extract_data(years=3)  # last 3 seasons
    sdf = transform_data(df)
    load_data_to_postgres(sdf)
    print("ETL process complete.")

if __name__ == "__main__":
    main()