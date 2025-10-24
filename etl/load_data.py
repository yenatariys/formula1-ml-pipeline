from sqlalchemy import create_engine

def load_to_postgres(df, table_name="f1_results_transformed"):
    engine = create_engine("postgresql+psycopg2://admin:admin123@f1_postgres:5432/f1_data")
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print(f"Data loaded to table {table_name}")