from sqlalchemy import create_engine

def load_data_to_postgres(sdf):
    """Load Spark DataFrame into PostgreSQL"""
    engine = create_engine("postgresql+psycopg2://admin:admin123@postgres:5432/f1_data")

    # Convert Spark DataFrame → Pandas for SQLAlchemy
    pdf = sdf.toPandas()
    pdf.to_sql("race_results", engine, if_exists="replace", index=False)
    print("Data successfully loaded into PostgreSQL.")