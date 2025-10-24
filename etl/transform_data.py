from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def transform_data(df):
    """Transform pandas dataframe using Spark  and save to Postgres"""

    ### Connect to Spark master di docker compose
    spark = SparkSession.builder \
        .appName("F1_Transform") \
        .master("spark://spark-master:7077") \
        .getOrCreate()
    
    ### Convert pandas DataFrame ke Spark DataFrame
    sdf = spark.createDataFrame(df)

    spark = SparkSession.builder.appName("F1_Transform").getOrCreate()
    sdf = spark.createDataFrame(df)

    # Clean and format data
    sdf = sdf.withColumn("year", col("year").cast("int")) \
             .withColumn("round", col("round").cast("int")) \
             .withColumn("position", col("position").cast("int")) \
             .withColumn("points", col("points").cast("float"))

    # Filter out null positions (DNF, DNS, etc.)
    sdf = sdf.filter(col("position").isNotNull())

    # Simpan hasil transformasi ke Postgres
    sdf.write \
        .format("jdbc") \
        .option("url", "jdbc:postgresql://f1_postgres:5432/f1_data") \
        .option("dbtable", "f1_results_tranformed") \
        .option("user", "admin") \
        .option("password", "admin123") \
        .mode("overwrite") \
        .save()

    print("Transformation complete and saved to Postgres.")
    return sdf