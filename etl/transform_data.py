from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def transform_data(df):
    """Transform pandas dataframe using Spark and save to Postgres"""

    ### Use Spark cluster (fallback to local if spark-master unavailable)
    spark = SparkSession.builder \
        .appName("F1_Transform") \
        .master("local[*]") \
        .config("spark.driver.host", "localhost") \
        .config("spark.jars.packages", "org.postgresql:postgresql:42.6.0") \
        .getOrCreate()
    
    ### Convert pandas DataFrame to Spark DataFrame
    sdf = spark.createDataFrame(df)

    # Clean and format data
    sdf = sdf.withColumn("year", col("year").cast("int")) \
             .withColumn("round", col("round").cast("int")) \
             .withColumn("position", col("position").cast("int")) \
             .withColumn("points", col("points").cast("float"))

    # Filter out null positions (DNF, DNS, etc.)
    sdf = sdf.filter(col("position").isNotNull())

    # Save to PostgreSQL
    sdf.write \
        .format("jdbc") \
        .option("url", "jdbc:postgresql://f1_postgres:5432/f1_data") \
        .option("dbtable", "f1_results") \
        .option("user", "admin") \
        .option("password", "admin123") \
        .option("driver", "org.postgresql.Driver") \
        .mode("overwrite") \
        .save()

    print("Transformation complete and saved to PostgreSQL.")
    return sdf