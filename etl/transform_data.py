from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def transform_data(df):
    """Transform pandas dataframe using Spark"""
    spark = SparkSession.builder.appName("F1_Transform").getOrCreate()
    sdf = spark.createDataFrame(df)

    # Clean and format data
    sdf = sdf.withColumn("year", col("year").cast("int")) \
             .withColumn("round", col("round").cast("int")) \
             .withColumn("position", col("position").cast("int")) \
             .withColumn("points", col("points").cast("float"))

    # Filter out null positions (DNF, DNS, etc.)
    sdf = sdf.filter(col("position").isNotNull())

    print("Transformation complete.")
    return sdf