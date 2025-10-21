from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def transform_data(df):
    """Transform pandas dataframe using Spark"""
    spark = SparkSession.builder.appName("F1_Transform").getOrCreate()
    sdf = spark.createDataFrame(df)

    # Clean and format data
    sdf = sdf.withColumn("season", col("season").cast("int")) \
             .withColumn("round", col("round").cast("int")) \
             .withColumn("laps", col("laps").cast("int"))

    # Example: filter out null winners
    sdf = sdf.filter(col("winner").isNotNull())

    print("Transformation complete.")
    return sdf