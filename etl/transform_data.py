from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def transform_data(df):
    """Transform pandas dataframe using Spark and save to Postgres"""

    ### Use Spark in local mode (no need for external Spark cluster)
    spark = SparkSession.builder \
        .appName("F1_Transform") \
        .master("local[*]") \
        .getOrCreate()
    
    ### Convert pandas DataFrame ke Spark DataFrame
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