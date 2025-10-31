from pyspark.sql import SparkSession


def main() -> None:
    spark = (
        SparkSession.builder.appName("Spark F1 Test")
        .master("spark://spark-master:7077")
        .getOrCreate()
    )

    print("Connected to Spark cluster successfully.")
    spark.stop()


if __name__ == "__main__":
    main()
