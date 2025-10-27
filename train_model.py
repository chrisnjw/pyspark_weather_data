#!/usr/bin/env python3
"""
Weather Data Machine Learning Pipeline
Predicts air temperature (TMP) using other weather variables
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    split,
    substring,
    when,
    hour,
    month,
    dayofyear,
    avg,
    sum as spark_sum,
    pow as spark_pow,
    abs as spark_abs,
)
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.param.shared import HasInputCol, HasOutputCol


def create_spark_session():
    """Create Spark session with appropriate configuration"""
    return (
        SparkSession.builder.appName("WeatherML")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )


def parse_weather_value(value_str, scale_factor=10):
    """
    Parse weather values from NOAA format
    Format: "value,quality,measurement,time" where value is scaled by 10
    """
    if not value_str or value_str == "99999,9,9,9" or value_str == "999999,9,9,9":
        return None

    try:
        parts = value_str.split(",")
        if len(parts) >= 1 and parts[0] != "99999" and parts[0] != "999999":
            value = float(parts[0]) / scale_factor
            # Check quality flag (second part)
            if len(parts) > 1 and parts[1] in [
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
            ]:
                return value
    except (ValueError, IndexError):
        pass
    return None


def parse_wind_value(wind_str):
    """Parse wind data: direction,speed_quality,speed,time"""
    if not wind_str or wind_str == "999,9,9999,9":
        return None, None

    try:
        parts = wind_str.split(",")
        if len(parts) >= 3 and parts[0] != "999" and parts[2] != "9999":
            direction = float(parts[0])
            speed = float(parts[2]) / 10.0  # Convert from 0.1 m/s to m/s
            return direction, speed
    except (ValueError, IndexError):
        pass
    return None, None


def load_and_preprocess_data(spark, data_path):
    """Load weather data and perform preprocessing using notebook pipeline"""
    print("Loading weather data from:", data_path)

    # Define schema for weather data
    schema = StructType(
        [
            StructField("STATION", StringType(), True),
            StructField("DATE", StringType(), True),
            StructField("SOURCE", StringType(), True),
            StructField("LATITUDE", DoubleType(), True),
            StructField("LONGITUDE", DoubleType(), True),
            StructField("ELEVATION", DoubleType(), True),
            StructField("NAME", StringType(), True),
            StructField("REPORT_TYPE", StringType(), True),
            StructField("CALL_SIGN", StringType(), True),
            StructField("QUALITY_CONTROL", StringType(), True),
            StructField("WND", StringType(), True),
            StructField("CIG", StringType(), True),
            StructField("VIS", StringType(), True),
            StructField("TMP", StringType(), True),
            StructField("DEW", StringType(), True),
            StructField("SLP", StringType(), True),
            StructField("AA1", StringType(), True),
            StructField("AA2", StringType(), True),
            StructField("AA3", StringType(), True),
            StructField("AJ1", StringType(), True),
            StructField("AY1", StringType(), True),
            StructField("AY2", StringType(), True),
            StructField("GA1", StringType(), True),
            StructField("GA2", StringType(), True),
            StructField("GA3", StringType(), True),
            StructField("GE1", StringType(), True),
            StructField("GF1", StringType(), True),
            StructField("IA1", StringType(), True),
            StructField("KA1", StringType(), True),
            StructField("KA2", StringType(), True),
            StructField("MA1", StringType(), True),
            StructField("MD1", StringType(), True),
            StructField("MW1", StringType(), True),
            StructField("OC1", StringType(), True),
            StructField("OD1", StringType(), True),
            StructField("SA1", StringType(), True),
            StructField("UA1", StringType(), True),
            StructField("REM", StringType(), True),
            StructField("EQD", StringType(), True),
        ]
    )

    # Load data from directory of CSV files
    print("Loading data from directory...")
    df = spark.read.option("header", "true").schema(schema).csv(data_path)

    print(f"Loaded {df.count()} records")

    # ================= Preprocessing =================
    # 1) Clean TMP -> temperature
    df_with_comma = df.where(col("TMP").isNotNull() & col("TMP").contains(","))
    df_tmp = (
        df_with_comma.withColumn("tmp_parts", split(col("TMP"), ","))
        .withColumn("tmp_value", col("tmp_parts")[0])
        .withColumn("tmp_flag", col("tmp_parts")[1])
    )
    df_tmp = df_tmp.where(
        (col("tmp_value") != "+9999") & (col("tmp_flag").isin(["1", "5"]))
    )
    df_tmp = df_tmp.withColumn(
        "temperature", col("tmp_value").cast(DoubleType()) / 10.0
    )

    # 2) Time features from DATE
    df_tmp = (
        df_tmp.withColumn("timestamp", col("DATE").cast(TimestampType()))
        .withColumn("hour", hour(col("timestamp")))
        .withColumn("month", month(col("timestamp")))
        .withColumn("day_of_year", dayofyear(col("timestamp")))
    )

    # 3) Cast geographic
    df_tmp = (
        df_tmp.withColumn("latitude", col("LATITUDE").cast(DoubleType()))
        .withColumn("longitude", col("LONGITUDE").cast(DoubleType()))
        .withColumn("elevation", col("ELEVATION").cast(DoubleType()))
    )

    # Helper to clean NOAA value,flag columns (like in notebook)
    def clean_weather_column(
        input_df, col_name, missing_code, quality_flags, scale_factor
    ):
        df_with_c = input_df.where(col(col_name).contains(","))
        df_p = (
            df_with_c.withColumn(f"{col_name}_parts", split(col(col_name), ","))
            .withColumn(f"{col_name}_value", col(f"{col_name}_parts")[0])
            .withColumn(f"{col_name}_flag", col(f"{col_name}_parts")[1])
        )
        df_good = df_p.where(
            (col(f"{col_name}_value") != missing_code)
            & (col(f"{col_name}_flag").isin(quality_flags))
        )
        clean_col_name = col_name.lower() + "_clean"
        df_final = df_good.withColumn(
            clean_col_name, col(f"{col_name}_value").cast(DoubleType()) / scale_factor
        )
        df_final = df_final.drop(
            col_name, f"{col_name}_parts", f"{col_name}_value", f"{col_name}_flag"
        )
        return df_final

    # 4) Clean DEW and SLP like notebook
    df_feat = clean_weather_column(df_tmp, "DEW", "+9999", ["1", "5"], 10.0)
    df_feat = clean_weather_column(df_feat, "SLP", "99999", ["1", "5"], 10.0)

    # 5) Final filter and bounds
    df_feat = df_feat.filter(col("temperature").isNotNull())
    df_feat = df_feat.filter((col("temperature") >= -100) & (col("temperature") <= 60))
    # elevation sentinel
    df_feat = df_feat.where(col("elevation") != -999.9)

    print(f"After preprocessing: {df_feat.count()} records")

    # Show sample
    print("Sample of processed features:")
    df_feat.select(
        "temperature",
        "dew_clean",
        "slp_clean",
        "latitude",
        "longitude",
        "elevation",
        "hour",
        "month",
        "day_of_year",
    ).show(5)

    return df_feat


def create_feature_pipeline():
    """Create feature engineering pipeline matching preprocessing.ipynb"""

    feature_cols = [
        "latitude",
        "longitude",
        "elevation",
        "dew_clean",
        "slp_clean",
        "hour",
        "month",
        "day_of_year",
    ]

    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="raw_features", handleInvalid="skip"
    )
    scaler = StandardScaler(
        inputCol="raw_features", outputCol="features", withStd=True, withMean=True
    )

    return Pipeline(stages=[assembler, scaler])


def train_models(train_df, test_df, output_path, spark):
    """Train and evaluate ML models"""

    print("Creating feature pipeline...")
    feature_pipeline = create_feature_pipeline()

    # Check data before fitting
    print(f"Training data count: {train_df.count()}")
    print(f"Test data count: {test_df.count()}")

    # Show sample of training data
    print("Sample of training data features:")
    train_df.select(
        "latitude",
        "longitude",
        "elevation",
        "dew_clean",
        "slp_clean",
        "hour",
        "month",
        "day_of_year",
    ).show(5)

    # Fit feature pipeline
    print("Fitting feature pipeline...")
    feature_model = feature_pipeline.fit(train_df)
    print("Transforming training data...")
    train_features = feature_model.transform(train_df)
    print("Transforming test data...")
    test_features = feature_model.transform(test_df)

    # Cache for performance
    train_features.cache()
    test_features.cache()

    print("Training Linear Regression...")

    # Linear Regression
    lr = LinearRegression(
        featuresCol="features",
        labelCol="temperature",
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.8,
    )

    # Cross-validation for Linear Regression (simplified for testing)
    lr_param_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.01, 0.1])  # Reduced from 3 to 2 values
        .addGrid(lr.elasticNetParam, [0.0, 1.0])  # Reduced from 3 to 2 values
        .build()
    )

    lr_evaluator = RegressionEvaluator(
        labelCol="temperature", predictionCol="prediction", metricName="rmse"
    )

    lr_cv = CrossValidator(
        estimator=lr,
        estimatorParamMaps=lr_param_grid,
        evaluator=lr_evaluator,
        numFolds=2,  # Reduced from 3 to 2 folds
    )

    lr_model = lr_cv.fit(train_features)
    lr_predictions = lr_model.transform(test_features)

    print("Training Decision Tree...")

    # Decision Tree
    dt = DecisionTreeRegressor(
        featuresCol="features",
        labelCol="temperature",
        maxDepth=10,
        minInstancesPerNode=5,
    )

    # Cross-validation for Decision Tree (simplified for testing)
    dt_param_grid = (
        ParamGridBuilder()
        .addGrid(dt.maxDepth, [5, 10])  # Reduced from 3 to 2 values
        .addGrid(dt.minInstancesPerNode, [5, 10])  # Reduced from 3 to 2 values
        .build()
    )

    dt_cv = CrossValidator(
        estimator=dt,
        estimatorParamMaps=dt_param_grid,
        evaluator=lr_evaluator,
        numFolds=2,  # Reduced from 3 to 2 folds
    )

    dt_model = dt_cv.fit(train_features)
    dt_predictions = dt_model.transform(test_features)

    # Save models
    print("Saving models...")
    lr_model.bestModel.write().overwrite().save(
        f"{output_path}/models/linear_regression"
    )
    dt_model.bestModel.write().overwrite().save(f"{output_path}/models/decision_tree")
    feature_model.write().overwrite().save(f"{output_path}/models/feature_pipeline")

    # Evaluate models
    print("Evaluating models...")

    def evaluate_model(predictions, model_name, spark):
        rmse = lr_evaluator.evaluate(predictions)

        # Calculate R² using PySpark functions
        label_col = "temperature"  # The actual label column name
        pred_col = "prediction"

        # Calculate mean of actual values
        mean_actual = predictions.select(avg(label_col)).collect()[0][0]

        # Calculate R²: 1 - (SS_res / SS_tot)
        ss_res = predictions.select(
            spark_sum(spark_pow(col(label_col) - col(pred_col), 2))
        ).collect()[0][0]
        ss_tot = predictions.select(
            spark_sum(spark_pow(col(label_col) - mean_actual, 2))
        ).collect()[0][0]

        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # Calculate MAE
        mae = predictions.select(
            avg(spark_abs(col(label_col) - col(pred_col)))
        ).collect()[0][0]

        return {
            "model": model_name,
            "rmse": float(rmse),
            "r2": float(r2) if r2 is not None else 0.0,
            "mae": float(mae) if mae is not None else 0.0,
        }

    lr_metrics = evaluate_model(lr_predictions, "Linear Regression", spark)
    dt_metrics = evaluate_model(dt_predictions, "Decision Tree", spark)

    # Save metrics
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "models": [lr_metrics, dt_metrics],
        "best_model": min([lr_metrics, dt_metrics], key=lambda x: x["rmse"])["model"],
    }

    # Save metrics to GCS
    metrics_json = json.dumps(metrics, indent=2)
    spark.sparkContext.parallelize([metrics_json]).coalesce(1).saveAsTextFile(
        f"{output_path}/metrics"
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Weather ML Training Pipeline")
    parser.add_argument("--data-path", required=True, help="GCS path to weather data")
    parser.add_argument("--output-path", required=True, help="GCS path for output")
    parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Training data ratio"
    )

    args = parser.parse_args()

    # Create Spark session
    spark = create_spark_session()

    try:
        # Load and preprocess data
        df = load_and_preprocess_data(spark, args.data_path)

        # Split data
        train_df, test_df = df.randomSplit(
            [args.train_ratio, 1 - args.train_ratio], seed=42
        )

        print(f"Training set: {train_df.count()} records")
        print(f"Test set: {test_df.count()} records")

        # Train models
        metrics = train_models(train_df, test_df, args.output_path, spark)

        # Print results
        print("\n=== Model Performance ===")
        for model_metrics in metrics["models"]:
            print(f"\n{model_metrics['model']}:")
            print(f"  RMSE: {model_metrics['rmse']:.4f}")
            print(f"  R²:   {model_metrics['r2']:.4f}")
            print(f"  MAE:  {model_metrics['mae']:.4f}")

        print(f"\nBest model: {metrics['best_model']}")
        print(f"Results saved to: {args.output_path}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
