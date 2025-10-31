#!/usr/bin/env python3
"""
Weather Data Machine Learning Pipeline
Predicts air temperature (TMP) using other weather variables

Main entry point for the ML training pipeline
"""

import argparse
from pyspark.sql import SparkSession
from data_loader import load_and_preprocess_data
from model_trainer import train_models


def create_spark_session():
    """Create Spark session with appropriate configuration"""
    return (
        SparkSession.builder.appName("WeatherML")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )


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

        # print(f"Training set: {train_df.count()} records")
        # print(f"Test set: {test_df.count()} records")

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
