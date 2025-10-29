"""
Model training and evaluation functions
"""

import json
from datetime import datetime
from pyspark.sql.functions import (
    col,
    avg,
    sum as spark_sum,
    pow as spark_pow,
    abs as spark_abs,
)
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


def evaluate_model(predictions, model_name, spark):
    """Evaluate model performance metrics"""
    evaluator = RegressionEvaluator(
        labelCol="temperature", predictionCol="prediction", metricName="rmse"
    )
    rmse = evaluator.evaluate(predictions)

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
    mae = predictions.select(avg(spark_abs(col(label_col) - col(pred_col)))).collect()[
        0
    ][0]

    return {
        "model": model_name,
        "rmse": float(rmse),
        "r2": float(r2) if r2 is not None else 0.0,
        "mae": float(mae) if mae is not None else 0.0,
    }


def train_models(train_df, test_df, output_path, spark):
    """Train and evaluate ML models"""

    from feature_pipeline import create_feature_pipeline

    print("Creating feature pipeline...")
    feature_pipeline = create_feature_pipeline()

    # count() is slow, so we don't need to print it
    # # Check data before fitting
    # print(f"Training data count: {train_df.count()}")
    # print(f"Test data count: {test_df.count()}")

    # Show sample of training data
    print("Sample of training data features:")
    train_df.select(
        "geo_x",
        "geo_y",
        "geo_z",
        "elevation",
        "dew_clean",
        "slp_clean",
        "wind_speed",
        "cloud_ceiling",
        "visibility",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
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

    print("Training Gradient Boosting Trees (similar to LightGBM)...")

    # Gradient Boosting Trees - similar to LightGBM, available in PySpark
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="temperature",
        maxIter=50,
        maxDepth=5,
    )

    # Cross-validation for Gradient Boosting Trees
    gbt_param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth, [3, 5])
        .addGrid(gbt.maxIter, [30, 50])
        .build()
    )

    gbt_cv = CrossValidator(
        estimator=gbt,
        estimatorParamMaps=gbt_param_grid,
        evaluator=RegressionEvaluator(
            labelCol="temperature", predictionCol="prediction", metricName="rmse"
        ),
        numFolds=2,
    )

    gbt_model = gbt_cv.fit(train_features)
    gbt_predictions = gbt_model.transform(test_features)

    lr_evaluator = RegressionEvaluator(
        labelCol="temperature", predictionCol="prediction", metricName="rmse"
    )

    print("Training Random Forest...")

    # Random Forest
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="temperature",
        maxDepth=10,
        minInstancesPerNode=5,
        numTrees=20,
    )

    # Cross-validation for Random Forest (simplified for testing)
    rf_param_grid = (
        ParamGridBuilder()
        .addGrid(rf.maxDepth, [5, 10])  # Reduced from 3 to 2 values
        .addGrid(rf.numTrees, [10, 20])  # Number of trees in the forest
        .build()
    )

    rf_cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=rf_param_grid,
        evaluator=lr_evaluator,
        numFolds=2,  # Reduced from 3 to 2 folds
    )

    rf_model = rf_cv.fit(train_features)
    rf_predictions = rf_model.transform(test_features)

    # Save to output path in a folder with the current date and time
    output_path = f"{output_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Save models
    print("Saving models...")
    gbt_model.bestModel.write().overwrite().save(
        f"{output_path}/models/gradient_boosting"
    )
    rf_model.bestModel.write().overwrite().save(f"{output_path}/models/random_forest")
    feature_model.write().overwrite().save(f"{output_path}/models/feature_pipeline")

    # Evaluate models
    print("Evaluating models...")

    gbt_metrics = evaluate_model(gbt_predictions, "Gradient Boosting Trees", spark)
    rf_metrics = evaluate_model(rf_predictions, "Random Forest", spark)

    # Save metrics
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "models": [gbt_metrics, rf_metrics],
        "best_model": min([gbt_metrics, rf_metrics], key=lambda x: x["rmse"])["model"],
    }

    # Save metrics to GCS
    metrics_json = json.dumps(metrics, indent=2)
    spark.sparkContext.parallelize([metrics_json]).coalesce(1).saveAsTextFile(
        f"{output_path}/metrics"
    )

    return metrics
