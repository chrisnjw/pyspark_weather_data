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
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit


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

    print("Training Gradient Boosting Trees")

    # Gradient Boosting Trees
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="temperature",
        maxIter=100,
        maxDepth=5,
        subsamplingRate=0.8,
    )

    # Cross-validation grid
    gbt_param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth, [6, 8])  # Moderate depth for 8 features
        .addGrid(gbt.maxIter, [30, 50])  # More iterations for stability
        .addGrid(gbt.stepSize, [0.05, 0.1, 0.2])
        # .addGrid(gbt.subsamplingRate, [0.8, 0.9])  # Prevent overfitting
        .build()
    )

    gbt_tvs = TrainValidationSplit(
        estimator=gbt,
        estimatorParamMaps=gbt_param_grid,
        evaluator=RegressionEvaluator(
            labelCol="temperature", predictionCol="prediction", metricName="rmse"
        ),
        trainRatio=0.8,  # 80% of data for training, 20% for validation
    )

    # CV on sample for hyperparameter tuning
    print("Running CV on sample (this may take a few minutes)...")
    gbt_model_cv = gbt_tvs.fit(train_features)

    # Extract best hyperparameters
    # best_idx = min(enumerate(gbt_model_cv.avgMetrics), key=lambda x: x[1])[0]
    param_map = gbt_model_cv.bestModel.extractParamMap()

    # Define subset of hyperparameters you care about
    subset_params = ["maxIter", "maxDepth", "stepSize", "minInstancesPerNode"]

    # Convert to plain Python dict
    best_params = {p.name: param_map[p] for p in param_map if p.name in subset_params}

    print(f"Best GBT hyperparameters: {best_params}")

    # Refit on full training data with best hyperparameters
    print("Refitting GBT on full training data with best hyperparameters...")
    final_gbt = GBTRegressor(
        featuresCol="features",
        labelCol="temperature",
    )
    # Apply best hyperparameters
    final_gbt = final_gbt.setParams(**best_params)

    gbt_model = final_gbt.fit(train_features)  # Train on ALL data
    gbt_predictions = gbt_model.transform(test_features)

    lr_evaluator = RegressionEvaluator(
        labelCol="temperature", predictionCol="prediction", metricName="rmse"
    )

    print("Training Random Forest...")

    # Random Forest
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="temperature",
        numTrees=10,
        minInstancesPerNode=10,
    )

    # Cross-validation grid
    rf_param_grid = (
        ParamGridBuilder()
        .addGrid(rf.maxDepth, [10, 20])  # Deeper trees for complex patterns
        .addGrid(rf.numTrees, [50, 100])  # More trees for stability
        .addGrid(rf.minInstancesPerNode, [10, 20])  # Pruning options
        .build()
    )

    rf_tvs = TrainValidationSplit(
        estimator=rf,
        estimatorParamMaps=rf_param_grid,
        evaluator=lr_evaluator,
        trainRatio=0.8,  # 80% of data for training, 20% for validation
    )

    # CV on sample for hyperparameter tuning
    print("Running CV (this may take a few minutes)...")
    rf_model_cv = rf_tvs.fit(train_features)

    # Extract best hyperparameters
    param_map = rf_model_cv.bestModel.extractParamMap()

    # Define subset of hyperparameters you care about
    subset_params = ["numTrees", "maxDepth", "minInstancesPerNode"]

    # Convert to plain Python dict
    best_params = {p.name: param_map[p] for p in param_map if p.name in subset_params}

    print(f"Best RF hyperparameters: {best_params}")

    # Refit on full training data with best hyperparameters
    print("Refitting Random Forest on full training data with best hyperparameters...")
    final_rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="temperature",
    )
    # Apply best hyperparameters
    final_rf = final_rf.setParams(**best_params)

    rf_model = final_rf.fit(train_features)  # Train on ALL data
    rf_predictions = rf_model.transform(test_features)

    # Save to output path in a folder with the current date and time
    output_path = f"{output_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Save models
    print("Saving models...")
    gbt_model.write().overwrite().save(f"{output_path}/models/gradient_boosting")
    rf_model.write().overwrite().save(f"{output_path}/models/random_forest")
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
