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


def three_phase_tuning(
    estimator, param_grid, train_features, train_ratio=0.8, top_n=3, parallelism=8
):
    """
    Three-phase hyperparameter tuning:
    1. Phase 1: 2% data - broad grid search, find top N candidates
    2. Phase 2: 10% data - narrow search on top candidates, find best
    3. Phase 3: 100% data - train final model with best hyperparameters

    Returns: Trained model on full dataset
    """
    evaluator = RegressionEvaluator(
        labelCol="temperature", predictionCol="prediction", metricName="rmse"
    )

    # Phase 1: 1% data - broad search
    print(f"  Phase 1: Grid search on 1% of data...")
    phase1_sample = train_features.sample(withReplacement=False, fraction=0.01, seed=42)
    phase1_sample.cache()

    phase1_tvs = TrainValidationSplit(
        estimator=estimator,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        trainRatio=train_ratio,
        parallelism=parallelism,
    )

    phase1_model = phase1_tvs.fit(phase1_sample)

    # Get top N candidates (lowest RMSE)
    param_map_list = phase1_model.getEstimatorParamMaps()
    metrics_list = (
        phase1_model.validationMetrics
    )  # TrainValidationSplit uses validationMetrics
    candidates = list(zip(param_map_list, metrics_list))
    candidates.sort(key=lambda x: x[1])  # Sort by RMSE (lower is better)
    top_candidates = candidates[:top_n]

    print(f"  Phase 1 complete. Top {top_n} candidates:")
    for i, (params, metric) in enumerate(top_candidates, 1):
        # Extract hyperparameters from ParamMap for display
        params_dict = {param.name: value for param, value in params.items()}
        params_str = ", ".join([f"{k}={v}" for k, v in sorted(params_dict.items())])
        print(f"    {i}. RMSE: {metric:.4f} | {params_str}")

    # Phase 2: 10% data - narrow search
    print(f"  Phase 2: Narrow search on 10% of data with top {top_n} candidates...")
    phase2_sample = train_features.sample(withReplacement=False, fraction=0.1, seed=42)
    phase2_sample.cache()

    # Create new grid with only top candidates
    phase2_param_grid = [params for params, _ in top_candidates]

    phase2_tvs = TrainValidationSplit(
        estimator=estimator,
        estimatorParamMaps=phase2_param_grid,
        evaluator=evaluator,
        trainRatio=train_ratio,
        parallelism=parallelism,
    )

    phase2_model = phase2_tvs.fit(phase2_sample)
    best_idx = min(enumerate(phase2_model.validationMetrics), key=lambda x: x[1])[0]
    best_params = phase2_model.getEstimatorParamMaps()[best_idx]
    best_metric = phase2_model.validationMetrics[best_idx]

    print(f"  Phase 2 complete. Best RMSE: {best_metric:.4f}")

    # Extract best hyperparameters for display
    param_map = phase2_model.bestModel.extractParamMap()
    subset_params = [
        "maxIter",
        "maxDepth",
        "stepSize",
        "minInstancesPerNode",
        "numTrees",
        "subsamplingRate",
        "maxBins",
        "featureSubsetStrategy",
    ]
    best_params_dict = {
        p.name: param_map[p] for p in param_map if p.name in subset_params
    }
    print(f"  Best hyperparameters: {best_params_dict}")

    # Phase 3: 100% data - final training
    print(f"  Phase 3: Training final model on full dataset...")
    final_estimator = estimator.__class__(
        featuresCol=estimator.getFeaturesCol(),
        labelCol=estimator.getLabelCol(),
    )

    # Apply best hyperparameters from ParamMap
    # Convert ParamMap (Param objects as keys) to dict with string keys
    param_dict = {param.name: value for param, value in best_params.items()}
    final_estimator = final_estimator.setParams(**param_dict)

    final_model = final_estimator.fit(train_features)
    print(f"  Phase 3 complete. Final model trained on full dataset.")

    return final_model


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

    print("Training Gradient Boosting Trees with 3-phase hyperparameter tuning")

    # Gradient Boosting Trees
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="temperature",
        maxIter=100,
        maxDepth=5,
        subsamplingRate=0.8,
    )

    gbt_param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth, [3, 5, 8])
        .addGrid(gbt.maxIter, [20, 50])
        .addGrid(gbt.stepSize, [0.05, 0.1])
        .addGrid(gbt.maxBins, [32, 64])
        .build()
    )

    # Use 3-phase tuning
    print("Starting 3-phase hyperparameter tuning for GBT...")
    gbt_model = three_phase_tuning(
        estimator=gbt,
        param_grid=gbt_param_grid,
        train_features=train_features,
        train_ratio=0.8,
        top_n=5,  # Keep top 5 candidates from Phase 1
        parallelism=8,
    )

    gbt_predictions = gbt_model.transform(test_features)

    lr_evaluator = RegressionEvaluator(
        labelCol="temperature", predictionCol="prediction", metricName="rmse"
    )

    print("Training Random Forest with 3-phase hyperparameter tuning")

    # Random Forest
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="temperature",
        numTrees=100,
        minInstancesPerNode=10,
    )

    rf_param_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [50, 100])
        .addGrid(rf.maxDepth, [6, 10, 14])
        .addGrid(rf.maxBins, [32, 64])
        .addGrid(rf.subsamplingRate, [0.8, 1.0])
        .addGrid(rf.featureSubsetStrategy, ["auto", "sqrt"])
        .build()
    )

    # Use 3-phase tuning
    print("Starting 3-phase hyperparameter tuning for RF...")
    rf_model = three_phase_tuning(
        estimator=rf,
        param_grid=rf_param_grid,
        train_features=train_features,
        train_ratio=0.8,
        top_n=5,  # Keep top 5 candidates from Phase 1
        parallelism=8,
    )

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
