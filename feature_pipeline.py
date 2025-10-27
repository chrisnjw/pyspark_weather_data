"""
Feature engineering pipeline for weather data
"""

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler


def create_feature_pipeline():
    """Create feature engineering pipeline matching preprocessing.ipynb"""

    feature_cols = [
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
    ]

    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="raw_features", handleInvalid="skip"
    )
    scaler = StandardScaler(
        inputCol="raw_features", outputCol="features", withStd=True, withMean=True
    )

    return Pipeline(stages=[assembler, scaler])
