"""
Data loading and preprocessing functions for weather data
"""

from pyspark.sql.functions import (
    col,
    split,
    substring,
    when,
    hour,
    dayofyear,
    sin,
    cos,
    toRadians,
)
from pyspark.sql.types import *


def clean_weather_column(
    input_df,
    col_name,
    missing_code,
    quality_flags,
    scale_factor,
    handle_signs=False,
):
    """Top-level helper to clean NOAA value,flag columns.
    Defined at module scope to avoid capturing in Spark task closures.
    """
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

    if handle_signs:
        df_good = df_good.withColumn(
            f"{col_name}_signed_value",
            when(
                col(f"{col_name}_value").startswith("+"),
                substring(col(f"{col_name}_value"), 2, 100),
            )
            .when(col(f"{col_name}_value").startswith("-"), col(f"{col_name}_value"))
            .otherwise(col(f"{col_name}_value")),
        )
        clean_col_name = col_name.lower() + "_clean"
        df_final = df_good.withColumn(
            clean_col_name,
            col(f"{col_name}_signed_value").cast(DoubleType()) / scale_factor,
        )
        df_final = df_final.drop(
            col_name,
            f"{col_name}_parts",
            f"{col_name}_value",
            f"{col_name}_flag",
            f"{col_name}_signed_value",
        )
    else:
        clean_col_name = col_name.lower() + "_clean"
        df_final = df_good.withColumn(
            clean_col_name, col(f"{col_name}_value").cast(DoubleType()) / scale_factor
        )
        df_final = df_final.drop(
            col_name, f"{col_name}_parts", f"{col_name}_value", f"{col_name}_flag"
        )
    return df_final


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

    # Records count is slow, so we don't need to print it
    # print(f"Loaded {df.count()} records")

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

    # 2) Time features from DATE with cyclic encoding
    df_tmp = (
        df_tmp.withColumn("timestamp", col("DATE").cast(TimestampType()))
        .withColumn("hour_raw", hour(col("timestamp")))
        .withColumn("day_of_year_raw", dayofyear(col("timestamp")))
    )

    # Apply cyclic encoding for hour (0-23) and day_of_year (1-365)
    # For hour: period = 24
    df_tmp = df_tmp.withColumn(
        "hour_sin", sin(col("hour_raw") * 2 * 3.14159265359 / 24)
    )
    df_tmp = df_tmp.withColumn(
        "hour_cos", cos(col("hour_raw") * 2 * 3.14159265359 / 24)
    )

    # For day_of_year: period = 365
    df_tmp = df_tmp.withColumn(
        "day_of_year_sin", sin(col("day_of_year_raw") * 2 * 3.14159265359 / 365)
    )
    df_tmp = df_tmp.withColumn(
        "day_of_year_cos", cos(col("day_of_year_raw") * 2 * 3.14159265359 / 365)
    )

    # 3) Geographic features - Spherical 3D Embedding
    df_tmp = (
        df_tmp.withColumn("latitude", col("LATITUDE").cast(DoubleType()))
        .withColumn("longitude", col("LONGITUDE").cast(DoubleType()))
        .withColumn("elevation", col("ELEVATION").cast(DoubleType()))
    )

    # Convert to radians
    df_tmp = df_tmp.withColumn("lat_rad", toRadians(col("latitude")))
    df_tmp = df_tmp.withColumn("lon_rad", toRadians(col("longitude")))

    # Convert to cartesian coordinates (spherical 3D embedding)
    df_tmp = df_tmp.withColumn("geo_x", cos(col("lat_rad")) * cos(col("lon_rad")))
    df_tmp = df_tmp.withColumn("geo_y", cos(col("lat_rad")) * sin(col("lon_rad")))
    df_tmp = df_tmp.withColumn("geo_z", sin(col("lat_rad")))

    # Drop intermediate columns
    df_tmp = df_tmp.drop(
        "lat_rad", "lon_rad", "latitude", "longitude", "hour_raw", "day_of_year_raw"
    )

    # Helper is now at module scope (see top of file)

    # 4) Clean DEW with sign handling
    df_feat = clean_weather_column(
        df_tmp, "DEW", "+9999", ["1", "5"], 10.0, handle_signs=True
    )

    # 5) Clean SLP
    df_feat = clean_weather_column(df_feat, "SLP", "99999", ["1", "5"], 10.0)

    # 6) Parse WND (wind speed) - 4th value
    df_wnd = df_feat.where(col("WND").isNotNull() & col("WND").contains(","))
    df_wnd = (
        df_wnd.withColumn("wnd_parts", split(col("WND"), ","))
        .withColumn("wnd_value", col("wnd_parts")[3])  # 4th value (0-indexed)
        .withColumn("wnd_flag", col("wnd_parts")[4])  # 5th value (0-indexed)
    )
    df_wnd = df_wnd.where(
        (col("wnd_value") != "9999") & (col("wnd_flag").isin(["1", "5"]))
    )
    df_feat = df_wnd.withColumn(
        "wind_speed", col("wnd_value").cast(DoubleType()) / 10.0
    )
    df_feat = df_feat.drop("wnd_parts", "wnd_value", "wnd_flag", "WND")

    # 7) Parse CIG (cloud ceiling height) - 1st value
    df_cig = df_feat.where(col("CIG").isNotNull() & col("CIG").contains(","))
    df_cig = (
        df_cig.withColumn("cig_parts", split(col("CIG"), ","))
        .withColumn("cig_value", col("cig_parts")[0])  # 1st value
        .withColumn("cig_flag", col("cig_parts")[1])  # 2nd value
    )
    df_cig = df_cig.where(
        (col("cig_value") != "99999") & (col("cig_flag").isin(["1", "5"]))
    )
    df_feat = df_cig.withColumn("cloud_ceiling", col("cig_value").cast(DoubleType()))
    df_feat = df_feat.drop("cig_parts", "cig_value", "cig_flag", "CIG")

    # 8) Parse VIS (horizontal visibility) - 1st value
    df_vis = df_feat.where(col("VIS").isNotNull() & col("VIS").contains(","))
    df_vis = (
        df_vis.withColumn("vis_parts", split(col("VIS"), ","))
        .withColumn("vis_value", col("vis_parts")[0])  # 1st value
        .withColumn("vis_flag", col("vis_parts")[1])  # 2nd value
    )
    df_vis = df_vis.where(
        (col("vis_value") != "999999") & (col("vis_flag").isin(["1", "5"]))
    )
    df_feat = df_vis.withColumn("visibility", col("vis_value").cast(DoubleType()))
    df_feat = df_feat.drop("vis_parts", "vis_value", "vis_flag", "VIS")

    # 9) Final filter and bounds
    df_feat = df_feat.filter(col("temperature").isNotNull())
    df_feat = df_feat.filter((col("temperature") >= -100) & (col("temperature") <= 60))
    # elevation sentinel
    df_feat = df_feat.where(col("elevation") != -999.9)

    # print(f"After preprocessing: {df_feat.count()} records")

    # Show sample
    print("Sample of processed features:")
    df_feat.select(
        "temperature",
        "dew_clean",
        "slp_clean",
        "wind_speed",
        "cloud_ceiling",
        "visibility",
        "geo_x",
        "geo_y",
        "geo_z",
        "elevation",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ).show(5)

    # print("\nFeature Statistics:")
    # df_feat.describe().show()

    return df_feat
