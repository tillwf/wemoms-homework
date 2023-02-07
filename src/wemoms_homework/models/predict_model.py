import click
import fastparquet
import functools as ft
import json
import logging
import os
import pandas as pd

from tensorflow.keras.models import load_model

from wemoms_homework.features import USER_FEATURES
from wemoms_homework.features import POST_FEATURES
from wemoms_homework.features import USER_POST_FEATURES

from wemoms_homework.config import load_config
from wemoms_homework.utils import load_data
from wemoms_homework.utils import load_datasets
from wemoms_homework.utils import load_features

from wemoms_homework.features.base_features import BaseFeatures
from wemoms_homework.features.extra_features import ExtraFeatures
from wemoms_homework.features.post_popularity import PostPopularity
from wemoms_homework.features.user_post_popularity import UserPostPopularity

CONF = load_config()
DATA_PATH = CONF["path"]["input_data_path"]
MODELS_ROOT = CONF["path"]["models_root"]
LOGS_ROOT = CONF["path"]["logs_root"]
INTERIM_ROOT = CONF["path"]["interim_data_root"]
OUTPUT_ROOT = CONF["path"]["output_data_root"]

FEATURE_DICT = {
    "base_features": BaseFeatures,
    "extra_features": ExtraFeatures,
    "post_popularity": PostPopularity,
    "user_post_popularity": UserPostPopularity
}

FEATURE_DEFINITIONS = CONF["feature_definitions"]
FEATURES = CONF["features"]


@click.group()
def predict():
    pass


@predict.command()
@click.option(
    '--testset-path',
    type=str,
    default=os.path.join(OUTPUT_ROOT, "test.csv"),
    help='Path of test dataset, default is {}'.format(
        os.path.join(OUTPUT_ROOT, "test.csv")
    )
)
@click.option(
    '--models-root',
    type=str,
    default=MODELS_ROOT,
    help='Path of models folder, default is {}'.format(
        MODELS_ROOT
    )
)
@click.option(
    '--output-root',
    type=str,
    default=OUTPUT_ROOT,
    help='Path of output folder, default is {}'.format(
        OUTPUT_ROOT
    )
)
@click.option(
    '--features',
    type=str,
    multiple=True,
    default=list(FEATURE_DICT.keys()),
    help='Features used for the training, default is {}'.format(
        list(FEATURE_DICT.keys())
    )
)
def make_predictions(testset_path, models_root, output_root, features, evaluate=True):
    logging.info("Make Prediction")

    logging.info("Reading test data")
    _, _, X_test = load_datasets()
    y_test = X_test.pop("has_been_opened")


    logging.info("Create features")
    data = load_features()

    # Select the feature to use based on the `features` param
    features_names = [f
        for feature_group in features
        for f in FEATURE_DEFINITIONS[feature_group]
        if FEATURE_DICT.get(feature_group)
    ]

    cols = []
    for filename in features_names:
        path = os.path.join(INTERIM_ROOT, f"{filename}.parquet")
        cols += fastparquet.ParquetFile(path).columns

    cols = set([c for c in cols if not c.startswith("__")])

    # Merge the "user based" features
    X_test = pd.merge_asof(
        X_test.sort_values(by="tracker_created_at"),
        data[["user_id", "tracker_created_at"] + USER_FEATURES].sort_values(by="tracker_created_at"),
        on="tracker_created_at",
        by="user_id",
        direction="backward"
    )

    # Merge the "post based" features
    X_test = pd.merge_asof(
        X_test.sort_values(by="tracker_created_at"),
        data[["trackable_id", "tracker_created_at"] + POST_FEATURES].sort_values(by="tracker_created_at"),
        on="tracker_created_at",
        by="trackable_id",
        direction="backward"
    )

    # Merge the "user/post based" features
    X_test = pd.merge_asof(
        X_test.sort_values(by="tracker_created_at"),
        data[["user_id", "trackable_id", "tracker_created_at"] + USER_POST_FEATURES].sort_values(by="tracker_created_at"),
        on="tracker_created_at",
        by=["user_id", "trackable_id"],
        direction="backward"
    )

    # Clean the dataset (bool to int, filter on numerical, fillna)
    X_test = (X_test[cols]
        .set_index(["user_id", "trackable_id", "tracker_created_at"])
        .replace({False: 0, True: 1})
        .select_dtypes(['number'])
        .fillna(0)
    )

    logging.info("Loading model")
    model = load_model(os.path.join(models_root, "final_model"))

    logging.info("Making predictions")
    raw_predictions = pd.DataFrame(
        model.predict(X_test),
        index=X_test.index,
        columns=["predictions"]
    )

    # Add columns to compute the metrics (Mean Rank, MAP@10, etc.)
    X_test["predictions"] = raw_predictions
    X_test["rank"] = (X_test
        .groupby(["user_id", "tracker_created_at"])['predictions']
        .rank(ascending=False)
    )
    X_test["has_been_opened"] = y_test.values

    # Print the mean rank
    mean_rank = X_test.loc[X_test['has_been_opened'] == 1, ["has_been_opened", "predictions", "rank"]]["rank"].mean()
    map_10 = 0
    
    logging.info(f"Mean Rank: {mean_rank}")
    logging.info(f"MAP@10: {map_10}")

    # Saving the predictions
    logging.info("Saving predictions")
    X_test.to_parquet(os.path.join(OUTPUT_ROOT, "raw_predictions.parquet"))
