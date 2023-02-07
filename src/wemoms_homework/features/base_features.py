import logging
import os
import pandas as pd

from wemoms_homework.config import load_config
from wemoms_homework.features.feature import Feature

IDS = [
    "trackable_id",
    "user_id",
    "tracker_created_at"
]

USER_FEATURES = [
    "user_is_mom",
    "user_is_pregnant",
    "user_is_trying",
    "user_country_code",
    "days_since_user_account_creation",
    "user_pregnancy_current_day",
    "user_pregnancy_current_week",
    "user_pregnancy_current_month",
    "user_pregnancy_current_trimester",
    "platform",
    "user_followings_count",
    "user_posts_count",
    "user_received_comments_count",
    "user_pictures_count",
    "user_likes_count",
    "user_children_count",
    "user_department"
]

POST_FEATURES = [
    "post_age_in_minutes",
    "author_children_count",
    "post_comments_count",
    "post_likes_count",
    "survey_answers_count",
    "has_picture",
    "has_text",
    "has_video",
    "author_department",
    "author_age",
    "author_amenorrhea_week",
]

LABEL = [
    "has_been_opened"
]

CONF = load_config()
DATA_PATH = CONF["path"]["input_data_path"]
OUTPUT_ROOT = CONF["path"]["interim_data_root"]

class BaseFeatures(Feature):

    @classmethod
    def extract_feature(cls, df, save=False):          
        logging.info("Keeping the base features")

        if save:
            os.makedirs(OUTPUT_ROOT, exist_ok=True)
            df[IDS + USER_FEATURES + POST_FEATURES].to_parquet(
                os.path.join(OUTPUT_ROOT, f"base_features.parquet"),
                index=False
            )

        return df[IDS + USER_FEATURES + POST_FEATURES]