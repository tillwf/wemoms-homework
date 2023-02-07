import logging
import os
import pandas as pd

from wemoms_homework.config import load_config
from wemoms_homework.features.feature import Feature

CONF = load_config()
DATA_PATH = CONF["path"]["input_data_path"]
OUTPUT_ROOT = CONF["path"]["interim_data_root"]


class PostPopularity(Feature):

    @classmethod
    def extract_feature(cls, df, save=False, windows=["1d", "7d", "28d"]):
        """Compute the popularity of a post using windows"""
        logging.info("Computing the features PostPopularity")

        def add_post_past_popularity(window):
            logging.info(f"\tComputing past {window} popularity")
            return (df[["tracker_created_at", "trackable_id", "has_been_opened"]]
                      .set_index('tracker_created_at')
                      .sort_index()
                      .groupby('trackable_id', sort=False)
                      .has_been_opened
                      .rolling(window, closed='left')
                      .agg({
                          f"post_last_{window}_views_count": len,
                          f"post_last_{window}_clicks_count": sum,
                          f"post_last_{window}_ratio": lambda x: sum(x)/len(x)
                      })
                      .fillna(0))
        
        df_res = pd.DataFrame()
        for window in windows:
            df_temp = add_post_past_popularity(window)
            df_res = pd.concat([df_res, df_temp], axis=0)
            if save:
                os.makedirs(OUTPUT_ROOT, exist_ok=True)
                df_temp.to_parquet(os.path.join(OUTPUT_ROOT, f"post_popularity_{window}.parquet"))

        return df_res