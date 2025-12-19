import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split

from hotel_reservation_mlops.logging.logger import get_logger
from hotel_reservation_mlops.exception.custom_exception import CustomException
from hotel_reservation_mlops.config.paths_config import *

logger = get_logger(__name__)


class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train_ratio"]

        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info(
            f"Initialized DataIngestion | bucket={self.bucket_name}, "
            f"file={self.file_name}, train_ratio={self.train_test_ratio}"
        )

    def download_csv_from_gcp(self):
        try:
            logger.info("Downloading CSV from GCS")

            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)

            blob.download_to_filename(RAW_FILE_PATH)

            logger.info(f"CSV downloaded successfully → {RAW_FILE_PATH}")

        except Exception as e:
            logger.exception("Failed to download CSV from GCS")
            raise CustomException("Failed to download CSV file", e)

    def split_data(self):
        try:
            logger.info("Splitting raw data into train/test")

            data = pd.read_csv(RAW_FILE_PATH)
            logger.info(f"Raw data shape: {data.shape}")

            train_data, test_data = train_test_split(
                data,
                test_size=1 - self.train_test_ratio,
                random_state=42,
            )

            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)

            logger.info(f"Train shape: {train_data.shape}")
            logger.info(f"Test shape: {test_data.shape}")
            logger.info("Train/Test split saved successfully")

        except Exception as e:
            logger.exception("Failed during train/test split")
            raise CustomException("Failed to split data", e)

    def run(self):
        logger.info("Running data ingestion steps")
        self.download_csv_from_gcp()
        self.split_data()
