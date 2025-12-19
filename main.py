import sys

from hotel_reservation_mlops.logging.logger import get_logger
from hotel_reservation_mlops.exception.custom_exception import CustomException
from hotel_reservation_mlops.components.data_ingestion import DataIngestion
from hotel_reservation_mlops.utils.common_functions import read_yaml
from hotel_reservation_mlops.config.paths_config import *

logger = get_logger(__name__)
if __name__ == "__main__":
    try:
        # ============================================================
        # 1) DATA INGESTION
        # ============================================================
        logger.info("=== hotel-reservation-mlops: Data Ingestion Pipeline Started ===")

        data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
        data_ingestion.run()
        logger.info("Data ingestion completed successfully.")
        

        logger.info("=== hotel-reservation-mlops: Data Ingestion Pipeline Finished ===")
    except Exception as e:
        logger.error(f"An error occurred in the Data Ingestion Pipeline: {e}")
        raise CustomException(e, sys)