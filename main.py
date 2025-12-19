import sys

from hotel_reservation_mlops.logging.logger import get_logger
from hotel_reservation_mlops.exception.custom_exception import CustomException
from hotel_reservation_mlops.components.data_ingestion import DataIngestion
from hotel_reservation_mlops.components.data_preprocessing import DataProcessor  
from hotel_reservation_mlops.utils.common_functions import read_yaml
from hotel_reservation_mlops.config.paths_config import *

logger = get_logger(__name__)

if __name__ == "__main__":
    try:
        # ============================================================
        # 1) DATA INGESTION
        # ============================================================
        logger.info("=== hotel-reservation-mlops: Data Ingestion Pipeline Started ===")
        config = read_yaml(CONFIG_PATH)

        data_ingestion = DataIngestion(config)
        data_ingestion.run()
        logger.info("Data ingestion completed successfully.")
        logger.info("=== hotel-reservation-mlops: Data Ingestion Pipeline Finished ===")

        # ============================================================
        # 2) DATA PROCESSING
        # ============================================================
        logger.info("=== hotel-reservation-mlops: Data Processing Pipeline Started ===")

        data_processor = DataProcessor(
            train_path=TRAIN_FILE_PATH,
            test_path=TEST_FILE_PATH,
            processed_dir=PROCESSED_DIR,
            config_path=CONFIG_PATH,
        )
        data_processor.process()

        logger.info("Data processing completed successfully.")
        logger.info("=== hotel-reservation-mlops: Data Processing Pipeline Finished ===")

    except Exception as e:
        logger.exception("Pipeline failed with an exception")
        raise CustomException(e, sys)
