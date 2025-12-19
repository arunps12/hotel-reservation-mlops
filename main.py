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
        logger.info("=== Hotel Reservation MLOps Pipeline Started ===")

        config = read_yaml(CONFIG_PATH)

        logger.info("Stage 1: Data Ingestion")
        DataIngestion(config).run()

        logger.info("Stage 2: Data Processing")
        DataProcessor(
            train_path=TRAIN_FILE_PATH,
            test_path=TEST_FILE_PATH,
            processed_dir=PROCESSED_DIR,
            config_path=CONFIG_PATH,
        ).process()

        logger.info("=== Pipeline completed successfully ===")

    except Exception as e:
        logger.exception("Pipeline failed")
        raise CustomException(e, sys)
