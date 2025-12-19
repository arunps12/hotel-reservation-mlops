import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

from hotel_reservation_mlops.logging.logger import get_logger
from hotel_reservation_mlops.exception.custom_exception import CustomException
from hotel_reservation_mlops.config.paths_config import *
from hotel_reservation_mlops.utils.common_functions import read_yaml, load_data

logger = get_logger(__name__)


class DataProcessor:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)

        os.makedirs(self.processed_dir, exist_ok=True)

        logger.info(
            f"Initialized DataProcessor | train={train_path}, test={test_path}"
        )

    def preprocess_data(self, df):
        try:
            logger.info(f"Preprocessing data | initial shape={df.shape}")

            df = df.drop(columns=["Unnamed: 0", "Booking_ID"], errors="ignore")
            df = df.drop_duplicates()

            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]

            logger.info(f"Categorical columns: {cat_cols}")
            logger.info(f"Numerical columns: {num_cols}")

            for col in cat_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

            skew_threshold = self.config["data_processing"]["skewness_threshold"]
            skewness = df[num_cols].skew()

            skewed_cols = skewness[skewness > skew_threshold].index.tolist()
            logger.info(f"Skewed columns (> {skew_threshold}): {skewed_cols}")

            for col in skewed_cols:
                df[col] = np.log1p(df[col])

            logger.info(f"Preprocessing completed | final shape={df.shape}")
            return df

        except Exception as e:
            logger.exception("Error during preprocessing")
            raise CustomException("Preprocessing failed", e)

    def balance_data(self, df):
        try:
            logger.info("Applying SMOTE for class balancing")

            X = df.drop(columns="booking_status")
            y = df["booking_status"]

            logger.info(f"Class distribution before SMOTE:\n{y.value_counts()}")

            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)

            logger.info(f"Class distribution after SMOTE:\n{pd.Series(y_res).value_counts()}")

            balanced_df = pd.DataFrame(X_res, columns=X.columns)
            balanced_df["booking_status"] = y_res

            return balanced_df

        except Exception as e:
            logger.exception("Error during SMOTE balancing")
            raise CustomException("Balancing failed", e)

    def select_features(self, df):
        try:
            logger.info("Selecting top features using RandomForest")

            X = df.drop(columns="booking_status")
            y = df["booking_status"]

            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)

            importance_df = pd.DataFrame(
                {"feature": X.columns, "importance": model.feature_importances_}
            ).sort_values(by="importance", ascending=False)

            k = self.config["data_processing"]["no_of_features"]
            selected_features = importance_df["feature"].head(k).tolist()

            logger.info(f"Selected features: {selected_features}")

            return df[selected_features + ["booking_status"]]

        except Exception as e:
            logger.exception("Feature selection failed")
            raise CustomException("Feature selection failed", e)

    def save_data(self, df, file_path):
        try:
            df.to_csv(file_path, index=False)
            logger.info(f"Saved processed data → {file_path} | shape={df.shape}")
        except Exception as e:
            logger.exception("Failed to save processed data")
            raise CustomException("Saving processed data failed", e)

    def process(self):
        logger.info("Starting data processing pipeline")

        train_df = load_data(self.train_path)
        test_df = load_data(self.test_path)

        train_df = self.preprocess_data(train_df)
        test_df = self.preprocess_data(test_df)

        train_df = self.balance_data(train_df)  

        train_df = self.select_features(train_df)
        test_df = test_df[train_df.columns]

        self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
        self.save_data(test_df, PROCESSED_TEST_DATA_PATH)
