import os
import joblib

import mlflow
import mlflow.sklearn

import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from hotel_reservation_mlops.logging.logger import get_logger
from hotel_reservation_mlops.exception.custom_exception import CustomException
from hotel_reservation_mlops.config.paths_config import (
    PROCESSED_TRAIN_DATA_PATH,
    PROCESSED_TEST_DATA_PATH,
    MODEL_OUTPUT_PATH,
)
from hotel_reservation_mlops.utils.common_functions import load_data

# Import params for model training
from hotel_reservation_mlops.config.model_params import LIGHTGM_PARAMS, RANDOM_SEARCH_PARAMS

logger = get_logger(__name__)


class ModelTraining:
    """
    Train a LightGBM model with RandomizedSearchCV, evaluate, save model,
    and log everything to local MLflow (mlruns/).
    """

    def __init__(self, train_path: str, test_path: str, model_output_path: str):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist = LIGHTGM_PARAMS
        self.search_cfg = RANDOM_SEARCH_PARAMS

        logger.info(
            f"Initialized ModelTraining | train={self.train_path} | test={self.test_path} | model_out={self.model_output_path}"
        )

    def load_and_split_data(self):
        try:
            logger.info(f"Loading processed train data from: {self.train_path}")
            train_df = load_data(self.train_path)
            logger.info(f"Train shape: {train_df.shape}")

            logger.info(f"Loading processed test data from: {self.test_path}")
            test_df = load_data(self.test_path)
            logger.info(f"Test shape: {test_df.shape}")

            if "booking_status" not in train_df.columns or "booking_status" not in test_df.columns:
                raise ValueError("Target column 'booking_status' not found in processed datasets.")

            X_train = train_df.drop(columns=["booking_status"])
            y_train = train_df["booking_status"]

            X_test = test_df.drop(columns=["booking_status"])
            y_test = test_df["booking_status"]

            logger.info(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
            logger.info(f"X_test shape: {X_test.shape} | y_test shape: {y_test.shape}")

            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.exception("Failed to load/split processed data for training")
            raise CustomException("Failed to load/split processed data for training", e)

    def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Initializing LightGBM model for hyperparameter tuning")

            base_model = lgb.LGBMClassifier(
                random_state=self.search_cfg.get("random_state", 42),
                verbosity=1
            )

            logger.info(
                "RandomizedSearchCV config | "
                f"n_iter={self.search_cfg.get('n_iter')} | "
                f"cv={self.search_cfg.get('cv')} | "
                f"scoring={self.search_cfg.get('scoring')} | "
                f"n_jobs={self.search_cfg.get('n_jobs')} | "
                f"verbose={self.search_cfg.get('verbose')}"
            )

            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=self.params_dist,
                n_iter=self.search_cfg.get("n_iter", 20),
                cv=self.search_cfg.get("cv", 5),
                n_jobs=self.search_cfg.get("n_jobs", -1),
                verbose=self.search_cfg.get("verbose", 2),
                random_state=self.search_cfg.get("random_state", 42),
                scoring=self.search_cfg.get("scoring", "accuracy"),
            )

            logger.info("Starting hyperparameter tuning (RandomizedSearchCV)")
            search.fit(X_train, y_train)

            best_model = search.best_estimator_
            best_params = search.best_params_

            logger.info(f"Hyperparameter tuning complete | best_score={search.best_score_}")
            logger.info(f"Best params: {best_params}")

            return best_model, best_params, search.best_score_

        except Exception as e:
            logger.exception("Model training/hyperparameter tuning failed")
            raise CustomException("Model training/hyperparameter tuning failed", e)

    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating model on test set")
            y_pred = model.predict(X_test)

            
            n_unique = len(set(y_test))
            avg = "binary" if n_unique == 2 else "weighted"

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average=avg, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average=avg, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, average=avg, zero_division=0)),
            }

            logger.info(
                "Test metrics | "
                f"accuracy={metrics['accuracy']:.4f} | "
                f"precision={metrics['precision']:.4f} | "
                f"recall={metrics['recall']:.4f} | "
                f"f1={metrics['f1']:.4f}"
            )

            return metrics

        except Exception as e:
            logger.exception("Model evaluation failed")
            raise CustomException("Model evaluation failed", e)

    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            joblib.dump(model, self.model_output_path)

            logger.info(f"Model saved to: {self.model_output_path}")
            return self.model_output_path

        except Exception as e:
            logger.exception("Saving model failed")
            raise CustomException("Saving model failed", e)

    def _setup_local_mlflow(self):
        """
        Local-only MLflow: logs to ./mlruns directory by default.
        """
        experiment_name = "hotel-reservation-mlops"
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set to: {experiment_name} (local)")

    def run(self):
        try:
            logger.info("Starting Model Training pipeline")
            self._setup_local_mlflow()

            run_name = "lgbm_random_search_local"

            with mlflow.start_run(run_name=run_name):
                # Log input datasets (as artifacts)
                if os.path.exists(self.train_path):
                    mlflow.log_artifact(self.train_path, artifact_path="datasets")
                if os.path.exists(self.test_path):
                    mlflow.log_artifact(self.test_path, artifact_path="datasets")

                # Load data
                X_train, y_train, X_test, y_test = self.load_and_split_data()

                # Train
                best_model, best_params, best_cv_score = self.train_lgbm(X_train, y_train)

                # Evaluate
                metrics = self.evaluate_model(best_model, X_test, y_test)

                # Save model locally
                model_path = self.save_model(best_model)

                # Log params + metrics
                mlflow.log_params(best_params)
                mlflow.log_metric("best_cv_score", float(best_cv_score))
                mlflow.log_metrics(metrics)

                # Log model to MLflow 
                mlflow.sklearn.log_model(best_model, artifact_path="model")

                #  log the saved .pkl file as an artifact
                mlflow.log_artifact(model_path, artifact_path="model_file")

                logger.info("Model Training pipeline completed successfully")

        except Exception as e:
            logger.exception("Model Training pipeline failed")
            raise CustomException("Model Training pipeline failed", e)


if __name__ == "__main__":
    trainer = ModelTraining(
        train_path=PROCESSED_TRAIN_DATA_PATH,
        test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH,
    )
    trainer.run()
