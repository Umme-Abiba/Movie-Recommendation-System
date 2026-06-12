import os
import yaml
import joblib
import mlflow
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


class ModelTrainer:

    def __init__(self):

        self.input_dir = "artifacts/split"

        self.model_dir = "artifacts/model"

        with open("params.yaml", "r") as file:
            self.params = yaml.safe_load(file)

    def run(self):

        X_train = pd.read_csv(
            f"{self.input_dir}/X_train.csv"
        )

        y_train = pd.read_csv(
            f"{self.input_dir}/y_train.csv"
        )

        model = RandomForestRegressor(
            n_estimators=self.params["model"]["n_estimators"],
            max_depth=self.params["model"]["max_depth"],
            random_state=self.params["model"]["random_state"]
        )

        mlflow.start_run()

        model.fit(
            X_train,
            y_train.values.ravel()
        )

        os.makedirs(
            self.model_dir,
            exist_ok=True
        )

        model_path = (
            f"{self.model_dir}/model.pkl"
        )

        joblib.dump(
            model,
            model_path
        )

        mlflow.log_param(
            "n_estimators",
            self.params["model"]["n_estimators"]
        )

        mlflow.log_param(
            "max_depth",
            self.params["model"]["max_depth"]
        )

        mlflow.sklearn.log_model(
            model,
            "random_forest_model"
        )

        mlflow.end_run()

        print("Model Training Completed")


if __name__ == "__main__":
    ModelTrainer().run()