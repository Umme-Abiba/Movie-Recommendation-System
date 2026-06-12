import os
import json
import joblib
import mlflow
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class ModelEvaluation:

    def __init__(self):

        self.model_path = (
            "artifacts/model/model.pkl"
        )

        self.split_dir = (
            "artifacts/split"
        )

        self.output_dir = (
            "artifacts/evaluation"
        )

    def run(self):

        model = joblib.load(
            self.model_path
        )

        X_test = pd.read_csv(
            f"{self.split_dir}/X_test.csv"
        )

        y_test = pd.read_csv(
            f"{self.split_dir}/y_test.csv"
        )

        preds = model.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            preds
        )

        mse = mean_squared_error(
            y_test,
            preds
        )

        r2 = r2_score(
            y_test,
            preds
        )

        metrics = {
            "MAE": float(mae),
            "MSE": float(mse),
            "R2": float(r2)
        }

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        with open(
            f"{self.output_dir}/metrics.json",
            "w"
        ) as file:
            json.dump(
                metrics,
                file,
                indent=4
            )

        print(metrics)

if __name__ == "__main__":
    ModelEvaluation().run()