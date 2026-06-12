import os
import yaml
import pandas as pd

from sklearn.model_selection import train_test_split


class DataSplit:

    def __init__(self):

        self.input_file = (
            "artifacts/feature_store/features.csv"
        )

        self.output_dir = (
            "artifacts/split"
        )

        with open("params.yaml", "r") as file:
            self.params = yaml.safe_load(file)

    def run(self):

        df = pd.read_csv(
            self.input_file
        )

        target_column = self.params[
            "training"
        ]["target_column"]

        X = df.drop(
            columns=[target_column]
        )

        y = df[target_column]

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=self.params[
                    "data_split"
                ]["test_size"],
                random_state=self.params[
                    "data_split"
                ]["random_state"]
            )
        )

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        X_train.to_csv(
            f"{self.output_dir}/X_train.csv",
            index=False
        )

        X_test.to_csv(
            f"{self.output_dir}/X_test.csv",
            index=False
        )

        y_train.to_csv(
            f"{self.output_dir}/y_train.csv",
            index=False
        )

        y_test.to_csv(
            f"{self.output_dir}/y_test.csv",
            index=False
        )

        print("Train-Test Split Completed")


if __name__ == "__main__":

    DataSplit().run()