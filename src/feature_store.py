import os
import pandas as pd

class FeatureStore:

    def __init__(self):

        self.input_file = "artifacts/featured/featured_data.csv"

        self.output_dir = "artifacts/feature_store"

        self.output_file = os.path.join(
            self.output_dir,
            "features.csv"
        )

    def run(self):

        df = pd.read_csv(self.input_file)

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        df.to_csv(
            self.output_file,
            index=False
        )

        print("Feature Store Created")


if __name__ == "__main__":
    FeatureStore().run()