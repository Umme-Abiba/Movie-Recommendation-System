import os
import pandas as pd


class FeatureEngineering:

    def __init__(self):
        self.input_path = "artifacts/processed/cleaned_data.csv"

        self.output_dir = "artifacts/featured"

        self.output_file = os.path.join(
            self.output_dir,
            "featured_data.csv"
        )

    def load_data(self):

        print("Loading cleaned data...")

        df = pd.read_csv(self.input_path)

        return df

    def create_features(self, df):

        print("Creating features...")

        # Release year extraction
        if "release_date" in df.columns:

            df["release_date"] = pd.to_datetime(
                df["release_date"],
                errors="coerce"
            )

            df["release_year"] = (
                df["release_date"].dt.year
            )

        # Budget numeric
        if "budget" in df.columns:

            df["budget"] = pd.to_numeric(
                df["budget"],
                errors="coerce"
            )

        # Runtime numeric
        if "runtime" in df.columns:

            df["runtime"] = pd.to_numeric(
                df["runtime"],
                errors="coerce"
            )

        # Popularity numeric
        if "popularity" in df.columns:

            df["popularity"] = pd.to_numeric(
                df["popularity"],
                errors="coerce"
            )

        # Vote average numeric
        if "vote_average" in df.columns:

            df["vote_average"] = pd.to_numeric(
                df["vote_average"],
                errors="coerce"
            )

        # Revenue numeric
        if "revenue" in df.columns:

            df["revenue"] = pd.to_numeric(
                df["revenue"],
                errors="coerce"
            )

        # Profit feature
        if (
            "budget" in df.columns
            and "revenue" in df.columns
        ):

            df["profit"] = (
                df["revenue"] - df["budget"]
            )

        return df

    def save_data(self, df):

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        df.to_csv(
            self.output_file,
            index=False
        )

        print(
            f"Featured data saved to: {self.output_file}"
        )

    def run(self):

        df = self.load_data()

        df = self.create_features(df)

        self.save_data(df)

        print("Feature Engineering Completed")


if __name__ == "__main__":

    obj = FeatureEngineering()

    obj.run()