import os
import pandas as pd


class FeatureEngineering:

    def __init__(self):

        self.input_path = (
            "artifacts/processed/cleaned_data.csv"
        )

        self.output_dir = (
            "artifacts/featured"
        )

        self.output_file = os.path.join(
            self.output_dir,
            "featured_data.csv"
        )

    def load_data(self):

        print("Loading cleaned data...")

        df = pd.read_csv(
            self.input_path
        )

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

            # Remove original date column
            df.drop(
                columns=["release_date"],
                inplace=True
            )

        # Convert genres to numeric
        if "genres" in df.columns:

            df["genres"] = (
                df["genres"]
                .astype("category")
                .cat.codes
            )

        numeric_columns = [
            "budget",
            "popularity",
            "runtime",
            "vote_average",
            "revenue"
        ]

        for col in numeric_columns:

            if col in df.columns:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        # Fill any remaining null values
        df = df.fillna(0)

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

        print(
            "Feature Engineering Completed"
        )


if __name__ == "__main__":

    obj = FeatureEngineering()

    obj.run()