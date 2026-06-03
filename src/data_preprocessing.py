import os
import pandas as pd


class DataPreprocessing:

    def __init__(self):

        self.input_file = "artifacts/raw/raw_data.csv"

        self.output_dir = "artifacts/processed"

        self.output_file = os.path.join(
            self.output_dir,
            "cleaned_data.csv"
        )

        self.selected_columns = [
            "budget",
            "genres",
            "popularity",
            "release_date",
            "revenue",
            "runtime",
            "vote_average"
        ]

    def load_data(self):

        print("Loading dataset...")

        df = pd.read_csv(
            self.input_file,
            low_memory=False
        )

        print(f"Original Shape: {df.shape}")

        return df

    def preprocess_data(self, df):

        print("Starting preprocessing...")

        # Keep required columns only
        df = df[self.selected_columns]

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Convert columns to numeric
        numeric_columns = [
            "budget",
            "popularity",
            "revenue",
            "runtime",
            "vote_average"
        ]

        for col in numeric_columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        # Remove rows where target is missing
        df = df.dropna(subset=["revenue"])

        # Fill numeric missing values with median
        for col in numeric_columns:

            df[col] = df[col].fillna(
                df[col].median()
            )

        # Fill categorical missing values
        df["genres"] = df["genres"].fillna(
            "Unknown"
        )

        # Fill missing release dates
        df["release_date"] = df["release_date"].fillna(
            "2000-01-01"
        )

        print(f"Processed Shape: {df.shape}")

        return df

    def save_processed_data(self, df):

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        df.to_csv(
            self.output_file,
            index=False
        )

        print(
            f"Processed data saved at: {self.output_file}"
        )

    def run(self):

        df = self.load_data()

        df = self.preprocess_data(df)

        self.save_processed_data(df)

        print("Data Preprocessing Completed")


if __name__ == "__main__":

    preprocessing = DataPreprocessing()

    preprocessing.run()