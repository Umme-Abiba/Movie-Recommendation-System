import os
import pandas as pd

print("Current Directory:")
print(os.getcwd())

class DataIngestion:

    def __init__(self):
        self.input_path = "Data/movies_metadata.csv"
        self.output_dir = "artifacts/raw"
        self.output_file = os.path.join(
            self.output_dir,
            "raw_data.csv"
        )

    def load_data(self):
        """
        Load dataset from source
        """

        print("Loading dataset...")

        df = pd.read_csv(
            self.input_path,
            low_memory=False
        )

        print(f"Dataset Shape: {df.shape}")

        return df

    def save_raw_data(self, df):
        """
        Save raw dataset into artifacts
        """

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        df.to_csv(
            self.output_file,
            index=False
        )

        print(
            f"Raw data saved at: {self.output_file}"
        )

    def run(self):

        df = self.load_data()

        self.save_raw_data(df)

        print("Data Ingestion Completed")


if __name__ == "__main__":

    ingestion = DataIngestion()

    ingestion.run()