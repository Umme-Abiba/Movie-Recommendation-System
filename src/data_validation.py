import os
import pandas as pd


class DataValidation:

    def __init__(self):

        self.input_file = "artifacts/raw/raw_data.csv"

        self.output_dir = "artifacts/validation"

        self.status_file = os.path.join(
            self.output_dir,
            "status.txt"
        )

        self.required_columns = [
            "budget",
            "genres",
            "popularity",
            "release_date",
            "revenue",
            "runtime",
            "vote_average"
        ]

    def load_data(self):

        print("Loading raw data...")

        df = pd.read_csv(
            self.input_file,
            low_memory=False
        )

        print(f"Dataset Shape: {df.shape}")

        return df

    def validate_columns(self, df):

        missing_columns = []

        for col in self.required_columns:

            if col not in df.columns:
                missing_columns.append(col)

        return missing_columns

    def validate_duplicates(self, df):

        duplicate_count = df.duplicated().sum()

        return duplicate_count

    def validate_null_values(self, df):

        return df.isnull().sum()

    def save_validation_report(
        self,
        missing_columns,
        duplicate_count,
        null_values
    ):

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        with open(
            self.status_file,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                "========== DATA VALIDATION REPORT ==========\n\n"
            )

            # Column Validation
            if len(missing_columns) == 0:

                file.write(
                    "All required columns are present.\n\n"
                )

            else:

                file.write(
                    f"Missing Columns: {missing_columns}\n\n"
                )

            # Duplicate Validation
            file.write(
                f"Duplicate Rows: {duplicate_count}\n\n"
            )

            # Null Values
            file.write(
                "Null Values:\n"
            )

            for column, count in null_values.items():

                file.write(
                    f"{column}: {count}\n"
                )

        print(
            f"Validation report saved at: {self.status_file}"
        )

    def run(self):

        df = self.load_data()

        missing_columns = self.validate_columns(df)

        duplicate_count = self.validate_duplicates(df)

        null_values = self.validate_null_values(df)

        self.save_validation_report(
            missing_columns,
            duplicate_count,
            null_values
        )

        print("Data Validation Completed")


if __name__ == "__main__":

    validation = DataValidation()

    validation.run()