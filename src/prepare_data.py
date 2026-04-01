import os
from typing import List
import pandas as pd


DEFAULT_COLUMNS: List[str] = [ # The raw dataset contains several many columns
    "loan_amnt",               # I keep some of the useful ones
    "term",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "emp_length",
    "home_ownership",
    "loan_status",
]


def load_subset(
    file_path: str,
    columns: List[str],
    nrows: int = 150000
    ) -> pd.DataFrame:
    """
    Load a subset of the LendingClub dataset.

    Args:
        file_path: Path to raw CSV file
        columns: Columns to select
        nrows: Number of rows to load

    Returns:
        pandas DataFrame
    """
    df = pd.read_csv(
        file_path,
        usecols=columns,
        nrows=nrows
    )

    return df


def basic_infos(df: pd.DataFrame) -> None:
    """
    Print basic diagnostics of the dataset.

    Args:
        df: Input DataFrame
    """
    print("\n Shape:")
    print(df.shape)

    print("\nHead:")
    print(df.head())

    print("\n Missing values:")
    print(df.isnull().sum())


def save_dataframe(df: pd.DataFrame, output_path: str) -> None:
    """
    Save DataFrame to CSV.

    Args:
        df: DataFrame to save
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved processed data to: {output_path}")


def prepare_dataset(
    input_path: str,
    output_path: str,
    columns: List[str] = DEFAULT_COLUMNS,
    nrows: int = 150000
) -> pd.DataFrame:
    """
    Full pipeline: load → inspect → save.

    Args:
        input_path: Raw dataset path
        output_path: Output dataset path
        columns: Columns to keep
        nrows: Number of rows to load

    Returns:
        Processed DataFrame
    """
    df = load_subset(input_path, columns, nrows)
    basic_infos(df)
    save_dataframe(df, output_path)
    return df


###############################################################################

if __name__ == "__main__":
    INPUT_PATH = "data/raw/accepted_2007_to_2018Q4.csv"
    OUTPUT_PATH = "data/processed/lendingclub_step1.csv"

    prepare_dataset(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        nrows=150000
    )