import os
from typing import List
import pandas as pd


DEFAULT_COLUMNS: List[str] = [ # The raw dataset contains many columns
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
    print(df.head(n=5))

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


def define_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only fully resolved loans and create binary target.
    The other statuses are ambiguous and not useful.

    Returns:
        DataFrame with 'target' column
    """

    if "loan_status" not in df.columns:
        raise ValueError(
            "Column 'loan_status' not found. "
            "Are you using an already processed dataset?"
        )

    df = df.copy()

    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])]

    df["target"] = df["loan_status"].apply(
        lambda x: 1 if x == "Charged Off" else 0
    )

    df = df.drop(columns=["loan_status"])

    return df


##############
# Some necessary cleanups.
# I am replacing strings with numerical values, more suitable for what comes next.


def clean_term(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the term column as numeric.
    """
    df = df.copy()
    df["term"] = df["term"].str.extract(r"(\d+)").astype(int)
    return df


def clean_interest_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["int_rate"] = df["int_rate"].replace("%", "", regex=True).astype(float)
    return df


def clean_emp_length(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {
        "< 1 year": 0,
        "1 year": 1,
        "2 years": 2,
        "3 years": 3,
        "4 years": 4,
        "5 years": 5,
        "6 years": 6,
        "7 years": 7,
        "8 years": 8,
        "9 years": 9,
        "10+ years": 10     # no more info than that, so I just put 10
    }
    df["emp_length"] = df["emp_length"].map(mapping)

    return df


def clean_home_ownership(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    mapping = {
        "RENT": 0,
        "OWN": 1,
        "MORTGAGE": 2,
        "OTHER": 3,     # Design choice:
        "NONE": 3,      # non-informative/unclear categories grouped together
        "ANY": 3
    }

    df["home_ownership"] = df["home_ownership"].map(mapping)

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all (the above) cleaning steps to the dataset.
    """
    df = define_target(df)
    df = clean_term(df)
    df = clean_interest_rate(df)
    df = clean_emp_length(df)
    df = clean_home_ownership(df)

    # Drop rows with missing values (for now?)
    df = df.dropna()

    return df


########################################
# Dataset processed in two steps: raw -> step1 -> clean

def create_subset(input_path: str, output_path: str, columns, nrows):
    df = load_subset(input_path, columns, nrows)

    save_dataframe(df, output_path)
    return df


def process_subset(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    print("\nBefore cleaning:", df.shape)

    df = clean_dataset(df)
    print("\nAfter cleaning:", df.shape)

    basic_infos(df)
    # Answers the question: do we need class weights or not?
    print(f"\nTarget values: {df['target'].value_counts()}")
    print(f"\nTarget proportions: {df['target'].value_counts(normalize=True)}")

    save_dataframe(df, output_path)
    return df


#############
# C++ preparation

def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y

def save_for_cpp(X: pd.DataFrame, y: pd.Series, base_path: str):
    X.to_csv(f"{base_path}_X.csv", index=False)
    y.to_csv(f"{base_path}_y.csv", index=False)


###############################################################################


if __name__ == "__main__":
    RAW_PATH = "data/raw/accepted_2007_to_2018Q4.csv"
    INTERIM_PATH = "data/processed/lendingclub_step1.csv"
    PROCESSED_PATH = "data/processed/lendingclub_clean.csv" # 2-step cleaning

    # Step 1: create subset
    create_subset(
        input_path=RAW_PATH,
        output_path=INTERIM_PATH,
        columns=DEFAULT_COLUMNS,
        nrows=150000
    )

    # Step 2: clean subset
    df = process_subset(
        input_path=INTERIM_PATH,
        output_path=PROCESSED_PATH
    )

    # Step 3: split for C++
    X, y = split_features_target(df)
    save_for_cpp(X, y, "data/processed/lendingclub")