
# ============================================================
# CROP RECOMMENDATION SYSTEM
# DATA CLEANING AND DATA QUALITY AUDIT
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path


# ------------------------------------------------------------
# 1. LOAD THE DATASET
# ------------------------------------------------------------

# Get the project directory automatically
BASE_DIR = Path(__file__).resolve().parent

# Specify the dataset location
DATASET_PATH = BASE_DIR / "dataset" / "Crop_recommendation.csv"

# Check whether the dataset exists
if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATASET_PATH}"
    )

# Read the CSV file
df = pd.read_csv(DATASET_PATH)

print("=" * 65)
print("             CROP RECOMMENDATION SYSTEM")
print("                 DATA CLEANING")
print("=" * 65)


# ------------------------------------------------------------
# 2. CHECK DATASET DIMENSIONS
# ------------------------------------------------------------

print("\n1. DATASET DIMENSIONS")
print("-" * 65)

print("Number of Rows    :", df.shape[0])
print("Number of Columns :", df.shape[1])


# ------------------------------------------------------------
# 3. CHECK AND STANDARDIZE COLUMN NAMES
# ------------------------------------------------------------

print("\n2. COLUMN NAME VALIDATION")
print("-" * 65)

# Remove unnecessary spaces from column names
df.columns = [
    str(column).strip()
    for column in df.columns
]

# Expected columns in our dataset
EXPECTED_COLUMNS = [
    "N", "P", "K",
    "temperature", "humidity",
    "ph", "rainfall", "label"
]

# Match column names without considering letter case
column_mapping = {
    column.lower(): column
    for column in EXPECTED_COLUMNS
}

rename_columns = {}

for column in df.columns:

    key = column.lower()

    if key in column_mapping:
        rename_columns[column] = column_mapping[key]

# Rename columns
df = df.rename(columns=rename_columns)

# Check for duplicate column names
if df.columns.duplicated().any():
    raise ValueError(
        "Duplicate column names found. Please inspect the CSV."
    )

# Identify missing and unexpected columns
missing_columns = [
    column for column in EXPECTED_COLUMNS
    if column not in df.columns
]

extra_columns = [
    column for column in df.columns
    if column not in EXPECTED_COLUMNS
]

print("Available Columns:", list(df.columns))

if missing_columns:

    print("Missing Required Columns:", missing_columns)

    raise ValueError(
        "The dataset does not contain all required columns."
    )

if extra_columns:
    print("Additional Columns:", extra_columns)

# Keep the required columns in a consistent order
# Additional columns are not required for this project
df = df[EXPECTED_COLUMNS].copy()

print("Required column validation completed.")


# ------------------------------------------------------------
# 4. CHECK DATA TYPES AND CONVERT NUMERIC FEATURES
# ------------------------------------------------------------

print("\n3. DATA TYPE VALIDATION")
print("-" * 65)

# Input features
FEATURES = [
    "N", "P", "K",
    "temperature", "humidity",
    "ph", "rainfall"
]

# Target variable
TARGET = "label"

# Store the number of invalid numerical entries
conversion_errors = {}

# Convert all input features into numeric types
for column in FEATURES:

    original_values = df[column]

    # Identify empty or whitespace-only values
    blank_mask = (
        original_values
        .astype("string")
        .str.strip()
        .eq("")
        .fillna(False)
    )

    # Treat blank values as missing values
    normalized_values = original_values.mask(
        blank_mask,
        np.nan
    )

    # Convert values into numeric format
    converted_values = pd.to_numeric(
        normalized_values,
        errors="coerce"
    )

    # Identify values that could not be converted
    invalid_mask = (
        normalized_values.notna() &
        converted_values.isna()
    )

    conversion_errors[column] = int(
        invalid_mask.sum()
    )

    # Update the working DataFrame
    df[column] = converted_values


print("\nData Types After Numeric Validation:")
print(df.dtypes)

print("\nNon-numeric Values Found:")

for column, count in conversion_errors.items():
    print(f"{column}: {count}")


# ------------------------------------------------------------
# 5. STANDARDIZE AND VALIDATE CROP LABELS
# ------------------------------------------------------------

print("\n4. CROP LABEL VALIDATION")
print("-" * 65)

# Remove unnecessary spaces and standardize letter case
df[TARGET] = (
    df[TARGET]
    .astype("string")
    .str.strip()
    .str.lower()
)

# Convert empty labels into missing values
df[TARGET] = df[TARGET].replace("", pd.NA)

# Expected crop categories in the selected dataset
EXPECTED_CROPS = {
    "apple", "banana", "blackgram", "chickpea",
    "coconut", "coffee", "cotton", "grapes",
    "jute", "kidneybeans", "lentil", "maize",
    "mango", "mothbeans", "mungbean", "muskmelon",
    "orange", "papaya", "pigeonpeas", "pomegranate",
    "rice", "watermelon"
}

# Get actual crop categories
actual_crops = set(
    df[TARGET].dropna().unique()
)

print(
    "Number of Crop Categories:",
    len(actual_crops)
)

print("\nCrop Categories:")
print(sorted(actual_crops))

# Compare expected and actual crop categories
missing_crops = EXPECTED_CROPS - actual_crops
unexpected_crops = actual_crops - EXPECTED_CROPS

print("\nEXPECTED CROP CATEGORY VALIDATION")

print("Missing Crop Categories:", sorted(missing_crops))
print("Unexpected Crop Categories:", sorted(unexpected_crops))

if not missing_crops and not unexpected_crops:

    print(
        "All 22 expected crop categories are present."
    )

else:

    print(
        "WARNING: Crop category mismatch detected."
    )


# ------------------------------------------------------------
# 6. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\n5. MISSING VALUE ANALYSIS")
print("-" * 65)

missing_values = df.isnull().sum()

print(missing_values)

total_missing = int(missing_values.sum())

print("\nTotal Missing Values:", total_missing)


# ------------------------------------------------------------
# 7. CHECK DUPLICATE RECORDS
# ------------------------------------------------------------

print("\n6. DUPLICATE RECORD ANALYSIS")
print("-" * 65)

# Check exact duplicate rows in the normalized data
duplicate_count = int(df.duplicated().sum())

print(
    "Number of Exact Duplicate Records:",
    duplicate_count
)


# ------------------------------------------------------------
# 8. CHECK INVALID NUMERICAL VALUES
# ------------------------------------------------------------

print("\n7. INVALID VALUE ANALYSIS")
print("-" * 65)

# Define conditions for invalid values
invalid_conditions = {

    "N": df["N"].lt(0),

    "P": df["P"].lt(0),

    "K": df["K"].lt(0),

    "humidity": (
        df["humidity"].lt(0) |
        df["humidity"].gt(100)
    ),

    "ph": (
        df["ph"].lt(0) |
        df["ph"].gt(14)
    ),

    "rainfall": df["rainfall"].lt(0)

}

invalid_value_counts = {}

for column, condition in invalid_conditions.items():

    count = int(
        condition.fillna(False).sum()
    )

    invalid_value_counts[column] = count

    print(f"{column}: {count} invalid values")


# Check infinite numerical values
print("\nINFINITE VALUES")

infinite_value_counts = {}

for column in FEATURES:

    values = df[column].to_numpy(
        dtype=float,
        na_value=np.nan
    )

    infinite_count = int(
        np.isinf(values).sum()
    )

    infinite_value_counts[column] = infinite_count

    print(
        f"{column}: {infinite_count}"
    )


# ------------------------------------------------------------
# 9. CHECK DUPLICATE INPUTS WITH DIFFERENT CROP LABELS
# ------------------------------------------------------------

print("\n8. CONFLICTING CROP LABEL ANALYSIS")
print("-" * 65)

# Convert numerical features to a NumPy array
feature_values = df[FEATURES].to_numpy(
    dtype=float,
    na_value=np.nan
)

# Consider only rows containing finite input values
finite_rows = np.isfinite(
    feature_values
).all(axis=1)

# Require a valid crop label as well
valid_rows = (
    finite_rows &
    df[TARGET].notna().to_numpy()
)

valid_df = df.loc[
    valid_rows,
    FEATURES + [TARGET]
]

# Find identical feature combinations with different labels
label_counts = (
    valid_df
    .groupby(FEATURES)[TARGET]
    .nunique()
)

conflicting_groups = label_counts[
    label_counts > 1
]

print(
    "Number of Identical Input Combinations "
    "With Different Labels:",
    len(conflicting_groups)
)


# ------------------------------------------------------------
# 10. NUMERICAL DATA SUMMARY
# ------------------------------------------------------------

print("\n9. NUMERICAL DATA SUMMARY")
print("-" * 65)

# Exclude infinite values from the statistical summary
# without modifying the original working DataFrame
finite_numeric_data = df[FEATURES].replace(
    [np.inf, -np.inf],
    np.nan
)

print(
    finite_numeric_data.describe()
)


# ------------------------------------------------------------
# 11. CHECK FEATURE VALUE RANGES
# ------------------------------------------------------------

print("\n10. FEATURE VALUE RANGES")
print("-" * 65)

for column in FEATURES:

    valid_values = finite_numeric_data[column].dropna()

    if valid_values.empty:

        print(
            f"{column}: No valid finite numerical values"
        )

    else:

        print(
            f"{column}: "
            f"Minimum = {valid_values.min()}, "
            f"Maximum = {valid_values.max()}"
        )


# ------------------------------------------------------------
# 12. DATASET CLEANING AUDIT SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("                 AUDIT SUMMARY")
print("=" * 65)

print("Total Records:", len(df))
print("Total Columns:", len(df.columns))

print("Missing Values:", total_missing)

print("Duplicate Records:", duplicate_count)

print(
    "Non-numeric Entries:",
    sum(conversion_errors.values())
)

print(
    "Invalid Numerical Values:",
    sum(invalid_value_counts.values())
)

print(
    "Infinite Values:",
    sum(infinite_value_counts.values())
)

print(
    "Crop Categories:",
    len(actual_crops)
)

print(
    "Missing Expected Crop Categories:",
    len(missing_crops)
)

print(
    "Unexpected Crop Categories:",
    len(unexpected_crops)
)

print(
    "Conflicting Input Combinations:",
    len(conflicting_groups)
)

print("\nDATA QUALITY AUDIT COMPLETED.")

print(
    "\nNOTE: No records have been deleted "
    "and the original CSV has not been modified."
)

print("=" * 65)
