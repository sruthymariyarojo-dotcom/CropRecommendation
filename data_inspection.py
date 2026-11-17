
import pandas as pd


# 1. LOAD THE DATASET

df = pd.read_csv("dataset/Crop_recommendation.csv")


# 2. DISPLAY DATASET

print("COMPLETE DATASET")
print(df)

# 3. DISPLAY FIRST FIVE RECORDS

print("\nFIRST FIVE RECORDS")
print(df.head())


# 4. DATASET DIMENSIONS

print("\nDATASET DIMENSIONS")
print("Rows and Columns:", df.shape)

# 5. COLUMN NAMES

print("\nCOLUMN NAMES")
print(df.columns)

# 6. DATA TYPES

print("\nDATA TYPES")
print(df.dtypes)

# 7. DATASET INFORMATION

print("\nDATASET INFORMATION")
df.info()

# 8. CHECK MISSING VALUES

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nTOTAL MISSING VALUES:", df.isnull().sum().sum())

# 9. CHECK DUPLICATE RECORDS

print("\nDUPLICATE RECORDS")
print("Number of Duplicates:", df.duplicated().sum())


# 10. CHECK UNIQUE CROP LABELS

print("\nUNIQUE CROP LABELS")

if "label" in df.columns:
    print("Number of Crop Categories:", df["label"].nunique())
    print("Crop Names:")
    print(sorted(df["label"].dropna().unique()))


# 11. NUMERICAL DATA SUMMARY

print("\nNUMERICAL DATA SUMMARY")
print(df.describe())

# 12. UNIQUE VALUE COUNTS

print("\nUNIQUE VALUE COUNTS")

for column in df.columns:
    print(column, ":", df[column].nunique())

# END OF DATASET INSPECTION

print("\nDATASET INSPECTION COMPLETED")
