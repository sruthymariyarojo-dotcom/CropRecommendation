
# ============================================================
# CROP RECOMMENDATION SYSTEM
# EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# ------------------------------------------------------------
# 1. LOAD THE DATASET
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "Crop_recommendation.csv"

REPORTS_DIR = BASE_DIR / "reports"

# Create reports folder if it does not exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATASET_PATH)

FEATURES = [
    "N", "P", "K",
    "temperature", "humidity",
    "ph", "rainfall"
]

TARGET = "label"

# Standardize column names
df.columns = [str(col).strip() for col in df.columns]

# Standardize crop labels
df[TARGET] = (
    df[TARGET]
    .astype("string")
    .str.strip()
    .str.lower()
)

# Verify numerical features
for column in FEATURES:
    df[column] = pd.to_numeric(
        df[column],
        errors="raise"
    )

# Stop if required data is incomplete
if df[FEATURES + [TARGET]].isnull().any().any():
    raise ValueError(
        "Missing values detected. Resolve data quality issues first."
    )

# Set visualization style
sns.set_theme(style="whitegrid")


print("=" * 65)
print("       CROP RECOMMENDATION SYSTEM - EDA")
print("=" * 65)


# ------------------------------------------------------------
# 2. BASIC DATASET INFORMATION
# ------------------------------------------------------------

print("\n1. DATASET INFORMATION")
print("-" * 65)

print("Total Records:", df.shape[0])
print("Total Features:", len(FEATURES))
print("Target Variable:", TARGET)


# ------------------------------------------------------------
# 3. CROP DISTRIBUTION
# ------------------------------------------------------------

print("\n2. CROP DISTRIBUTION")
print("-" * 65)

crop_counts = df[TARGET].value_counts().sort_index()

print(crop_counts)

print("\nNumber of Crop Categories:", len(crop_counts))

if crop_counts.nunique() == 1:
    print("Dataset Class Distribution: Balanced")
else:
    print("Dataset Class Distribution: Not Equal")

# Create crop distribution chart
plt.figure(figsize=(14, 7))

crop_counts.plot(
    kind="bar",
    edgecolor="black"
)

plt.title(
    "Distribution of Crop Categories",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Crop Type")
plt.ylabel("Number of Records")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "crop_distribution.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 4. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n3. DESCRIPTIVE STATISTICS")
print("-" * 65)

statistics = df[FEATURES].describe()

print(statistics)

# Save statistical summary
statistics.to_csv(
    REPORTS_DIR / "descriptive_statistics.csv"
)


# ------------------------------------------------------------
# 5. FEATURE DISTRIBUTION
# ------------------------------------------------------------

print("\n4. FEATURE DISTRIBUTION ANALYSIS")
print("-" * 65)

fig, axes = plt.subplots(
    4, 2,
    figsize=(14, 16)
)

axes = axes.flatten()

for index, feature in enumerate(FEATURES):

    sns.histplot(
        data=df,
        x=feature,
        bins=25,
        kde=True,
        ax=axes[index]
    )

    axes[index].set_title(
        f"Distribution of {feature}"
    )

    axes[index].set_xlabel(feature)
    axes[index].set_ylabel("Frequency")

# Remove the unused eighth subplot
fig.delaxes(axes[7])

plt.suptitle(
    "Distribution of Input Features",
    fontsize=17,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "feature_distributions.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 6. OUTLIER ANALYSIS USING BOXPLOTS
# ------------------------------------------------------------

print("\n5. OUTLIER ANALYSIS")
print("-" * 65)

fig, axes = plt.subplots(
    4, 2,
    figsize=(14, 14)
)

axes = axes.flatten()

for index, feature in enumerate(FEATURES):

    sns.boxplot(
        x=df[feature],
        ax=axes[index]
    )

    axes[index].set_title(
        f"Boxplot of {feature}"
    )

    axes[index].set_xlabel(feature)

fig.delaxes(axes[7])

plt.suptitle(
    "Outlier Analysis of Input Features",
    fontsize=17,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "feature_boxplots.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 7. CORRELATION ANALYSIS
# ------------------------------------------------------------

print("\n6. FEATURE CORRELATION")
print("-" * 65)

correlation = df[FEATURES].corr()

print(correlation.round(3))

plt.figure(figsize=(11, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    center=0
)

plt.title(
    "Correlation Heatmap of Input Features",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "correlation_heatmap.png",
    dpi=300
)

plt.close()

correlation.to_csv(
    REPORTS_DIR / "feature_correlation.csv"
)


# ------------------------------------------------------------
# 8. CROP-WISE FEATURE SUMMARY
# ------------------------------------------------------------

print("\n7. CROP-WISE FEATURE SUMMARY")
print("-" * 65)

crop_summary = (
    df.groupby(TARGET)[FEATURES]
    .mean()
    .round(2)
)

print(crop_summary)

crop_summary.to_csv(
    REPORTS_DIR / "crop_feature_summary.csv"
)


# ------------------------------------------------------------
# 9. FINAL REPORT
# ------------------------------------------------------------

print("\n" + "=" * 65)

print("EXPLORATORY DATA ANALYSIS COMPLETED")

print("\nThe following files have been generated:")

for file in sorted(REPORTS_DIR.iterdir()):

    if file.is_file():
        print(file.name)

print("=" * 65)
