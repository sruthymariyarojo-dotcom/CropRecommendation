
# ============================================================
# CROP RECOMMENDATION SYSTEM
# DATA PREPROCESSING, MODEL TRAINING AND EVALUATION
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import sklearn

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "Crop_recommendation.csv"

MODEL_DIR = BASE_DIR / "model"

REPORTS_DIR = BASE_DIR / "reports"

MODEL_PATH = MODEL_DIR / "crop_recommendation_model.joblib"

# Create folders if they do not exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Seven input features
FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

# Target variable
TARGET = "label"

# Expected crop categories
EXPECTED_CROPS = {
    "apple", "banana", "blackgram", "chickpea",
    "coconut", "coffee", "cotton", "grapes",
    "jute", "kidneybeans", "lentil", "maize",
    "mango", "mothbeans", "mungbean", "muskmelon",
    "orange", "papaya", "pigeonpeas", "pomegranate",
    "rice", "watermelon"
}

# Configuration
RANDOM_STATE = 42

TEST_SIZE = 0.20

CV_FOLDS = 5


print("=" * 70)
print("             CROP RECOMMENDATION SYSTEM")
print("       DATA PREPROCESSING AND MODEL TRAINING")
print("=" * 70)


# ============================================================
# 2. LOAD AND VALIDATE THE DATASET
# ============================================================

print("\n1. LOADING DATASET")
print("-" * 70)

# Check dataset availability
if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Standardize column names
df.columns = [
    str(column).strip()
    for column in df.columns
]

# Required columns
REQUIRED_COLUMNS = FEATURES + [TARGET]

# Check required columns
missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Select required columns in the correct order
df = df[REQUIRED_COLUMNS].copy()

# Standardize crop labels
df[TARGET] = (
    df[TARGET]
    .astype("string")
    .str.strip()
    .str.lower()
)

# Convert numerical features
for column in FEATURES:

    df[column] = pd.to_numeric(
        df[column],
        errors="raise"
    )

# Check missing values
if df.isnull().any().any():
    raise ValueError(
        "Missing values detected. "
        "Resolve data quality issues before training."
    )

# Check infinite and non-finite numerical values
if not np.isfinite(
    df[FEATURES].to_numpy(dtype=float)
).all():

    raise ValueError(
        "Infinite or non-finite numerical values detected."
    )

# Verify crop categories
actual_crops = set(df[TARGET].unique())

if actual_crops != EXPECTED_CROPS:

    raise ValueError(
        "Crop category mismatch detected.\n"
        f"Missing: {sorted(EXPECTED_CROPS - actual_crops)}\n"
        f"Unexpected: {sorted(actual_crops - EXPECTED_CROPS)}"
    )

print("Dataset loaded successfully.")

print("Total Records:", len(df))

print("Number of Features:", len(FEATURES))

print("Number of Crop Categories:", len(actual_crops))


# ============================================================
# 3. SEPARATE INPUT FEATURES AND TARGET VARIABLE
# ============================================================

print("\n2. SEPARATING FEATURES AND TARGET")
print("-" * 70)

# X contains the seven input features
X = df[FEATURES].copy()

# y contains the crop labels
y = df[TARGET].copy()

print("Input Features (X):")
print(list(X.columns))

print("\nTarget Variable (y):")
print(TARGET)

print("\nX Shape:", X.shape)

print("y Shape:", y.shape)


# ============================================================
# 4. SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

print("\n3. TRAIN-TEST SPLITTING")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y

)

print("Training Records:", len(X_train))

print("Testing Records:", len(X_test))

print(
    "Training Percentage:",
    round(len(X_train) / len(df) * 100, 2),
    "%"
)

print(
    "Testing Percentage:",
    round(len(X_test) / len(df) * 100, 2),
    "%"
)

# Verify that every crop is represented
if set(y_train.unique()) != set(y_test.unique()):

    raise ValueError(
        "Some crop categories are missing "
        "from the training or testing set."
    )

print("\nCrop Distribution in Training Data:")
print(y_train.value_counts().sort_index())

print("\nCrop Distribution in Testing Data:")
print(y_test.value_counts().sort_index())


# ============================================================
# 5. DEFINE MACHINE LEARNING MODELS
# ============================================================

print("\n4. INITIALIZING MACHINE LEARNING MODELS")
print("-" * 70)

models = {

    # Decision Tree
    "Decision Tree": Pipeline([
        (
            "classifier",
            DecisionTreeClassifier(
                random_state=RANDOM_STATE
            )
        )
    ]),

    # Random Forest
    "Random Forest": Pipeline([
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]),

    # K-Nearest Neighbors
    # StandardScaler is required for distance-based KNN
    "K-Nearest Neighbors": Pipeline([

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )

    ]),

    # Gaussian Naive Bayes
    "Gaussian Naive Bayes": Pipeline([
        (
            "classifier",
            GaussianNB()
        )
    ])

}

print("Models initialized:")

for model_name in models:
    print("-", model_name)


# ============================================================
# 6. MODEL COMPARISON USING CROSS-VALIDATION
# ============================================================

print("\n5. MODEL COMPARISON USING CROSS-VALIDATION")
print("-" * 70)

# Five-fold stratified cross-validation
cv = StratifiedKFold(

    n_splits=CV_FOLDS,

    shuffle=True,

    random_state=RANDOM_STATE

)

# Evaluation metrics
scoring = {

    "accuracy": "accuracy",

    "precision": "precision_macro",

    "recall": "recall_macro",

    "f1": "f1_macro"

}

comparison_results = []

for model_name, model in models.items():

    print(f"\nEvaluating: {model_name}")

    # Perform cross-validation using training data only
    scores = cross_validate(

        model,

        X_train,

        y_train,

        cv=cv,

        scoring=scoring,

        n_jobs=1,

        return_train_score=False,

        error_score="raise"

    )

    # Calculate average evaluation scores
    result = {

        "Model": model_name,

        "CV Accuracy": scores["test_accuracy"].mean(),

        "CV Accuracy Std": scores["test_accuracy"].std(),

        "CV Precision (Macro)": scores["test_precision"].mean(),

        "CV Recall (Macro)": scores["test_recall"].mean(),

        "CV F1-Score (Macro)": scores["test_f1"].mean()

    }

    comparison_results.append(result)

    print(
        "Mean CV Accuracy:",
        round(result["CV Accuracy"], 4)
    )

    print(
        "Mean CV F1-Score:",
        round(result["CV F1-Score (Macro)"], 4)
    )


# ============================================================
# 7. COMPARE AND SELECT THE BEST MODEL
# ============================================================

print("\n6. MODEL COMPARISON RESULTS")
print("-" * 70)

comparison_df = pd.DataFrame(comparison_results)

# Rank models using macro F1-score first,
# followed by accuracy
comparison_df = comparison_df.sort_values(

    by=[
        "CV F1-Score (Macro)",
        "CV Accuracy"
    ],

    ascending=False

).reset_index(drop=True)

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

# Save model comparison results
comparison_df.to_csv(

    REPORTS_DIR / "model_comparison.csv",

    index=False

)

# Select the highest-ranked model
best_model_name = comparison_df.iloc[0]["Model"]

best_model = models[best_model_name]

print("\nSelected Model:", best_model_name)


# ============================================================
# 8. TRAIN THE SELECTED MODEL
# ============================================================

print("\n7. TRAINING THE SELECTED MODEL")
print("-" * 70)

# Train using the complete training dataset
best_model.fit(

    X_train,

    y_train

)

print("Selected model trained successfully.")


# ============================================================
# 9. FINAL MODEL EVALUATION
# ============================================================

print("\n8. FINAL MODEL EVALUATION")
print("-" * 70)

# Predict using the untouched test dataset
y_pred = best_model.predict(X_test)

# Calculate evaluation metrics
test_accuracy = accuracy_score(
    y_test,
    y_pred
)

test_precision = precision_score(

    y_test,

    y_pred,

    average="macro",

    zero_division=0

)

test_recall = recall_score(

    y_test,

    y_pred,

    average="macro",

    zero_division=0

)

test_f1 = f1_score(

    y_test,

    y_pred,

    average="macro",

    zero_division=0

)

print("Selected Model:", best_model_name)

print(
    "Test Accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)

print(
    "Test Precision:",
    round(test_precision, 4)
)

print(
    "Test Recall:",
    round(test_recall, 4)
)

print(
    "Test F1-Score:",
    round(test_f1, 4)
)


# ============================================================
# 10. GENERATE AND SAVE CLASSIFICATION REPORT
# ============================================================

print("\n9. CLASSIFICATION REPORT")
print("-" * 70)

classes = sorted(y.unique())

# Generate the classification report as a dictionary
report_dict = classification_report(

    y_test,

    y_pred,

    labels=classes,

    target_names=classes,

    output_dict=True,

    zero_division=0

)

# Display the report in the terminal
print(
    classification_report(

        y_test,

        y_pred,

        labels=classes,

        target_names=classes,

        zero_division=0

    )
)

# Keep only dictionary-valued entries.
# The overall accuracy is a float, so it is excluded here.
# Overall accuracy is saved separately in final_test_metrics.csv.

report_rows = {

    name: metrics

    for name, metrics in report_dict.items()

    if isinstance(metrics, dict)

}

# Convert the classification report into a DataFrame
classification_df = pd.DataFrame.from_dict(

    report_rows,

    orient="index"

)

classification_df.index.name = "Class"

# Save classification report
classification_df.to_csv(

    REPORTS_DIR / "classification_report.csv"

)

print("Classification report saved successfully.")


# ============================================================
# 11. GENERATE AND SAVE CONFUSION MATRIX
# ============================================================

print("\n10. CONFUSION MATRIX")
print("-" * 70)

cm = confusion_matrix(

    y_test,

    y_pred,

    labels=classes

)

confusion_df = pd.DataFrame(

    cm,

    index=classes,

    columns=classes

)

confusion_df.index.name = "Actual Crop"

confusion_df.columns.name = "Predicted Crop"

print(confusion_df)

# Save confusion matrix values
confusion_df.to_csv(

    REPORTS_DIR / "confusion_matrix.csv"

)

# Generate confusion matrix visualization
plt.figure(figsize=(16, 14))

sns.heatmap(

    confusion_df,

    annot=True,

    fmt="d",

    cmap="Blues",

    linewidths=0.5,

    cbar=True

)

plt.title(

    f"Confusion Matrix - {best_model_name}",

    fontsize=16,

    fontweight="bold"

)

plt.xlabel("Predicted Crop")

plt.ylabel("Actual Crop")

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(

    REPORTS_DIR / "confusion_matrix.png",

    dpi=300

)

plt.close()

print("Confusion matrix saved successfully.")


# ============================================================
# 12. SAVE FINAL TEST METRICS
# ============================================================

print("\n11. SAVING EVALUATION RESULTS")
print("-" * 70)

test_metrics = {

    "Model": best_model_name,

    "Accuracy": test_accuracy,

    "Precision (Macro)": test_precision,

    "Recall (Macro)": test_recall,

    "F1-Score (Macro)": test_f1

}

metrics_df = pd.DataFrame(

    [test_metrics]

)

metrics_df.to_csv(

    REPORTS_DIR / "final_test_metrics.csv",

    index=False

)

print(
    metrics_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

print("Final evaluation metrics saved successfully.")


# ============================================================
# 13. SAVE THE TRAINED MODEL
# ============================================================

print("\n12. SAVING TRAINED MODEL")
print("-" * 70)

# Save the fitted model and its metadata together
model_artifact = {

    "model": best_model,

    "model_name": best_model_name,

    "feature_names": FEATURES,

    "target_name": TARGET,

    "classes": classes,

    "test_metrics": test_metrics,

    "sklearn_version": sklearn.__version__

}

joblib.dump(

    model_artifact,

    MODEL_PATH

)

print("Model saved successfully.")

print("Model Location:", MODEL_PATH)


# ============================================================
# 14. VERIFY THE SAVED MODEL
# ============================================================

print("\n13. VERIFYING SAVED MODEL")
print("-" * 70)

# Load the saved model artifact
loaded_artifact = joblib.load(MODEL_PATH)

loaded_model = loaded_artifact["model"]

# Select one test record for verification
sample_input = X_test.iloc[[0]]

# Predict using the original fitted model
original_prediction = best_model.predict(
    sample_input
)[0]

# Predict using the reloaded model
loaded_prediction = loaded_model.predict(
    sample_input
)[0]

# Verify that both predictions match
if original_prediction != loaded_prediction:

    raise RuntimeError(
        "Saved model verification failed. "
        "Predictions do not match."
    )

print("Sample Input:")
print(sample_input)

print("\nPredicted Crop:", loaded_prediction)

print("Actual Crop:", y_test.iloc[0])

print("Original and reloaded model predictions match.")

print("Saved model loaded and verified successfully.")


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("       MODEL TRAINING AND EVALUATION COMPLETED")

print("=" * 70)

print("Best Model:", best_model_name)

print(
    "Final Test Accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)

print("Training Records:", len(X_train))

print("Testing Records:", len(X_test))

print("Crop Categories:", len(classes))

print("\nSaved Model:")
print(MODEL_PATH)

print("\nGenerated Report Files:")

for file in sorted(REPORTS_DIR.iterdir()):

    if file.is_file():
        print(file.name)

print("\nAll operations completed successfully.")

print("=" * 70)
