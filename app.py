
from pathlib import Path
import math

import joblib
import pandas as pd

from flask import (
    Flask,
    abort,
    render_template,
    request,
    send_from_directory
)


# ============================================================
# CROPWISE - FLASK WEB APPLICATION
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT PATHS AND MODEL CONFIGURATION
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
REPORTS_DIR = BASE_DIR / "reports"

MODEL_PATH = MODEL_DIR / "crop_recommendation_model.joblib"


# Exact feature order used during model training.
MODEL_FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]


# The 22 crop categories available in the dataset.
CROP_CATEGORIES = [
    "apple",
    "banana",
    "blackgram",
    "chickpea",
    "coconut",
    "coffee",
    "cotton",
    "grapes",
    "jute",
    "kidneybeans",
    "lentil",
    "maize",
    "mango",
    "mothbeans",
    "mungbean",
    "muskmelon",
    "orange",
    "papaya",
    "pigeonpeas",
    "pomegranate",
    "rice",
    "watermelon"
]


# Minimum and maximum values observed during the dataset audit.
# These are advisory ranges, not strict input restrictions.

DATASET_RANGES = {
    "N": (0.0, 140.0),
    "P": (5.0, 145.0),
    "K": (5.0, 205.0),
    "temperature": (8.825674745, 43.67549305),
    "humidity": (14.25803981, 99.98187601),
    "ph": (3.504752314, 9.93509073),
    "rainfall": (20.21126747, 298.5601175)
}


FEATURE_LABELS = {
    "N": "Nitrogen",
    "P": "Phosphorus",
    "K": "Potassium",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "ph": "Soil pH",
    "rainfall": "Rainfall"
}


# Allowed report images.
ALLOWED_REPORT_IMAGES = {
    "crop_distribution.png",
    "correlation_heatmap.png",
    "confusion_matrix.png",
    "feature_distributions.png",
    "feature_boxplots.png"
}


# ------------------------------------------------------------
# 2. INITIALIZE FLASK APPLICATION
# ------------------------------------------------------------

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR)
)


# ------------------------------------------------------------
# 3. LOAD THE SAVED MACHINE LEARNING MODEL
# ------------------------------------------------------------

def load_trained_model():
    """
    Load the saved model artifact and extract the trained Pipeline.

    The saved Joblib file contains a dictionary.
    The actual trained Pipeline is stored under the 'model' key.
    """

    if not MODEL_PATH.is_file():

        app.logger.error(
            "Trained model file not found: %s",
            MODEL_PATH
        )

        return None, [], (
            "The trained model file was not found. "
            "Please check the model folder."
        )

    try:

        # Load the complete saved artifact.
        artifact = joblib.load(MODEL_PATH)

        if not isinstance(artifact, dict):

            raise TypeError(
                "The saved model artifact is not a dictionary."
            )

        # ----------------------------------------------------
        # Verify saved feature names
        # ----------------------------------------------------

        saved_features = artifact.get("feature_names")

        if not isinstance(saved_features, (list, tuple)):

            raise ValueError(
                "The artifact does not contain valid feature names."
            )

        saved_features = [
            str(feature) for feature in saved_features
        ]

        if saved_features != MODEL_FEATURES:

            raise ValueError(
                "The saved model feature order does not match "
                "the application's feature order."
            )

        # ----------------------------------------------------
        # Extract the actual trained Pipeline
        # ----------------------------------------------------

        trained_model = artifact.get("model")

        if trained_model is None:

            raise ValueError(
                "The saved artifact does not contain the model."
            )

        if not callable(
            getattr(trained_model, "predict", None)
        ):

            raise TypeError(
                "The extracted model does not provide predict()."
            )

        # ----------------------------------------------------
        # Verify crop classes
        # ----------------------------------------------------

        saved_classes = artifact.get("classes")

        if not isinstance(saved_classes, (list, tuple)):

            raise ValueError(
                "The artifact does not contain valid crop classes."
            )

        saved_classes = [
            str(value).strip().lower()
            for value in saved_classes
        ]

        if len(saved_classes) != len(set(saved_classes)):

            raise ValueError(
                "Duplicate crop classes were found in the artifact."
            )

        if set(saved_classes) != set(CROP_CATEGORIES):

            raise ValueError(
                "Saved crop categories do not match "
                "the application's crop categories."
            )

        # ----------------------------------------------------
        # Verify Pipeline class order
        # ----------------------------------------------------

        pipeline_classes = getattr(
            trained_model,
            "classes_",
            None
        )

        if pipeline_classes is not None:

            model_classes = [
                str(value).strip().lower()
                for value in pipeline_classes
            ]

            if set(model_classes) != set(saved_classes):

                raise ValueError(
                    "Pipeline classes do not match "
                    "the saved artifact classes."
                )

        else:

            # Use artifact class ordering if the Pipeline
            # does not expose classes_.
            model_classes = saved_classes

        # ----------------------------------------------------
        # Verify Pipeline feature names, if available
        # ----------------------------------------------------

        pipeline_features = getattr(
            trained_model,
            "feature_names_in_",
            None
        )

        if pipeline_features is not None:

            pipeline_features = [
                str(feature)
                for feature in pipeline_features
            ]

            if pipeline_features != MODEL_FEATURES:

                raise ValueError(
                    "Pipeline feature order does not match "
                    "the expected feature order."
                )

        # ----------------------------------------------------
        # Log model information
        # ----------------------------------------------------

        app.logger.info(
            "CropWise model loaded successfully: %s",
            artifact.get("model_name", "Unknown model")
        )

        app.logger.info(
            "Verified model features: %s",
            saved_features
        )

        app.logger.info(
            "Verified crop categories: %d",
            len(model_classes)
        )

        if not callable(
            getattr(trained_model, "predict_proba", None)
        ):

            app.logger.warning(
                "The model does not support predict_proba(). "
                "Only the primary prediction will be displayed."
            )

        # Return the actual Pipeline, not the dictionary.
        return trained_model, model_classes, None

    except Exception:

        app.logger.exception(
            "Failed to load the saved CropWise model artifact."
        )

        return None, [], (
            "The trained model could not be loaded. "
            "Check the model artifact and installed packages. "
            "See the terminal for details."
        )


# Load the model once when Flask starts.
PREDICTION_MODEL, MODEL_CLASSES, MODEL_LOAD_ERROR = load_trained_model()


# ------------------------------------------------------------
# 4. SHARED TEMPLATE INFORMATION
# ------------------------------------------------------------

@app.context_processor
def inject_app_status():

    return {
        "model_file_available": MODEL_PATH.is_file(),
        "model_loaded": PREDICTION_MODEL is not None
    }


# ------------------------------------------------------------
# 5. DASHBOARD PAGE
# ------------------------------------------------------------

@app.route("/", methods=["GET"], endpoint="home")
def home():

    return render_template(
        "dashboard.html",
        active_page="dashboard"
    )


# ------------------------------------------------------------
# 6. CROP RECOMMENDATION PAGE
# ------------------------------------------------------------

@app.route("/recommend", methods=["GET"], endpoint="recommend")
def recommend():

    return render_template(
        "recommend.html",
        active_page="recommend",
        input_values={},
        errors={},
        prediction=None,
        prediction_options=[],
        has_probability_scores=False,
        range_warnings=[],
        global_error=None
    )


# ------------------------------------------------------------
# 7. GET TOP CROP CLASS ESTIMATES
# ------------------------------------------------------------

def get_top_crop_classes(
    input_frame,
    primary_prediction,
    limit=3
):
    """
    Return the primary predicted crop and up to two alternatives.

    predict() determines the primary crop.
    predict_proba() provides probability estimates for the classes.

    These estimates represent the model's classification output.
    They do not represent guaranteed real-world crop suitability.
    """

    # Keep the primary prediction even if probability
    # estimates are not available.
    top_options = [
        {
            "crop": primary_prediction,
            "probability_pct": None
        }
    ]

    # Check whether the model supports predict_proba().
    if not callable(
        getattr(PREDICTION_MODEL, "predict_proba", None)
    ):

        return top_options, False

    try:

        probability_output = (
            PREDICTION_MODEL.predict_proba(input_frame)
        )

        if len(probability_output) != 1:

            raise ValueError(
                "Unexpected number of probability output rows."
            )

        probability_row = probability_output[0]

        if len(probability_row) != len(MODEL_CLASSES):

            raise ValueError(
                "Probability output does not match the model classes."
            )

        ranked_classes = []

        # Match each probability with its corresponding crop.
        for crop_name, probability in zip(
            MODEL_CLASSES,
            probability_row
        ):

            probability = float(probability)

            if (
                not math.isfinite(probability)
                or not 0.0 <= probability <= 1.0
            ):

                raise ValueError(
                    "Invalid probability value returned by the model."
                )

            ranked_classes.append(
                {
                    "crop": crop_name,
                    "probability_pct": round(
                        probability * 100.0,
                        2
                    )
                }
            )

        # Sort from highest to lowest probability.
        ranked_classes.sort(
            key=lambda item: item["probability_pct"],
            reverse=True
        )

        # Keep predict() output as the primary crop.
        primary_item = next(
            (
                item for item in ranked_classes
                if item["crop"] == primary_prediction
            ),
            None
        )

        if primary_item is None:

            raise ValueError(
                "Primary prediction was not found in probability classes."
            )

        # Select the highest-ranked alternatives.
        alternatives = [
            item for item in ranked_classes
            if item["crop"] != primary_prediction
        ]

        top_options = (
            [primary_item]
            + alternatives[:max(0, limit - 1)]
        )

        return top_options, True

    except Exception:

        # Do not lose a valid primary prediction if
        # probability calculation fails.
        app.logger.exception(
            "Could not calculate alternative crop estimates."
        )

        return [
            {
                "crop": primary_prediction,
                "probability_pct": None
            }
        ], False


# ------------------------------------------------------------
# 8. PROCESS CROP PREDICTION
# ------------------------------------------------------------

@app.route("/predict", methods=["POST"], endpoint="predict")
def predict():

    # Collect the seven submitted input values.
    input_values = {
        feature: request.form.get(feature, "").strip()
        for feature in MODEL_FEATURES
    }

    errors = {}
    numeric_values = {}
    range_warnings = []

    prediction = None
    prediction_options = []
    has_probability_scores = False
    global_error = None

    # --------------------------------------------------------
    # Parse and validate required inputs
    # --------------------------------------------------------

    for feature in MODEL_FEATURES:

        raw_value = input_values[feature]

        if not raw_value:

            errors[feature] = "This field is required."
            continue

        try:

            numeric_value = float(raw_value)

        except (TypeError, ValueError):

            errors[feature] = "Enter a valid numeric value."
            continue

        if not math.isfinite(numeric_value):

            errors[feature] = "Enter a finite numeric value."
            continue

        numeric_values[feature] = numeric_value

    # --------------------------------------------------------
    # Basic numerical validation
    # --------------------------------------------------------

    for feature in ("N", "P", "K"):

        if (
            feature in numeric_values
            and numeric_values[feature] < 0
        ):

            errors[feature] = "This value cannot be negative."

    if "humidity" in numeric_values:

        if not 0 <= numeric_values["humidity"] <= 100:

            errors["humidity"] = (
                "Humidity must be between 0 and 100%."
            )

    if "ph" in numeric_values:

        if not 0 <= numeric_values["ph"] <= 14:

            errors["ph"] = (
                "Soil pH must be between 0 and 14."
            )

    if "rainfall" in numeric_values:

        if numeric_values["rainfall"] < 0:

            errors["rainfall"] = (
                "Rainfall cannot be negative."
            )

    # --------------------------------------------------------
    # Return form if validation fails
    # --------------------------------------------------------

    if errors:

        return render_template(
            "recommend.html",
            active_page="recommend",
            input_values=input_values,
            errors=errors,
            prediction=None,
            prediction_options=[],
            has_probability_scores=False,
            range_warnings=[],
            global_error=None
        )

    # --------------------------------------------------------
    # Check model availability
    # --------------------------------------------------------

    if PREDICTION_MODEL is None:

        global_error = (
            MODEL_LOAD_ERROR
            or "The trained model is unavailable. "
               "Check the model folder and restart Flask."
        )

        return render_template(
            "recommend.html",
            active_page="recommend",
            input_values=input_values,
            errors={},
            prediction=None,
            prediction_options=[],
            has_probability_scores=False,
            range_warnings=[],
            global_error=global_error
        )

    # --------------------------------------------------------
    # Check observed dataset ranges
    # --------------------------------------------------------

    for feature in MODEL_FEATURES:

        observed_min, observed_max = DATASET_RANGES[feature]

        value = numeric_values[feature]

        if value < observed_min or value > observed_max:

            range_warnings.append(
                f"{FEATURE_LABELS[feature]} ({value:g}) "
                f"is outside the observed dataset range "
                f"({observed_min:g} to {observed_max:g})."
            )

    # --------------------------------------------------------
    # Generate primary prediction and alternatives
    # --------------------------------------------------------

    try:

        # Maintain the exact feature order used in training.
        input_frame = pd.DataFrame(
            [
                [
                    numeric_values[feature]
                    for feature in MODEL_FEATURES
                ]
            ],
            columns=MODEL_FEATURES
        )

        # Generate the primary crop prediction.
        model_output = PREDICTION_MODEL.predict(
            input_frame
        )

        if len(model_output) != 1:

            raise ValueError(
                "The model returned an unexpected number of predictions."
            )

        prediction = str(
            model_output[0]
        ).strip().lower()

        if prediction not in CROP_CATEGORIES:

            raise ValueError(
                f"Unrecognised crop category: {prediction}"
            )

        # Obtain the primary crop and up to two alternatives.
        prediction_options, has_probability_scores = (
            get_top_crop_classes(
                input_frame,
                prediction,
                limit=3
            )
        )

    except Exception:

        app.logger.exception(
            "Crop prediction failed."
        )

        global_error = (
            "The model could not generate a prediction for these values. "
            "Please check the inputs and try again."
        )

        prediction = None
        prediction_options = []
        has_probability_scores = False

    return render_template(
        "recommend.html",
        active_page="recommend",
        input_values=input_values,
        errors={},
        prediction=prediction,
        prediction_options=prediction_options,
        has_probability_scores=has_probability_scores,
        range_warnings=range_warnings,
        global_error=global_error
    )


# ------------------------------------------------------------
# 9. MODEL AND DATASET PAGE
# ------------------------------------------------------------

@app.route("/model", methods=["GET"], endpoint="model_info")
def model_info():

    comparison_path = REPORTS_DIR / "model_comparison.csv"
    metrics_path = REPORTS_DIR / "final_test_metrics.csv"

    model_comparison = []
    test_metrics = None

    required_comparison_columns = {
        "Model",
        "CV Accuracy",
        "CV F1-Score (Macro)"
    }

    required_metric_columns = {
        "Model",
        "Accuracy",
        "Precision (Macro)",
        "Recall (Macro)",
        "F1-Score (Macro)"
    }

    # --------------------------------------------------------
    # Read model comparison report
    # --------------------------------------------------------

    if comparison_path.is_file():

        try:

            comparison_df = pd.read_csv(
                comparison_path
            )

            if required_comparison_columns.issubset(
                comparison_df.columns
            ):

                for column in [
                    "CV Accuracy",
                    "CV F1-Score (Macro)"
                ]:

                    comparison_df[column] = pd.to_numeric(
                        comparison_df[column],
                        errors="coerce"
                    )

                comparison_df = comparison_df.dropna(
                    subset=[
                        "Model",
                        "CV Accuracy",
                        "CV F1-Score (Macro)"
                    ]
                )

                comparison_df = comparison_df.sort_values(
                    by=[
                        "CV F1-Score (Macro)",
                        "CV Accuracy"
                    ],
                    ascending=False
                )

                model_comparison = comparison_df.to_dict(
                    orient="records"
                )

            else:

                app.logger.warning(
                    "Required columns missing from model_comparison.csv."
                )

        except Exception:

            app.logger.exception(
                "Could not read model_comparison.csv."
            )

    # --------------------------------------------------------
    # Read final test metrics
    # --------------------------------------------------------

    if metrics_path.is_file():

        try:

            metrics_df = pd.read_csv(
                metrics_path
            )

            if (
                not metrics_df.empty
                and required_metric_columns.issubset(
                    metrics_df.columns
                )
            ):

                metrics_row = metrics_df.iloc[0].copy()

                numeric_columns = [
                    "Accuracy",
                    "Precision (Macro)",
                    "Recall (Macro)",
                    "F1-Score (Macro)"
                ]

                for column in numeric_columns:

                    metrics_row[column] = pd.to_numeric(
                        metrics_row[column],
                        errors="coerce"
                    )

                if metrics_row[numeric_columns].notna().all():

                    test_metrics = metrics_row.to_dict()

                else:

                    app.logger.warning(
                        "Invalid metric values in final_test_metrics.csv."
                    )

            else:

                app.logger.warning(
                    "final_test_metrics.csv is empty or missing columns."
                )

        except Exception:

            app.logger.exception(
                "Could not read final_test_metrics.csv."
            )

    # Determine selected model.
    if test_metrics and test_metrics.get("Model"):

        selected_model = str(
            test_metrics["Model"]
        )

    elif model_comparison:

        selected_model = str(
            model_comparison[0]["Model"]
        )

    else:

        selected_model = None

    # --------------------------------------------------------
    # Check report image availability
    # --------------------------------------------------------

    chart_files = [
        "crop_distribution.png",
        "correlation_heatmap.png",
        "confusion_matrix.png",
        "feature_distributions.png",
        "feature_boxplots.png"
    ]

    chart_available = {
        filename: (REPORTS_DIR / filename).is_file()
        for filename in chart_files
    }

    return render_template(
        "model_dataset.html",
        active_page="model",
        model_comparison=model_comparison,
        test_metrics=test_metrics,
        selected_model=selected_model,
        crop_categories=CROP_CATEGORIES,
        chart_available=chart_available
    )


# ------------------------------------------------------------
# 10. SERVE APPROVED REPORT IMAGES
# ------------------------------------------------------------

@app.route(
    "/reports/<filename>",
    methods=["GET"],
    endpoint="report_image"
)
def report_image(filename):

    if filename not in ALLOWED_REPORT_IMAGES:

        abort(404)

    image_path = REPORTS_DIR / filename

    if not image_path.is_file():

        abort(404)

    return send_from_directory(
        str(REPORTS_DIR),
        filename,
        conditional=True
    )


# ------------------------------------------------------------
# 11. ABOUT CROPWISE PAGE
# ------------------------------------------------------------

@app.route("/about", methods=["GET"], endpoint="about")
def about():

    return render_template(
        "about.html",
        active_page="about"
    )


# ------------------------------------------------------------
# 12. RUN FLASK APPLICATION
# ------------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
