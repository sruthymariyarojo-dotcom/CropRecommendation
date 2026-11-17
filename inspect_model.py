
import joblib

MODEL_PATH = "model/crop_recommendation_model.joblib"

print("=" * 60)
print("CROPWISE - SAVED MODEL INSPECTION")
print("=" * 60)

# Load the saved model artifact
saved_object = joblib.load(MODEL_PATH)

print("\nObject Type:", type(saved_object))

# Inspect dictionary contents
if isinstance(saved_object, dict):

    print("\nDICTIONARY KEYS:")
    print(list(saved_object.keys()))

    print("\nCONTENTS:")

    for key, value in saved_object.items():

        print("\nKey:", key)
        print("Value Type:", type(value))
        print(
            "Has predict method:",
            callable(getattr(value, "predict", None))
        )

        # Inspect nested dictionaries if present
        if isinstance(value, dict):

            for nested_key, nested_value in value.items():

                print("  Nested Key:", nested_key)
                print("  Nested Type:", type(nested_value))
                print(
                    "  Nested predict method:",
                    callable(getattr(nested_value, "predict", None))
                )

else:

    print("\nSaved object is not a dictionary.")

print("\n" + "=" * 60)
print("INSPECTION COMPLETED")
print("=" * 60)

