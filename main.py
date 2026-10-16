import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from src.pipeline import build_pipeline
from src.preprocessing.preprocess import  type_conversion,  check_missings
from src.data import load_data , time_based_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Load the dataset
DATA_PATH = r"data/raw/tickets_clean.csv"

df = load_data(DATA_PATH)

# 2. Inspect missing values
missing_percentages = check_missings(df)

print("\nMissing values (%):")
print(missing_percentages[missing_percentages > 0])

# 3. Convert the specified column types
df = type_conversion(df)

# 4. Split chronologically: earlier incidents train the model,
#    later incidents evaluate it.
train_df, test_df = time_based_split(df, train_size=0.8)

# 5. Separate features and target
X_train = train_df.drop(columns=["priority"])
y_train = train_df["priority"].astype(str)

X_test = test_df.drop(columns=["priority"])
y_test = test_df["priority"].astype(str)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(
    f"Training date range: {train_df['created_date'].min()} "
    f"to {train_df['created_date'].max()}"
)
print(
    f"Testing date range: {test_df['created_date'].min()} "
    f"to {test_df['created_date'].max()}"
)

# 6. Inspect class distributions
print("\nTraining priority distribution (%):")
print((y_train.value_counts(normalize=True) * 100).round(2).sort_index())

print("\nTesting priority distribution (%):")
print((y_test.value_counts(normalize=True) * 100).round(2).sort_index())

# 7. Build the complete pipeline
pipeline = build_pipeline()

# 8. Train the pipeline
print("\nTraining the pipeline...")
pipeline.fit(X_train, y_train)

print("Training completed.")

# 9. Generate predictions on the training and test sets
print("\nGenerating training predictions...")
train_predictions = pipeline.predict(X_train)

print("\nGenerating test predictions...")
predictions = pipeline.predict(X_test)

# 10. Evaluate training performance
print("\n========== TRAINING PERFORMANCE ==========")

print(
    "Training accuracy:",
    accuracy_score(y_train, train_predictions)
)

print(
    "Training weighted F1:",
    f1_score(
        y_train,
        train_predictions,
        average="weighted",
        zero_division=0
    )
)

print(
    "Training macro F1:",
    f1_score(
        y_train,
        train_predictions,
        average="macro",
        zero_division=0
    )
)

# Evaluate test performance
print("\n========== TEST PERFORMANCE ==========")

print("Test accuracy:", accuracy_score(y_test, predictions))

print(
    "Test weighted F1:",
    f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )
)

print(
    "Test macro F1:",
    f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )
)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=["Critical", "High", "Low", "Medium"],
        zero_division=0
    )
)

# Plot the test confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    labels=["Critical", "High", "Low", "Medium"],
    cmap="Blues",
    values_format="d"
)

plt.title("AIOps Incident Triage - Test Confusion Matrix")
plt.tight_layout()
plt.show()