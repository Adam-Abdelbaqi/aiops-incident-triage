import pandas as pd
from sklearn.metrics import f1_score , accuracy_score , recall_score
from src.pipeline import build_pipeline
from src.data import load_data , time_based_split
from src.preprocessing.preprocess import check_missings , type_conversion


def validate_tuning_data(X_train: pd.DataFrame,y_train: pd.Series):

    # Validate types
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a Pandas DataFrame")

    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a Pandas Series")

    # Validate non-empty data
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("X_train and y_train cannot be empty")

    # Validate matching number of samples
    if len(X_train) != len(y_train):
        raise ValueError(
            f"Shape mismatch: X_train has {len(X_train)} samples "
            f"but y_train has {len(y_train)} samples")



def tune_hyperparameters(X_train:pd.DataFrame , y_train:pd.Series):
    """Tune Logistic Regression's C using chronological validation."""

    # Validate training data
    validate_tuning_data(X_train , y_train)

    # # Use the first 80% for fitting and the final 20% for validation.
    # This assumes X_train and y_train are already chronologically ordered.
    split_index = int(len(X_train) * 0.80)

    if split_index == 0 or split_index >= len(X_train):
        raise ValueError(
            "Not enough samples to create fitting and validation sets."
        )

    X_fit = X_train.iloc[:split_index].copy()
    X_val = X_train.iloc[split_index:].copy()

    y_fit = y_train.iloc[:split_index].copy()
    y_val = y_train.iloc[split_index:].copy()

    # Logisitic Regression Regularization paramter Candidates
    C_values = [3.0, 5.0, 10.0, 20.0, 30.0]
    results = []

    for C in C_values:
        print(f"\nEvaluating for C {C}..")
        # Initializing the Pipeline
        pipeline = build_pipeline(C= C)

        pipeline.fit(X_fit , y_fit)
        preds = pipeline.predict(X_val)

        result= {
            "C" : C,
            "Accuracy" : accuracy_score(y_true= y_val , y_pred= preds),
            "Weighted F1" : f1_score(y_true= y_val , y_pred= preds , average= "weighted"),
            "Macro F1" : f1_score(y_true= y_val , y_pred= preds , average= "macro"),
            "Critical_recall" : recall_score(y_true=y_val , y_pred= preds , labels=["Critical"] , average= None , zero_division= 0)[0],
        }

        results.append(result)

        print(
            f"Validation Macro F1: {result['Macro F1']:.4f} | "
            f"Critical Recall: {result['Critical_recall']:.4f}"
        )

    
    results_df = pd.DataFrame(results).sort_values(
        by="Macro F1",
        ascending=False,
    ).reset_index(drop=True)

    print("\n========== TUNING RESULTS ==========")
    print(results_df.to_string(index=False))

    best_C = results_df.loc[0, "C"]

    print(f"\nBest C: {best_C}")
    print(
        f"Best validation Macro F1: "
        f"{results_df.loc[0, 'Macro F1']:.4f}"
    )
    print(
        f"Validation Critical Recall: "
        f"{results_df.loc[0, 'Critical_recall']:.4f}"
    )

    return {
        "best_params": {"classifier__C": best_C},
        "results": results_df,
    }



# 1. Load the dataset
DATA_PATH = r"C:\Users\Dell\Desktop\Adam\NLP Projects\AIOps Incident Triage\data\raw\tickets_clean.csv"

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





tuning_results = tune_hyperparameters(X_train, y_train)

best_params = tuning_results["best_params"]
results_df = tuning_results["results"]
