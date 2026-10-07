from sklearn.metrics import precision_score , classification_report , accuracy_score , recall_score ,f1_score , confusion_matrix , ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

def validate_evaluation_data(X_test:pd.DataFrame , y_test: pd.Series):

    # Validate that X_test is a DataFrame and y_test is a Series:
    if not isinstance(X_test , pd.DataFrame):
        raise TypeError("X_test Must be a DataFrame")

    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test Must be a Pandas Series")

    # Validate that X_test and y_test is not Empty
    if (len(X_test) == 0) or (len(y_test)) == 0:
        raise ValueError("X_test and y_test Cannot Be Empty")

    # Validate that the shape of X_test matches the shape y_test 
    if len(X_test) != len(y_test): 
        raise ValueError(f"Shape Mismatch: X_test has {len(X_test)} Samples "
                         f"but y_test has {len(y_test)} Samples")



def generate_predictions(model , X_test:pd.DataFrame):
    preds = model.predict(X_test)

    return preds

def calculate_metrics(y_test:pd.Series , preds):
    metrics = {
        "accuracy" : accuracy_score(y_test , preds),
        "precision_weighted" : precision_score(y_true= y_test , y_pred= preds , average= "weighted"),
        "recall_weighted" : recall_score(y_test , preds , average="weighted"),
        "f1_weighted" : f1_score(y_test , preds , average= "weighted"),
        "f1_macro" : f1_score(y_test , preds , average= "macro"),
        "classification_report" : classification_report(y_test , preds)
    }
    return metrics

def evaluate_model(model , X_test:pd.DataFrame , y_test: pd.Series):
    # Validation of the Data
    validate_evaluation_data(X_test , y_test)
    # Generating Predictions
    preds = generate_predictions(model , X_test)

    # Calculating The metrics
    metrics = calculate_metrics(y_test , preds)

    return metrics


def plot_confusion_matrix(y_test:pd.Series , preds , labels = None , title = "Confusion Matrix" , cmap = "Blues"):

    # Compute the Confusion Matrix Array
    cm = confusion_matrix(y_test , preds)
    # Set up Display Object
    disp = ConfusionMatrixDisplay(confusion_matrix=cm , display_labels= labels)

    # Render Plot
    fig , axes = plt.subplots(figsize = (6,6))
    disp.plot(cmap= cmap , ax = axes , colorbar= True, values_format= "d")
    axes.set_title(title)
    plt.show()

    return fig , axes

def plot_feature_importance(model, title="Feature Importance"):

    if not model.is_fitted:
        raise ValueError("Model must be fitted before plotting feature importance")

    importances = model.model.feature_importances_
    feature_names = model.feature_names

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances})

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.barh(importance_df["feature"], importance_df["importance"])

    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    plt.tight_layout()
    plt.show()

    return fig, ax


    

