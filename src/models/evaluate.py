from sklearn.metrics import precision_score , classification_report , accuracy_score , recall_score ,f1_score , confusion_matrix , ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_is_fitted


def generate_predictions(model , X_test:pd.DataFrame):
    preds = model.predict(X_test)

    return preds

def calculate_metrics(y_test:pd.Series , preds):
    metrics = {
        "accuracy" : accuracy_score(y_test , preds),
        "precision_weighted" : precision_score(y_true= y_test , y_pred= preds , average= "weighted" , zero_division=0),
        "recall_weighted" : recall_score(y_test , preds , average="weighted" , zero_division=0),
        "f1_weighted" : f1_score(y_test , preds , average= "weighted" , zero_division=0),
        "f1_macro" : f1_score(y_test , preds , average= "macro" , zero_division= 0),
        "classification_report" : classification_report(y_test , preds , output_dict= True , zero_division=0)
    }
    return metrics

def evaluate_model(model , X_test:pd.DataFrame , y_test: pd.Series):
    # Generating Predictions
    preds = generate_predictions(model , X_test)

    # Calculating The metrics
    metrics = calculate_metrics(y_test , preds)

    return metrics




def plot_confusion_matrix(y_test:pd.Series , preds , title = "Confusion Matrix" , cmap = "Blues"):

    # Compute the Confusion Matrix Array
    cm = confusion_matrix(y_test , preds)
    # Set up Display Object
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Render Plot
    fig , axes = plt.subplots(figsize = (6,6))
    disp.plot(cmap= cmap , ax = axes , colorbar= True, values_format= "d")
    axes.set_title(title)
    plt.show()

    return fig , axes

def plot_feature_importance(model, title="Feature Importance", feature_names=None):
    """Plot importances from a fitted RandomForestClassifier or fitted pipeline.

    If a pipeline is supplied, generic names are used unless feature_names are
    explicitly provided, because the combined text/tabular array has no native
    DataFrame column names.
    """
    classifier = model
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        classifier = model.named_steps["classifier"]

    check_is_fitted(classifier, attributes=["feature_importances_"])
    importances = classifier.feature_importances_

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    if len(feature_names) != len(importances):
        raise ValueError("The number of feature names must match the number of importances.")

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}).sort_values(by="importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df["feature"], importance_df["importance"])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig, ax


    

