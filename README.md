# AIOps Incident Triage

## Overview

A machine learning project for classifying IT support incidents by priority using text descriptions and structured categorical features.

The pipeline combines semantic representations of incident subjects and descriptions with categorical metadata to predict one of four priority levels: **Critical, High, Medium, or Low**.

The project focuses on practical machine learning engineering, including feature extraction, scikit-learn pipelines, chronological data splitting, model evaluation, and automated testing.

## Problem Statement

IT support teams receive incidents with varying levels of urgency. Automatically classifying incident priority can help organize incoming tickets and support more efficient triage.

This project formulates incident triage as a supervised multiclass classification problem. The model learns from historical incident records to predict the priority of an incident from its textual content and categorical attributes.

## Architecture

The pipeline consists of two parallel feature-extraction branches whose outputs are combined before classification.

1. **Text feature extraction**
   - Uses the pretrained `all-MiniLM-L6-v2` SentenceTransformer model.
   - Generates semantic embeddings from the incident `subject` and `description`.
   - Processes text in batches.

2. **Tabular feature extraction**
   - Selects the `category` and `sub_category` features.
   - Applies `OneHotEncoder` with `handle_unknown="ignore"` to support previously unseen categories.

3. **Feature combination**
   - Uses scikit-learn's `FeatureUnion` to combine text embeddings and encoded categorical features.

4. **Classification**
   - Uses Logistic Regression to predict the incident priority.
   - Uses `C=100`, `max_iter=2000`, and `random_state=42`.

### Architecture diagram

```text
Incident Data
     |
     +---------------------------+
     |                           |
     v                           v
Subject + Description      Category + Subcategory
     |                           |
     v                           v
SentenceTransformer         One-Hot Encoding
(all-MiniLM-L6-v2)                |
     |                           |
     +-------------+-------------+
                   |
                   v
              FeatureUnion
                   |
                   v
          Logistic Regression
                (C=100)
                   |
                   v
        Predicted Incident Priority
     Critical / High / Medium / Low
```

## Dataset and Preprocessing

The dataset contains historical IT support incident records, including textual descriptions, categorical attributes, and priority labels.

The target variable is `priority`, with four classes: Critical, High, Medium, and Low.

The project uses preprocessing and validation steps to prepare the data for model training. Columns that could introduce target leakage or depend on post-resolution information are excluded from the predictive feature set.

The data is split chronologically into training and testing sets to better reflect the task of predicting priorities for later incidents.

## Model Evaluation

The final model was evaluated using accuracy, weighted F1-score, macro F1-score, and per-class precision and recall.

| Metric | Training | Test |
|---|---:|---:|
| Accuracy | 86.24% | 81.91% |
| Weighted F1-score | 86.21% | 81.89% |
| Macro F1-score | 85.99% | 81.20% |

### Test-set performance by class

| Priority | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Critical | 0.81 | 0.78 | 0.79 |
| High | 0.80 | 0.77 | 0.78 |
| Low | 0.83 | 0.83 | 0.83 |
| Medium | 0.83 | 0.85 | 0.84 |

The model achieved an overall test accuracy of approximately 81.9%. Macro F1-score provides an additional view of performance across the four priority classes rather than allowing the more frequent classes to dominate the overall evaluation.

**Evaluation note:** The test set was examined during earlier feature and model comparisons. These results should therefore be treated as exploratory test performance rather than a completely untouched estimate of generalization.

## Testing

Automated tests are implemented using `pytest`.

The current test suite covers:

- Data-processing behavior.
- Pipeline construction and classifier configuration.
- Model fitting and prediction.
- Prediction behavior before fitting.
- Handling previously unseen categorical values.
- End-to-end pipeline fitting and prediction.

Run the full test suite from the project root:

```bash
python -m pytest
```

Current result: **15 tests passed**.

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- SentenceTransformers
- PyTorch
- pytest

## Limitations and Future Improvements

- Evaluate the model on a future holdout dataset that has not been used for model or feature selection.
- Investigate additional text representations and feature-engineering strategies.
- Monitor performance across priority classes, especially high-urgency incidents.
- Evaluate the inference workflow using new incident records.
- Consider adding a lightweight user interface or API for interactive predictions.

## Conclusion

AIOps Incident Triage demonstrates a machine learning workflow that combines semantic text embeddings with structured categorical data for multiclass incident-priority classification. It integrates feature extraction, a unified scikit-learn pipeline, chronological evaluation, and automated testing into a reproducible project structure.
