# AIOps Incident Triage

## Overview

A machine learning project that classifies IT support incidents into four priority levels: **Critical, High, Medium, and Low**.

The project combines semantic text embeddings generated from incident subjects and descriptions with structured categorical features. A scikit-learn pipeline integrates feature extraction and Logistic Regression into a unified classification workflow.

The project demonstrates practical skills in NLP feature extraction, supervised machine learning, chronological validation, model evaluation, hyperparameter tuning, and automated testing.

## Problem Statement

IT support teams receive incidents with different levels of urgency. Correctly identifying incident priority can help teams organize incoming tickets and direct attention toward critical issues.

This project formulates incident triage as a supervised multiclass classification problem. It uses historical incident information to predict the priority assigned to each ticket.

The four target classes are:

- Critical
- High
- Medium
- Low

## Model Architecture

The pipeline has two parallel feature-extraction branches.

### 1. Text feature extraction

- Uses the pretrained `all-MiniLM-L6-v2` SentenceTransformer model.
- Combines the `subject` and `description` fields into a single text input for each incident.
- Generates dense semantic embeddings in batches.
- Uses the pretrained transformer as a frozen feature extractor.

### 2. Categorical feature extraction

- Selects `category` and `sub_category`.
- Applies scikit-learn's `OneHotEncoder`.
- Uses `handle_unknown="ignore"` to support categories not encountered during fitting.

### 3. Feature integration and classification

- Combines the text and categorical feature branches using scikit-learn's `FeatureUnion`.
- Trains a Logistic Regression classifier on the combined feature representation.

**Classifier configuration**

| Parameter | Value |
|---|---|
| Model | Logistic Regression |
| Regularization parameter (`C`) | 100 |
| Maximum iterations | 2,000 |
| Random state | 42 |

### Architecture diagram

```text
                 Incident Data
                      |
           +----------+----------+
           |                     |
           v                     v
 Subject + Description      Category + Subcategory
           |                     |
           v                     v
 SentenceTransformer        One-Hot Encoder
 all-MiniLM-L6-v2                |
           |                     |
           +----------+----------+
                      |
                      v
                  FeatureUnion
                      |
                      v
              Logistic Regression
                   C = 100
                      |
                      v
             Predicted Priority
       Critical / High / Medium / Low
```

## Project Structure

```text
aiops-incident-triage/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── EDA.ipynb
│   └── multimodal_baseline.ipynb
├── src/
│   ├── data.py
│   ├── pipeline.py
│   ├── features/
│   │   ├── feature_extraction.py
│   │   ├── feature_engineering.py
│   │   └── dim_reducer.py
│   ├── preprocessing/
│   │   └── preprocess.py
│   └── models/
│       └── evaluate.py
├── tests/
│   ├── test_models.py
│   ├── test_pipeline.py
│   └── test_processing.py
├── main.py
├── tuning.py
├── requirements.txt
└── README.md
```

## Dataset and Preprocessing

The project uses incident records containing textual information, categorical attributes, timestamps, and priority labels.

Place the cleaned dataset at:

```text
data/raw/tickets_clean.csv
```

The dataset is not assumed to be included in the repository. Ensure that the CSV is available at this location before running the training or tuning scripts.

The preprocessing workflow includes data loading, date conversion, chronological ordering, missing-value inspection, and column-type conversion.

Features that could introduce target leakage or rely on information available only after incident resolution are excluded from model training. The target column, `priority`, is separated from the input features.

## Data Splitting Strategy

The dataset is sorted chronologically using the incident creation timestamp and divided into:

- **80% training data:** earlier incidents used to fit the model.
- **20% test data:** later incidents used to evaluate model performance.

This strategy better reflects the intended use case of predicting priorities for future incidents than a random split would.

Hyperparameter tuning uses a further chronological split within the training data. The test set should not be used to select hyperparameters.

## Model Evaluation

The model is evaluated using accuracy, weighted F1-score, macro F1-score, and per-class precision, recall, and F1-score.

### Overall performance

| Metric | Training | Test |
|---|---:|---:|
| Accuracy | 86.24% | 81.91% |
| Weighted F1-score | 86.21% | 81.89% |
| Macro F1-score | 85.99% | 81.20% |

### Test performance by priority class

| Priority | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Critical | 0.81 | 0.78 | 0.79 |
| High | 0.80 | 0.77 | 0.78 |
| Low | 0.83 | 0.83 | 0.83 |
| Medium | 0.83 | 0.85 | 0.84 |

The test accuracy is approximately 81.9%, with a macro F1-score of approximately 81.2%. Macro F1 gives equal weight to each class, making it useful for examining performance across priority levels.

A confusion matrix is also generated by the training and evaluation script to help identify which priority classes are most frequently confused.

**Evaluation limitation:** The test set was examined during earlier model and feature comparisons. The reported test metrics should therefore be treated as exploratory results, not as an unbiased final estimate of generalization to completely unseen data.

## Hyperparameter Tuning

The `tuning.py` script evaluates candidate values of the Logistic Regression regularization parameter `C`.

The tuning workflow:

1. Validates the input training data.
2. Creates a chronological fitting/validation split within the training set.
3. Fits a separate pipeline for each candidate `C`.
4. Evaluates accuracy, weighted F1-score, macro F1-score, and Critical-class recall.
5. Ranks the candidates by validation macro F1-score.

The current selected configuration uses `C=100`, based on the experiments conducted so far.

## Installation and Usage

### Prerequisites

- Python installed on your system.
- Git, if cloning the repository.
- Access to the cleaned dataset at the expected path.

### 1. Clone the repository

```bash
git clone https://github.com/Adam-Abdelbaqi/aiops-incident-triage.git
cd aiops-incident-triage
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the dataset

Place the cleaned CSV file at:

```text
data/raw/tickets_clean.csv
```

The training and tuning scripts expect this file to exist at that location.

### 5. Train and evaluate the model

From the repository root, run:

```powershell
python main.py
```

The script loads and preprocesses the data, creates the chronological split, trains the pipeline, reports training and test metrics, prints the classification report, and displays the test confusion matrix.

The first run may download the pretrained SentenceTransformer model, requiring an internet connection.

### 6. Tune the hyperparameter

```powershell
python tuning.py
```

This evaluates the configured candidate values of `C` using the chronological validation split and reports the results.

### 7. Run automated tests

```powershell
python -m pytest
```

The test suite covers data processing, pipeline construction, model fitting and prediction, handling unseen categorical values, and end-to-end pipeline behavior.

The previously verified test result was **15 passed**. Run the suite again after any code or dependency changes to confirm the current result.

## Technologies

- **Python** — implementation and workflow orchestration
- **pandas and NumPy** — data manipulation and numerical operations
- **scikit-learn** — preprocessing, feature integration, Logistic Regression, and evaluation
- **SentenceTransformers** — semantic text embeddings
- **PyTorch** — transformer model execution
- **matplotlib** — confusion matrix visualization
- **pytest** — automated testing
- **Jupyter** — exploratory data analysis and experiments

## Limitations and Future Improvements

- Evaluate the final model on a completely untouched, future holdout dataset.
- Investigate additional text representations and feature-engineering approaches.
- Analyze errors by priority class, with particular attention to Critical incidents.
- Evaluate inference behavior on new incident records and monitor prediction quality.
- Consider a lightweight interface for interactive predictions once the core workflow is stable.

## Conclusion

AIOps Incident Triage demonstrates an end-to-end machine learning workflow for multiclass incident-priority classification. By combining pretrained semantic text embeddings with categorical features in a scikit-learn pipeline, the project integrates NLP feature extraction, supervised classification, chronological evaluation, hyperparameter tuning, and automated testing.