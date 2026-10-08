from sentence_transformers import SentenceTransformer
import torch
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.utils.validation import check_is_fitted
import numpy as np


class TransformerExtractor(BaseEstimator , TransformerMixin):
    """
    PyTorch-native text feature extractor wrapping SentenceTransformer models
    for incident logs and enterprise text pipelines.
    """
    
    def __init__(self, text_columns: tuple = ("subject" , "description"), model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        self.text_columns = text_columns
        self.model_name = model_name
        self.batch_size = batch_size
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        Initializes the pretrained SentenceTransformer model.

        No model parameters are learned because the transformer
        is frozen. The fit method exists to comply with the
        Scikit-Learn estimator API.
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if X.empty:
            raise ValueError("Input DataFrame cannot be empty.")

        # Validate that the Batch size is positive and > 0
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        # Validate required text columns
        cols_to_check = list(self.text_columns)
        missing_cols = [col for col in cols_to_check if col not in X.columns]

        if missing_cols:
            raise KeyError(f"Missing text columns: {missing_cols}")

        # Determine device
        self.device_ = ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.model_ = SentenceTransformer( model_name_or_path= self.model_name, device=self.device_ , )

        except Exception as e:
            raise RuntimeError(
                f"Failed to load SentenceTransformer" 
                f"model '{self.model_name}': {e}") from e

        return self

    def transform(self, X: pd.DataFrame):
        """
        Combines the configured text columns and generates
        dense sentence embeddings.
        """

        # Ensure fit() has been called
        check_is_fitted(self, "model_")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if X.empty:
            raise ValueError("Input DataFrame cannot be empty.")

        # Validate required columns
        cols_to_check = list(self.text_columns)
        missing_cols = [col for col in cols_to_check if col not in X.columns]

        if missing_cols:
            raise KeyError(f"Missing text columns: {missing_cols}")
           

        # Combine subject + description into one text string
        text_data = (X[list(self.text_columns)].fillna("").astype(str).agg(" ".join, axis=1))

        sentences = text_data.tolist()

        try:
            embeddings = self.model_.encode(sentences, batch_size=self.batch_size, convert_to_tensor=False, device=self.device_)

        except Exception as e:
            raise RuntimeError(f"Failed to generate text embeddings: {e}") from e

        return np.asarray(embeddings)
    

        