from sentence_transformers import SentenceTransformer
import torch
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class TransformerExtractor(BaseEstimator , TransformerMixin):
    """
    PyTorch-native text feature extractor wrapping SentenceTransformer models
    for incident logs and enterprise text pipelines.
    """
    text_col = ["description" , "subject"]
    def __init__(self, text_column: str = text_col, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        self.text_column = text_column
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            raise ValueError(f"Failed to load Sentence Transformer model {self.model_name}") from e

    def fit(self, X: pd.DataFrame, y=None):
        """
        The pre-trained transformer is frozen, so no parameters are learned here.
        We simply return self to comply with Scikit-Learn pipeline standards.
        """
        return self

    def transform(self, X: pd.DataFrame):
        """
        Extracts the target text column from X and generates dense vector embeddings.
        """
        if self.text_column not in X.columns:
            raise KeyError(f"Column '{self.text_column}' not found in the input DataFrame.")

        # Isolate the feature column and handle any missing text
        sentences = X[self.text_column].fillna("").astype(str).tolist()

        try:
            # Note: convert_to_tensor=False outputs a NumPy array, 
            # which is strictly required for the Scikit-Learn PCA step next.
            embeddings = self.model.encode(
                sentences, 
                batch_size=self.batch_size, 
                convert_to_tensor=False, 
                device=self.device
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate text embeddings: {e}") from e
            
        return embeddings

    

        