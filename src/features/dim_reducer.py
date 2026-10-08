import pandas as pd
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator , TransformerMixin
from sklearn.utils.validation import check_is_fitted

class EmbeddingReduction(BaseEstimator , TransformerMixin):
    def __init__(self , n_components = 30 , random_state = 42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self , X , y=None):
        self.pca_ = PCA(n_components= self.n_components , random_state= self.random_state)
        self.pca_.fit(X)
        
        variance = self.pca_.explained_variance_ratio_.sum()
        print(f"Fitted PCA: Reduced to {self.n_components} Components.")
        print(f"Retained Variance Explained is {variance * 100:.2f}%")

        return self

    def transform(self , X):
        check_is_fitted(self ,"pca_")
        return self.pca_.transform(X)