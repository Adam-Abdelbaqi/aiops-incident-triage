import pandas as pd
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator , TransformerMixin

class EmbeddingReduction(BaseEstimator , TransformerMixin):
    def __init__(self , n_components = 30 , random_state = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components= self.n_components , random_state= self.random_state)

    def fit(self , X , y=None):
        self.pca.fit(X)
        
        variance = self.pca.explained_variance_ratio_.sum()
        print(f"Fitted PCA: Reduced to {self.n_components} Components.")
        print(f"Retained Variance Explained is {variance * 100:.2f}%")

        return self

    def transform(self , X , y = None):
        return self.pca.transform(X)