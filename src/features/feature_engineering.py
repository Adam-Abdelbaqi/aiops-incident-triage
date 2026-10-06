import pandas as pd
from sklearn.base import TransformerMixin , BaseEstimator

class TabularExtractor(BaseEstimator , TransformerMixin):
    def __init__(self , cols_to_drop = None , date_col = "created_date"):

        if cols_to_drop is None:
            self.cols_to_drop = ["status", "category", "satisfaction_rating", "resolved_date", "ticket_id", "resolution"]
        else:
            self.cols_to_drop = cols_to_drop

        self.date_col = date_col
        self.sub_category_map_ = {}
        self.department_map_ = {}

    def fit(self , X: pd.DataFrame , y = None):
        """Learns the Feature Encodings from the Training set"""

        if "sub_category" in X.columns:
            # Calculate The Percentage of each sub_category appears in the training set
            self.sub_category_map_ = X["sub_category"].value_counts(normalize= True).to_dict()

        if "department" in X.columns:
            self.department_map_ = X["department"].value_counts(normalize=True).to_dict()

        return self

    def transform(self , X:pd.DataFrame):
        """
        Applies Transformation  (dropping , rolling , encoding)
        """
        X_transformed = X.copy()
        # Drop redundant columns that either cause data leakage or has no predictive power
        cols_to_remove = [col for col in self.cols_to_drop if col in X_transformed.columns]
        X_transformed = X_transformed.drop(columns=cols_to_remove)

        # 2. Apply Sub-Category And Department Frequency Encoding
        if 'sub_category' in X_transformed.columns:
            # Map the learned frequencies. Fill unseen test categories with 0 to prevent NaNs.
            X_transformed['sub_category_encoded'] = X_transformed['sub_category'].map(self.sub_category_map_).fillna(0)
            X_transformed = X_transformed.drop(columns=['sub_category'])

        if 'department' in X_transformed.columns:
            X_transformed["department_encoded"] = X_transformed["department"].map(self.department_map_).fillna(0)
            X_transformed = X_transformed.drop(columns=["department"])
        

        # 3. Temporal Engineering & Target Leakage Prevention
        if self.date_col in X_transformed.columns:
            X_transformed[self.date_col] = pd.to_datetime(X_transformed[self.date_col])
            dates_only = X_transformed[self.date_col].dt.date
            
            # Calculate Global Velocity: Trailing 3-day volume
            daily_volume = X_transformed.groupby(dates_only).size()
            
            # .shift(1) pushes the sums down one day so today's prediction only uses yesterday's totals
            trailing_3d_volume = daily_volume.rolling(window=3, min_periods=1).sum().shift(1).fillna(0)
            
            # Map the shifted rolling metric back to the dataframe
            X_transformed['global_volume_past_3d'] = dates_only.map(trailing_3d_volume)

            # Extract basic numeric temporal features
            X_transformed['created_dayofweek'] = X_transformed[self.date_col].dt.dayofweek
            X_transformed['is_weekend'] = X_transformed['created_dayofweek'].isin([5, 6]).astype(int)
            
            # Drop the raw datetime column as LightGBM requires numeric/encoded features
            X_transformed = X_transformed.drop(columns=[self.date_col])

        return X_transformed

    