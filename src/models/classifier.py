from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class RandomForestModel:

    def __init__(self, n_estimators=200, max_depth=None, max_features="sqrt", min_samples_split=2, class_weight=None, random_state=42):
        
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.random_state = random_state

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            class_weight=self.class_weight,
            random_state=self.random_state)
    
        self.is_fitted = False
       

    def train(self , X_train: pd.DataFrame , y_train: pd.Series):
        """
       Train the Random Forest classifier after validating the training data.
        """
        # Validate Input types
        if not isinstance(X_train , pd.DataFrame):
            raise TypeError("X_train Data Must be a Pandas DataFrame")

        if not isinstance(y_train , pd.Series):
            raise TypeError("y_train Must be a Pandas Series")

        # Validate that training data is not empty
        if (len(X_train) == 0) or (len(y_train)) == 0:
            raise ValueError("Training Set cannot be Empty")

         # Validate that X_train contains features
        if (X_train.shape[1] == 0):
            raise ValueError("X_train cannot have  zero Features")

        # Ensuring that X_train and y_train have similar rows
        assert (len(X_train) == len(y_train)) , (f"Shape Mismatch: X_train has {len(X_train)} samples,"
                                                 f"but y_train has {len(y_train)}")

        self.feature_names = X_train.columns.to_list()
        
        # Fitting the RandomForest Model
        self.model.fit(X_train , y_train)
        self.is_fitted = True
       
        return self.model


    def predict(self , X_test: pd.DataFrame):
        
        # Validating Input Data type
        if not isinstance(X_test , pd.DataFrame):
            raise TypeError("X_test must be a Pandas DataFrame")

        # Validating that the Testing Set is not Empty
        if len(X_test) == 0:
            raise ValueError("Testing Set cannot be Empty")

        # Validate that X_test contain features
        if (X_test.shape[1]) == 0:
            raise ValueError("X_test Cannot have Zero Features")

        # Validating that Training phase should happen first
        if not self.is_fitted:
            raise RuntimeError("This Model Instance is not Fitted yet. Call 'train()' "
                               "With Appropriate Data Before Using 'predict'. ")

        # Validate that X_train and X_test have the same features
        if X_test.columns.to_list() != self.feature_names:
            raise ValueError("X_train And X_test Should Have The Same Features")


        # Perform Predicitons on X_test
        preds = self.model.predict(X_test)

        return preds
    

        