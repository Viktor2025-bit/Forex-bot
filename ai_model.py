import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import os

class ForexModel:
    """
    Wrapper for XGBoost model to predict forex market movements.
    """
    
    def __init__(self, model_path='models/xgboost_forex.json', **kwargs):
        self.params = {
            'max_depth': kwargs.get('max_depth', 5),
            'learning_rate': kwargs.get('learning_rate', 0.05),
            'n_estimators': kwargs.get('n_estimators', 300),
            'objective': 'binary:logistic',
            'base_score': 0.5,
            'eval_metric': 'logloss',
            'n_jobs': -1,
            'random_state': 42
        }
        self.model = xgb.XGBClassifier(**self.params)
        self.model_path = model_path
        self._is_trained = False
        
    def train(self, X, y):
        """
        Train the model on features X and target y.
        """
        print("Training XGBoost model...")
        self.model.fit(X, y, verbose=True)
        self._is_trained = True
        print("Training complete.")
        
    def predict(self, X):
        """
        Predict signals for new data.
        Returns:
            np.array: Probabilities of class 1 (Buy/Up)
        """
        if not self._is_trained:
            # Try loading if not trained in this session
            if not self.load_model():
                raise RuntimeError("Model is neither trained nor loaded.")
                
        # Return probabilities for the positive class (1)
        return self.model.predict_proba(X)[:, 1]
    
    def save_model(self):
        """Save specific model to disk."""
        if not os.path.exists('models'):
            os.makedirs('models')
            
        self.model.get_booster().save_model(self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load model from disk."""
        if os.path.exists(self.model_path):
            self.model.load_model(self.model_path)
            self._is_trained = True
            print(f"Model loaded from {self.model_path}")
            return True
        return False
    
    def get_feature_importance(self):
        """Return feature importance map."""
        return self.model.feature_importances_
