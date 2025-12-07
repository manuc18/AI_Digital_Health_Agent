from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import numpy as np
import pickle
from pathlib import Path

class SeverityClassifier:
    def __init__(self, model_type: str = 'logistic'):
        """
        Initialize the SeverityClassifier.
        
        Args:
            model_type (str): 'logistic' or 'xgboost' (using GradientBoostingClassifier for simplicity).
        """
        if model_type == 'logistic':
            self.model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        elif model_type == 'xgboost':
            self.model = GradientBoostingClassifier()
        else:
            raise ValueError("model_type must be 'logistic' or 'xgboost'")
            
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the classifier.
        
        Args:
            X (np.ndarray): Feature matrix (e.g., BERT embeddings).
            y (np.ndarray): Labels (0: Low, 1: Medium, 2: High).
        """
        print(f"Training {self.model.__class__.__name__}...")
        self.model.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict severity.
        
        Args:
            X (np.ndarray): Feature matrix.
            
        Returns:
            np.ndarray: Predicted labels.
        """
        return self.model.predict(X)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray):
        """
        Evaluate the model.
        """
        y_pred = self.predict(X)
        print(classification_report(y, y_pred, target_names=['Low', 'Medium', 'High']))
        
    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load(self, path: Path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
