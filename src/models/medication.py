import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import List

class MedicationAdvisor:
    def __init__(self):
        """
        Initialize the MedicationAdvisor using Nearest Neighbors for retrieval.
        """
        self.index = NearestNeighbors(n_neighbors=1, metric='cosine')
        self.responses = []
        self.is_fitted = False
        
    def fit(self, embeddings: np.ndarray, responses: List[str]):
        """
        Fit the retrieval model with historical data.
        
        Args:
            embeddings (np.ndarray): Matrix of query embeddings (Knowledge Base).
            responses (List[str]): List of doctor responses corresponding to queries.
        """
        print("Fitting Medication Advisor (Retrieval System)...")
        self.responses = responses
        self.index.fit(embeddings)
        self.is_fitted = True
        
    def predict(self, query_embeddings: np.ndarray) -> List[str]:
        """
        Retrieve the most relevant doctor response for new queries.
        
        Args:
            query_embeddings (np.ndarray): Embeddings of the new patient queries.
            
        Returns:
            List[str]: Recommended responses.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
            
        # Find the nearest neighbor for each query
        distances, indices = self.index.kneighbors(query_embeddings)
        
        recommendations = []
        for idx in indices.flatten():
            recommendations.append(self.responses[idx])
            
        return recommendations
