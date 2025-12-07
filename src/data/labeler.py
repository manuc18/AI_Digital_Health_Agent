import pandas as pd
from typing import List

class HeuristicLabeler:
    def __init__(self):
        self.high_severity_keywords = [
            'emergency', 'severe', 'chest pain', 'breathing difficulty', 'unconscious', 
            'bleeding', 'stroke', 'heart attack', 'suicide', 'trauma', 'high fever',
            'pneumonia', 'covid', 'coronavirus'
        ]
        self.medium_severity_keywords = [
            'pain', 'fever', 'infection', 'flu', 'vomiting', 'diarrhea', 'dizziness', 
            'migraine', 'rash', 'swelling', 'fracture', 'burn'
        ]
        # Low severity is the default if no keywords match
        
    def get_severity(self, text: str) -> int:
        """
        Determine severity based on keywords.
        0: Low, 1: Medium, 2: High
        """
        text = text.lower()
        for keyword in self.high_severity_keywords:
            if keyword in text:
                return 2
        for keyword in self.medium_severity_keywords:
            if keyword in text:
                return 1
        return 0
        
    def label_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a 'severity' column to the DataFrame.
        """
        print("Applying heuristic labeling...")
        df['severity'] = df['patient_query_clean'].apply(self.get_severity)
        print("Label distribution:")
        print(df['severity'].value_counts())
        return df
