import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from src.config import TRAIN_DATA_PATH

def load_meddialog_data(file_path: Path = TRAIN_DATA_PATH) -> pd.DataFrame:
    """
    Load MedDialog dataset from a JSON file and convert it to a DataFrame.
    
    Args:
        file_path (Path): Path to the JSON file.
        
    Returns:
        pd.DataFrame: DataFrame containing patient_query and doctor_response.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at {file_path}")
        
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    records = []
    for dialogue in data:
        description = dialogue.get('description', '')
        utterances = dialogue.get('utterances', [])
        
        # Extract doctor response (usually the second utterance)
        doctor_response = ''
        # Simple heuristic: Look for the first response starting with "doctor:"
        # or just take the second utterance if available and it's not the patient again
        # The notebook logic was: if len >= 2 and utterances[1].startswith("doctor:")
        
        for utterance in utterances:
            if utterance.startswith("doctor:"):
                doctor_response = utterance[len("doctor:"):].strip()
                break
        
        # Fallback if loop didn't find "doctor:" prefix but there are utterances
        if not doctor_response and len(utterances) > 1:
             # Assuming alternating turns, 2nd is doctor
             if not utterances[1].startswith("patient:"):
                 doctor_response = utterances[1]

        records.append({
            'patient_query': description,
            'doctor_response': doctor_response
        })
        
    return pd.DataFrame(records)
