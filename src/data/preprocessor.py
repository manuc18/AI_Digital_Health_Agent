import re
import pandas as pd
from typing import Optional

def clean_text(text: str) -> str:
    """
    Clean text by converting to lowercase, removing special characters,
    and normalizing whitespace.
    
    Args:
        text (str): Input text.
        
    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    # Remove special characters but keep punctuation that might be useful for structure
    # The notebook used: re.sub(r'[^a-z0-9\s.,?!]', '', text)
    text = re.sub(r'[^a-z0-9\s.,?!]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning and filtering to the DataFrame.
    
    Args:
        df (pd.DataFrame): Raw DataFrame.
        
    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    # Clean text columns
    df['patient_query_clean'] = df['patient_query'].apply(clean_text)
    df['doctor_response_clean'] = df['doctor_response'].apply(clean_text)
    
    # Remove empty rows based on clean text
    df.dropna(subset=['patient_query_clean', 'doctor_response_clean'], inplace=True)
    df = df[df['patient_query_clean'] != '']
    df = df[df['doctor_response_clean'] != '']
    
    return df
