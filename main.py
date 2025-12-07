import pandas as pd
import numpy as np
import torch
from src.config import TRAIN_DATA_PATH, PROCESSED_DATA_DIR
from src.data.loader import load_meddialog_data
from src.data.preprocessor import preprocess_dataframe
from src.data.eda import EDA
from src.data.labeler import HeuristicLabeler
from src.models.clinical_bert import SymptomExtractor
from src.models.classifier import SeverityClassifier
from src.models.medication import MedicationAdvisor
from src.models.summarizer import ConversationSummarizer

def main():
    # 1. Data Loading & Cleaning
    print("Step 1: Loading Data...")
    try:
        df = load_meddialog_data(TRAIN_DATA_PATH)
        print(f"Loaded {len(df)} records.")
    except FileNotFoundError:
        print(f"File not found at {TRAIN_DATA_PATH}. Please ensure data is present.")
        return

    print("Step 2: Preprocessing Data...")
    df = preprocess_dataframe(df)
    print(f"Data after cleaning: {len(df)} records.")
    
    # 2. EDA
    print("Step 3: Running EDA...")
    eda = EDA(df)
    eda.run()
    
    # 3. Labeling
    print("Step 4: Generating Heuristic Labels...")
    labeler = HeuristicLabeler()
    df = labeler.label_dataframe(df)
    
    # 4. Feature Extraction (using Fine-tuned ClinicalBERT)
    print("Step 5: Extracting Features with ClinicalBERT...")
    # This will automatically load the fine-tuned model if it exists
    extractor = SymptomExtractor()
    
    # Process in batches to avoid OOM
    batch_size = 32
    all_features = []
    texts = df['patient_query_clean'].tolist()
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        features = extractor.extract_features(batch_texts)
        all_features.append(features.cpu())
        
    features = torch.cat(all_features, dim=0)
    print(f"Extracted features shape: {features.shape}")
    
    # 5. Train Severity Classifier
    print("Step 6: Training Severity Classifier...")
    classifier = SeverityClassifier(model_type='logistic')
    
    X = features.numpy()
    y = df['severity'].values
    
    # Split into train/test for evaluation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    classifier.train(X_train, y_train)
    print("Evaluation on Test Set:")
    classifier.evaluate(X_test, y_test)
    
    # 6. Medication Guidance
    print("Step 7: Medication Guidance (Retrieval)...")
    advisor = MedicationAdvisor()
    advisor.fit(X, df['doctor_response'].tolist())
    
    # Test on a sample query
    test_idx = 0
    test_query_emb = X[test_idx].reshape(1, -1)
    recommendation = advisor.predict(test_query_emb)
    print(f"\nQuery: {df.iloc[test_idx]['patient_query'][:100]}...")
    print(f"Recommended Response: {recommendation[0][:100]}...")
    
    # 7. Summarization
    print("Step 8: Conversation Summarization (T5)...")
    summarizer = ConversationSummarizer()
    conversation = f"Patient: {df.iloc[test_idx]['patient_query']} Doctor: {df.iloc[test_idx]['doctor_response']}"
    summary = summarizer.summarize(conversation)
    print(f"\nOriginal Conversation Length: {len(conversation)}")
    print(f"Summary: {summary}")
    
    print("End-to-End Pipeline completed successfully!")

if __name__ == "__main__":
    main()
