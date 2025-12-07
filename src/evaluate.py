import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.config import TRAIN_DATA_PATH, PROCESSED_DATA_DIR
from src.data.loader import load_meddialog_data
from src.data.preprocessor import preprocess_dataframe
from src.data.labeler import HeuristicLabeler
from src.models.clinical_bert import SymptomExtractor
from src.models.classifier import SeverityClassifier
from pathlib import Path

def evaluate_model():
    print("Step 1: Loading and Preprocessing Data...")
    df = load_meddialog_data(TRAIN_DATA_PATH)
    df = preprocess_dataframe(df)
    
    print("Step 2: Generating Heuristic Labels...")
    labeler = HeuristicLabeler()
    df = labeler.label_dataframe(df)
    
    print("Step 3: Extracting Features (using Fine-tuned ClinicalBERT)...")
    extractor = SymptomExtractor()
    
    # Process in batches
    batch_size = 32
    all_features = []
    texts = df['patient_query_clean'].tolist()
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        features = extractor.extract_features(batch_texts)
        all_features.append(features.cpu())
        
    X = torch.cat(all_features, dim=0).numpy()
    y = df['severity'].values
    
    print("Step 4: Splitting Data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Step 5: Training Classifier...")
    classifier = SeverityClassifier(model_type='logistic')
    classifier.train(X_train, y_train)
    
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    
    # Train Metrics
    y_train_pred = classifier.predict(X_train)
    print("\n--- Training Set Metrics ---")
    print(classification_report(y_train, y_train_pred, target_names=['Low', 'Medium', 'High']))
    print("Confusion Matrix (Train):")
    print(confusion_matrix(y_train, y_train_pred))
    
    # Test Metrics
    y_test_pred = classifier.predict(X_test)
    print("\n--- Test Set Metrics ---")
    print(classification_report(y_test, y_test_pred, target_names=['Low', 'Medium', 'High']))
    print("Confusion Matrix (Test):")
    print(confusion_matrix(y_test, y_test_pred))
    
    # Save report
    report_path = PROCESSED_DATA_DIR.parent / "reports"
    report_path.mkdir(parents=True, exist_ok=True)
    with open(report_path / "metrics.txt", "w") as f:
        f.write("EVALUATION REPORT\n")
        f.write("="*50 + "\n")
        f.write("\n--- Test Set Metrics ---\n")
        f.write(classification_report(y_test, y_test_pred, target_names=['Low', 'Medium', 'High']))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_test_pred)))
        
    print(f"\nReport saved to {report_path / 'metrics.txt'}")

if __name__ == "__main__":
    evaluate_model()
