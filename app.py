import streamlit as st
import pandas as pd
import torch
import numpy as np
from src.config import TRAIN_DATA_PATH
from src.data.loader import load_meddialog_data
from src.data.preprocessor import preprocess_dataframe
from src.data.labeler import HeuristicLabeler
from src.models.clinical_bert import SymptomExtractor
from src.models.classifier import SeverityClassifier
from src.models.medication import MedicationAdvisor
from src.models.summarizer import ConversationSummarizer

# Page Config
st.set_page_config(
    page_title="AI Digital Health Agent",
    page_icon="🏥",
    layout="wide"
)

@st.cache_resource
def load_models_and_data():
    """Load all models and data once and cache them."""
    with st.spinner("Loading models and knowledge base..."):
        # 1. Load Data (Knowledge Base)
        df = load_meddialog_data(TRAIN_DATA_PATH)
        df = preprocess_dataframe(df)
        
        # Label data for classifier training (if not pre-trained)
        labeler = HeuristicLabeler()
        df = labeler.label_dataframe(df)
        
        # 2. Symptom Extractor
        extractor = SymptomExtractor()
        
        # 3. Extract Features for KB (This might take a moment)
        # Use FULL dataset for better matching
        kb_df = df.copy()
        kb_features = extractor.extract_features(kb_df['patient_query_clean'].tolist()).cpu()
        
        # 4. Train Classifier (Quickly on subset)
        classifier = SeverityClassifier(model_type='logistic')
        classifier.train(kb_features.numpy(), kb_df['severity'].values)
        
        # 5. Fit Medication Advisor
        # Pass ORIGINAL doctor responses for display
        advisor = MedicationAdvisor()
        advisor.fit(kb_features.numpy(), kb_df['doctor_response'].tolist())
        
        # 6. Summarizer
        summarizer = ConversationSummarizer()
        
        return extractor, classifier, advisor, summarizer, kb_df

def main():
    st.title("🏥 AI-Powered Digital Health Agent")
    st.markdown("""
    **Conversational AI for Symptom Analysis & Doctor Connectivity**
    
    Enter your symptoms below to get an AI assessment.
    """)
    
    # Load resources
    try:
        extractor, classifier, advisor, summarizer, kb_df = load_models_and_data()
        st.success("System Ready!")
    except Exception as e:
        st.error(f"Error loading system: {e}")
        return

    # User Input
    with st.form("symptom_form"):
        user_query = st.text_area("Describe your symptoms:", height=100, 
                                placeholder="e.g., I have a severe headache and high fever for 2 days.")
        submitted = st.form_submit_button("Analyze")

    if submitted and user_query:
        col1, col2 = st.columns([1, 1])
        
        # 1. Feature Extraction
        with st.spinner("Analyzing symptoms..."):
            features = extractor.extract_features([user_query]).cpu().numpy()
        
        # 2. Severity Classification
        severity_pred = classifier.predict(features)[0]
        severity_map = {0: "Low", 1: "Medium", 2: "High"}
        severity_label = severity_map.get(severity_pred, "Unknown")
        
        with col1:
            st.subheader("Severity Assessment")
            if severity_label == "High":
                st.error(f"⚠️ Severity: **{severity_label}**")
                st.warning("Please consult a doctor immediately.")
            elif severity_label == "Medium":
                st.warning(f"⚠️ Severity: **{severity_label}**")
            else:
                st.info(f"ℹ️ Severity: **{severity_label}**")
                
        # 3. Medication Guidance
        # Get recommendation and index to find the matched query
        # Note: MedicationAdvisor currently returns just the response. 
        # We can update it to return index, or just trust the response.
        # For now, let's just show the response.
        recommendation = advisor.predict(features)[0]
        
        with col2:
            st.subheader("AI Medical Guidance")
            st.info(f"**Based on similar cases:**\n\n{recommendation}")
            
        # 4. Summarization
        st.divider()
        st.subheader("Consultation Summary")
        with st.spinner("Generating summary..."):
            conversation = f"Patient: {user_query} Doctor: {recommendation}"
            summary = summarizer.summarize(conversation)
            st.write(summary)
            
    # Sidebar - Debug Info
    with st.sidebar:
        st.header("System Info")
        st.write(f"Knowledge Base Size: {len(kb_df)} records")
        st.write("Models Loaded:")
        st.code("ClinicalBERT\nLogisticRegression\nT5-Small")

if __name__ == "__main__":
    main()
