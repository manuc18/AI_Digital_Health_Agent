import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
from src.config import CLINICAL_BERT_MODEL_NAME, MAX_SEQ_LENGTH, PROCESSED_DATA_DIR
from pathlib import Path

class SymptomExtractor:
    def __init__(self, model_name: str = None):
        """
        Initialize the SymptomExtractor with ClinicalBERT.
        
        Args:
            model_name (str): Hugging Face model name or path.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check if fine-tuned model exists
        finetuned_path = PROCESSED_DATA_DIR.parent / "models" / "fine_tuned_bert"
        if model_name is None:
            if finetuned_path.exists():
                print(f"Found fine-tuned model at {finetuned_path}")
                model_name = str(finetuned_path)
            else:
                model_name = CLINICAL_BERT_MODEL_NAME
                
        print(f"Loading model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def extract_features(self, texts: List[str]) -> torch.Tensor:
        """
        Extract embeddings from the input texts using ClinicalBERT.
        
        Args:
            texts (List[str]): List of input texts (patient queries).
            
        Returns:
            torch.Tensor: CLS token embeddings.
        """
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use the CLS token embedding (first token) as the sentence representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings

    def extract_symptoms(self, text: str) -> List[str]:
        """
        Extract symptoms and medical entities using a pre-trained NER pipeline.
        
        Args:
            text (str): Input text.
            
        Returns:
            List[str]: List of extracted entities (symptoms, diseases).
        """
        from transformers import pipeline
        
        # Load NER pipeline (cached)
        if not hasattr(self, 'ner_pipeline'):
            print("Loading NER pipeline (d4data/biomedical-ner-all)...")
            # We use a specific biomedical NER model
            self.ner_pipeline = pipeline(
                "token-classification", 
                model="d4data/biomedical-ner-all", 
                tokenizer="d4data/biomedical-ner-all",
                aggregation_strategy="simple", # Merges subwords into words
                device=0 if torch.cuda.is_available() else -1
            )
            
        results = self.ner_pipeline(text)
        
        # Filter for relevant entities if needed, or return all
        # Common tags in this model: B-Sign_symptom, I-Sign_symptom, B-Diagnostic_procedure, etc.
        # We will return the text of identified entities.
        entities = [result['word'] for result in results]
        return list(set(entities)) # Deduplicate
