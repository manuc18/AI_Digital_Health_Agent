from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List

class ConversationSummarizer:
    def __init__(self, model_name: str = "t5-small"):
        """
        Initialize the ConversationSummarizer with T5.
        
        Args:
            model_name (str): Hugging Face model name.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading summarization model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def summarize(self, text: str) -> str:
        """
        Generate a summary for the given text.
        
        Args:
            text (str): Input text (conversation).
            
        Returns:
            str: Generated summary.
        """
        # T5 specific prefix
        input_text = "summarize: " + text
        
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"], 
                max_length=150, 
                min_length=40, 
                length_penalty=2.0, 
                num_beams=4, 
                early_stopping=True
            )
            
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
