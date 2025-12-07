import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM, 
    DataCollatorForLanguageModeling, 
    Trainer, 
    TrainingArguments
)
from src.config import CLINICAL_BERT_MODEL_NAME, PROCESSED_DATA_DIR, TRAIN_DATA_PATH
from src.data.loader import load_meddialog_data
from src.data.preprocessor import preprocess_dataframe
from pathlib import Path

class ClinicalBertFinetuner:
    def __init__(self, model_name: str = CLINICAL_BERT_MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.output_dir = PROCESSED_DATA_DIR.parent / "models" / "fine_tuned_bert"

    def prepare_dataset(self, texts):
        inputs = self.tokenizer(
            texts, 
            truncation=True, 
            padding="max_length", 
            max_length=128, 
            return_tensors="pt"
        )
        
        # Create a Dataset object
        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            
            def __getitem__(self, idx):
                return {key: val[idx] for key, val in self.encodings.items()}
            
            def __len__(self):
                return len(self.encodings.input_ids)
                
        return TextDataset(inputs)

    def train(self, epochs: int = 3, batch_size: int = 8):
        print("Loading and preprocessing data for fine-tuning...")
        df = load_meddialog_data(TRAIN_DATA_PATH)
        df = preprocess_dataframe(df)
        
        # Combine patient query and doctor response for MLM
        texts = (df['patient_query_clean'] + " " + df['doctor_response_clean']).tolist()
        
        dataset = self.prepare_dataset(texts)
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=500,
            save_total_limit=2,
            logging_steps=50,
            learning_rate=2e-5,
            weight_decay=0.01,
            push_to_hub=False,
            report_to="none" # Disable wandb/mlflow
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        
        print("Starting fine-tuning...")
        trainer.train()
        
        print(f"Saving fine-tuned model to {self.output_dir}")
        trainer.save_model(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))

if __name__ == "__main__":
    finetuner = ClinicalBertFinetuner()
    # Using small epochs for demonstration/speed in this environment
    finetuner.train(epochs=1, batch_size=4)
