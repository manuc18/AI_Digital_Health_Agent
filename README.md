# AI Digital Health Agent 🏥

AI-powered digital health assistant capable of symptom analysis, severity classification, medication guidance, and consultation summarization.

## 🚀 Features

*   **Symptom Extraction**: Utilizes **Fine-tuned ClinicalBERT** and **Biomedical NER** (`d4data/biomedical-ner-all`) to understand patient queries and extract specific symptoms.
*   **Severity Classification**: Classifies health issues as **Low**, **Medium**, or **High** severity using a Logistic Regression model trained on heuristically labeled data.
*   **Medication Guidance**: Retrieves relevant doctor responses from a knowledge base of 480+ medical dialogues using semantic search.
*   **Consultation Summarization**: Generates concise summaries of the patient-doctor interaction using **T5-small**.
*   **Interactive UI**: A user-friendly web interface built with **Streamlit**.

## 🛠️ Tech Stack

*   **Language**: Python 3.9+
*   **ML/NLP**: PyTorch, Hugging Face Transformers, Scikit-learn
*   **Models**: 
    *   `emilyalsentzer/Bio_ClinicalBERT` (Fine-tuned)
    *   `d4data/biomedical-ner-all` (NER)
    *   `t5-small` (Summarization)
*   **App Framework**: Streamlit
*   **Visualization**: Matplotlib, Seaborn, WordCloud

## 📂 Project Structure

```
AI_Digital_Health_Agent/
├── src/
│   ├── data/           # Data loading, cleaning, labeling, EDA
│   ├── models/         # BERT, Classifier, NER, Summarizer
│   └── evaluate.py     # Model evaluation script
├── data/               # Raw and processed data
├── main.py             # End-to-end backend pipeline
├── app.py              # Streamlit frontend application
└── requirements.txt    # Project dependencies
```

## ⚡ Quick Start

1.  **Data Setup**:
    *   Download the **MedDialog** dataset from Kaggle: [MedDialogue Dataset](https://www.kaggle.com/datasets/hrezaeiyork/meddialogue)
    *   Create a folder `data/MedDialog/` and extract the files there.
    *   Ensure `english-train.json` is present in `data/MedDialog/`.

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install accelerate
    ```

3.  **Generate Models (Optional but Recommended)**:
    The fine-tuned models are excluded from the repo due to size. To generate them locally:
    ```bash
    python src/models/finetune.py
    ```
    *If skipped, the system will use the base ClinicalBERT model.*

4.  **Run the Web App**:
    ```bash
    streamlit run app.py
    ```

5.  **Run the Backend Pipeline** (Optional):
    To verify data processing, fine-tuning, and training:
    ```bash
    python main.py
    ```

6.  **Evaluate Models** (Optional):
    To see detailed accuracy metrics:
    ```bash
    python src/evaluate.py
    ```

## 📊 Performance

The severity classifier achieves **~74% accuracy** on the test set, with a high F1-score (**0.85**) for detecting **High Severity** (emergency) cases.

## 📝 License

This project is for educational purposes. The medical advice provided by the AI should not replace professional medical consultation.