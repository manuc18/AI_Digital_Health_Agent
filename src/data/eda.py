import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from pathlib import Path
from src.config import PROCESSED_DATA_DIR

class EDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.output_dir = PROCESSED_DATA_DIR / "eda"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_length_distribution(self):
        """Plot the distribution of patient query lengths."""
        plt.figure(figsize=(10, 6))
        lengths = self.df['patient_query_clean'].str.split().apply(len)
        sns.histplot(lengths, bins=50, kde=True)
        plt.title('Distribution of Patient Query Lengths (Words)')
        plt.xlabel('Length')
        plt.ylabel('Count')
        plt.savefig(self.output_dir / "query_length_dist.png")
        print(f"Saved length distribution plot to {self.output_dir / 'query_length_dist.png'}")
        plt.close()

    def generate_wordcloud(self):
        """Generate a word cloud for patient queries."""
        text = " ".join(self.df['patient_query_clean'].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Patient Queries')
        plt.savefig(self.output_dir / "wordcloud.png")
        print(f"Saved word cloud to {self.output_dir / 'wordcloud.png'}")
        plt.close()

    def run(self):
        print("Running EDA...")
        self.plot_length_distribution()
        self.generate_wordcloud()
        print("EDA completed.")
