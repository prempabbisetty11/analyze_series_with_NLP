from transformers import pipeline
import torch
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
import os
import sys
import pathlib

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader import load_subtitles_dataset
import nltk

# Download punkt for sentence tokenization
nltk.download('punkt', download_dir='c:/code_live/nltk_data')

class ThemeClassifier():
    def __init__(self, theme_list):
        self.model_name = "facebook/bart-large-mnli"
        self.device = 0 if torch.cuda.is_available() else -1
        self.theme_list = theme_list
        self.theme_classifier = self.load_model(self.device)

    def load_model(self, device):
        theme_classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=device,
            framework="pt"
        )
        return theme_classifier

    def get_themes_inference(self, script):
        script_sentences = sent_tokenize(script)
        sentence_batch_size = 20
        script_batches = []
        for index in range(0, len(script_sentences), sentence_batch_size):
            sent = " ".join(script_sentences[index:index + sentence_batch_size])
            script_batches.append(sent)
        theme_output = self.theme_classifier(
            script_batches,
            self.theme_list,
            multi_label=True
        )
        themes = {}
        for output in theme_output:
            for label, score in zip(output['labels'], output['scores']):
                if label not in themes:
                    themes[label] = []
                themes[label].append(score)
        themes = {key: np.mean(np.array(value)) for key, value in themes.items()}
        return themes

    def get_themes(self, dataset_path, save_path=None):
        # Read Save Output if Exists
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            missing_themes = [theme for theme in self.theme_list if theme not in df.columns]
            if not missing_themes:
                return df
            else:
                print(f"Missing themes in saved file: {missing_themes}. Recomputing themes.")

        # load model
        df = load_subtitles_dataset(dataset_path)

        # Run Inference
        output_themes = df['script'].apply(self.get_themes_inference)
        themes_df = pd.DataFrame(output_themes.tolist())
        df[themes_df.columns] = themes_df

        # Save the output
        if save_path is not None:
            df.to_csv(save_path, index=False)
        return df