import spacy
from nltk.tokenize import sent_tokenize
import os
import sys
import pathlib
from ast import literal_eval
import re
folder_path = pathlib.Path().parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))
import pandas as pd
from glob import glob

class NameEntityRecognizer:
    def __init__(self):
        self.nlp_model = self.load_model()

    def load_model(self):
        nlp = spacy.load("en_core_web_trf")
        return nlp

    def get_ners_inference(self, script):
        script_sentences = sent_tokenize(script)
        ner_output = []

        for sentence in script_sentences:
            doc = self.nlp_model(sentence)
            ners = set()
            for entity in doc.ents:
                if entity.label_ == "PERSON":
                    full_name = entity.text
                    first_name = full_name.split(" ")[0]
                    first_name = first_name.strip()
                    ners.add(first_name)
            ner_output.append(ners)

        return ner_output

    @staticmethod
    def load_subtitles_dataset(dataset_path):
        subtitles_path = glob(os.path.join(dataset_path, "*.ass"))

        scripts = []
        episode_num = []

        for file_path in subtitles_path:
            try:
                file_content = pd.read_csv(file_path, sep='\n', header=None, encoding='utf-8', engine='python', quoting=3)
                lines = file_content[0].tolist()
            except Exception as e:
                print(f"⚠️ Skipping file due to read error: {file_path}, Reason: {e}")
                continue

            # Filter only Dialogue lines
            dialogue_lines = [line for line in lines if line.startswith("Dialogue:")]

            # Extract the 10th field and clean
            script_lines = [",".join(line.split(',')[9:]).replace('\\N', ' ') for line in dialogue_lines]
            script = " ".join(script_lines)

            # Extract episode number
            match = re.search(r'(\d+)\.ass$', file_path)
            episode = int(match.group(1)) if match else 0

            scripts.append(script)
            episode_num.append(episode)

        df = pd.DataFrame({"episode": episode_num, "script": scripts})
        return df

    def get_ners(self, dataset_path, save_path=None):
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df['ners'] = df['ners'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
            return df

        # load dataset
        df = self.load_subtitles_dataset(dataset_path)
        df = df.head(10)

        # Run Inferences
        df['ners'] = df['script'].apply(self.get_ners_inference)

        if save_path is not None:
            df.to_csv(save_path, index=False)

        return df