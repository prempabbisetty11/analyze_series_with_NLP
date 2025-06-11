from glob import glob
import pandas as pd


def load_subtitles_dataset(dataset_path):
    subtitles_path = glob(dataset_path)

    scripts = []
    episode_num = []

    for path in subtitles_path:
        try:
            with open(path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            with open(path, 'r', encoding='utf-8-sig') as file:
                lines = file.readlines()

        # Skip header
        lines = lines[27:]
        # Extract and clean lines
        lines = [",".join(line.split(',')[9:]) for line in lines]
        lines = [line.replace('\\N', ' ') for line in lines]
        script = " ".join(lines)

        # Extract episode number from file name
        episode = int(path.split('-')[-1].split('.')[0].strip())

        scripts.append(script)
        episode_num.append(episode)

    # Create DataFrame
    df = pd.DataFrame.from_dict({"episode": episode_num, "script": scripts})
    return df
