from glob import glob
import os
import pandas as pd

def load_subtitles_dataset(dataset_path):
    from glob import glob
    import os
    import pandas as pd

    subtitles_path = glob(os.path.join(dataset_path, "*.ass"))

    scripts = []
    episode_num = []

    for file_path in subtitles_path:
        try:
            with open(file_path, encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f" Skipping file due to read error: {file_path}, Reason: {e}")
            continue

        # Skip header
        lines = lines[27:]
        # Extract and clean lines
        lines = [",".join(line.split(',')[9:]) for line in lines if line.startswith("Dialogue:")]
        lines = [line.replace('\\N', ' ') for line in lines]
        script = " ".join(lines)

        # Extract episode number from file name
        try:
            episode = int(os.path.splitext(os.path.basename(file_path))[0].split('-')[-1])
        except Exception:
            episode = 0

        scripts.append(script)
        episode_num.append(episode)

    df = pd.DataFrame.from_dict({"episode": episode_num, "script": scripts})
    return df