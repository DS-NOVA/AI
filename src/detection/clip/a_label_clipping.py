import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)

LABEL_DIR = Path("data/results/personal_label")
OUTPUT_DIR = Path("data/results/clip_label")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLIP_LENGTH = 16

def convert_frame_to_clip_labels():
    csv_paths = list(LABEL_DIR.glob("*.csv"))

    for csv_path in tqdm(csv_paths, desc="Converting frame labels to clip labels"):
        video_name = csv_path.stem
        df = pd.read_csv(csv_path)

        
        label_cols = df.columns[1:]
        df[label_cols] = df[label_cols].applymap(
            lambda x: str(x).strip().upper() == 'TRUE' if pd.notna(x) else False
        ).astype(int)

        clip_rows = []

        num_clips = len(df) // CLIP_LENGTH
        for i in range(num_clips):
            start = i * CLIP_LENGTH
            end = start + CLIP_LENGTH
            clip_df = df.iloc[start:end]
            label = (clip_df[label_cols].sum(axis=0) > 0).astype(int)
            clip_id = f"{video_name}_clip_{i:04d}"
            row = [clip_id] + label.tolist()
            clip_rows.append(row)

        if len(df) % CLIP_LENGTH > 0:
            clip_df = df.iloc[num_clips * CLIP_LENGTH:]
            label = (clip_df[label_cols].sum(axis=0) > 0).astype(int)
            clip_id = f"{video_name}_clip_{num_clips:04d}"
            row = [clip_id] + label.tolist()
            clip_rows.append(row)

        out_df = pd.DataFrame(clip_rows, columns=["clip_id"] + label_cols.tolist())
        out_path = OUTPUT_DIR / f"{video_name}.csv"
        out_df.to_csv(out_path, index=False)

if __name__ == "__main__":
    convert_frame_to_clip_labels()
