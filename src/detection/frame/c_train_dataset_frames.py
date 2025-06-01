import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

FEATURE_DIR = Path("data/features")
LABEL_DIR = Path("data/results/personal_label")

def build_frame_dataset(test_size=0.2, random_state=42, return_meta=False):
    X, Y = [], []
    meta = []  # (video, frame_id)
    label_names = None

    for label_csv in tqdm(LABEL_DIR.glob("*.csv"), desc="Building frame-level dataset"):
        video_name = label_csv.stem
        feature_video_dir = FEATURE_DIR / video_name
        if not feature_video_dir.exists():
            continue

        label_df = pd.read_csv(label_csv)
        if label_names is None:
            label_names = label_df.columns[1:]

        frame_labels = label_df[label_names].applymap(
            lambda x: str(x).strip().upper() == 'TRUE' if pd.notna(x) else False
        ).astype(int).values

        for clip_file in feature_video_dir.glob("*.npy"):
            clip = np.load(clip_file)  # (T, 768)
            clip_index = int(clip_file.stem.split('_')[-1])
            start_idx = clip_index * 16
            end_idx = start_idx + clip.shape[0]
            if end_idx > len(frame_labels):
                continue

            labels = frame_labels[start_idx:end_idx]
            if clip.shape[0] != labels.shape[0]:
                continue

            for i in range(clip.shape[0]):
                X.append(clip[i])
                Y.append(labels[i])
                meta.append((video_name, start_idx + i))

    X = np.stack(X)
    Y = np.stack(Y)
    meta = np.array(meta)

    # split용 인덱스 만들기
    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=random_state, shuffle=True)
    X_train, X_test = X[idx_train], X[idx_test]
    Y_train, Y_test = Y[idx_train], Y[idx_test]
    meta_train, meta_test = meta[idx_train], meta[idx_test]

    if return_meta:
        # label_names를 리스트로 반환
        return X_train, X_test, Y_train, Y_test, pd.DataFrame(meta_test, columns=["video", "frame_id"]), list(label_names)
    else:
        return X_train, X_test, Y_train, Y_test
