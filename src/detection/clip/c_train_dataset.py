import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

FEATURE_DIR = Path("data/features")
LABEL_DIR = Path("data/results/clip_label")

def build_training_dataset(test_size=0.2, random_state=42):
    X, Y = [], []
    missed = 0
    skipped_due_to_length = 0
    skipped_due_to_label_shape = 0

    standard_labels = None  # 기준 라벨 열 이름

    for label_csv in tqdm(LABEL_DIR.glob("*.csv"), desc="Building training data"):
        video_name = label_csv.stem
        feature_video_dir = FEATURE_DIR / video_name
        if not feature_video_dir.exists():
            continue

        label_df = pd.read_csv(label_csv)

        # 기준 라벨 열 설정
        current_labels = label_df.columns[1:]
        if standard_labels is None:
            standard_labels = current_labels
        elif not all(col in current_labels for col in standard_labels):
            print(f"[경고] {label_csv.name} 라벨 열 불일치 → 건너뜀")
            skipped_due_to_label_shape += len(label_df)
            continue

        label_dict = {
            row['clip_id']: row[standard_labels].values.astype(int)
            for _, row in label_df.iterrows()
        }

        for clip_id, label in label_dict.items():
            feature_path = feature_video_dir / f"{clip_id}.npy"
            if not feature_path.exists():
                missed += 1
                continue

            feature = np.load(feature_path)
            if feature.ndim == 2:
                T = feature.shape[0]
                if T < 16:
                    skipped_due_to_length += 1
                    continue
                pooled = feature.mean(axis=0)
            elif feature.ndim == 1 and feature.shape[0] == 768:
                pooled = feature
            else:
                continue

            X.append(pooled)
            Y.append(label)

    print(f"[✔] 총 {len(X)}개 샘플 완료")
    print(f"[!] 특징 파일 누락: {missed}개")
    print(f"[!] 프레임 부족(T<16): {skipped_due_to_length}개")
    print(f"[!] 라벨 열 불일치 건너뜀: {skipped_due_to_label_shape}개")

    X = np.stack(X)
    Y = np.stack(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state, shuffle=True
    )
    return X_train, X_test, Y_train, Y_test


if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = build_training_dataset()
    print("Train:", X_train.shape, Y_train.shape)
    print("Test :", X_test.shape, Y_test.shape)
