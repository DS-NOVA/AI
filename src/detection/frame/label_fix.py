import pandas as pd
import numpy as np
from pathlib import Path

FEATURE_DIR = Path("data/features")
LABEL_DIR = Path("data/results/personal_label")

for csv_path in LABEL_DIR.glob("*.csv"):
    video_name = csv_path.stem
    feature_video_dir = FEATURE_DIR / video_name
    if not feature_video_dir.exists():
        continue

    # features/영상폴더/*.npy의 프레임 총합 구하기
    total_frames = 0
    for f in feature_video_dir.glob("*.npy"):
        arr = np.load(f)
        total_frames += arr.shape[0]

    # 라벨 csv의 행 수
    df = pd.read_csv(csv_path, dtype=str)
    label_rows = len(df)

    if total_frames > label_rows:
        print(f"[수정] {video_name}: features 프레임 {total_frames} > 라벨 {label_rows} → frame_id=0 행 추가")
        # 첫 행 생성
        first_row = {col: 'FALSE' for col in df.columns}
        first_row[df.columns[0]] = '0'
        df = pd.concat([pd.DataFrame([first_row]), df], ignore_index=True)
        df.to_csv(csv_path, index=False)
    else:
        print(f"[패스] {video_name}: features 프레임 {total_frames} == 라벨 {label_rows} (수정 불필요)")
