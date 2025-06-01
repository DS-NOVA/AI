import numpy as np
import pandas as pd
from pathlib import Path

FEATURE_DIR = Path("data/features")
LABEL_DIR = Path("data/results/personal_label")

for label_csv in LABEL_DIR.glob("*.csv"):
    video_name = label_csv.stem
    feature_video_dir = FEATURE_DIR / video_name
    if not feature_video_dir.exists():
        print(f"[X] {video_name} : feature 폴더 없음")
        continue

    # 프레임 단위 라벨 불러오기
    label_df = pd.read_csv(label_csv)
    label_names = label_df.columns[1:]
    frame_label_count = len(label_df)

    print(f"\n=== {video_name} ===")
    print(f"프레임 라벨 총 개수: {frame_label_count}")

    total_frames_in_feature = 0
    for clip_file in sorted(feature_video_dir.glob("*.npy")):
        clip = np.load(clip_file)
        clip_idx = int(clip_file.stem.split('_')[-1])
        frame_start = clip_idx * 16
        frame_end = frame_start + clip.shape[0]
        total_frames_in_feature += clip.shape[0]

        # 라벨 범위 초과 확인
        if frame_end > frame_label_count:
            print(f"[경고] {clip_file.name}: feature 프레임({clip.shape[0]}) + 라벨 범위({frame_start}~{frame_end-1}) → 라벨 부족")
        elif clip.shape[0] != min(16, frame_label_count - frame_start):
            print(f"[경고] {clip_file.name}: feature 프레임({clip.shape[0]}) vs. 예상 라벨({min(16, frame_label_count - frame_start)}) 불일치")

    print(f"features 내 프레임 총합: {total_frames_in_feature}")

    if total_frames_in_feature != frame_label_count:
        print(f"[!] 전체 프레임 총합이 라벨 개수와 다름! ({total_frames_in_feature} vs {frame_label_count})")
