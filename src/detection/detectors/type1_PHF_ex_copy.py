import cv2
import numpy as np
import csv
import os
from tqdm import tqdm

def rgb_to_luminance(frame_rgb: np.ndarray) -> np.ndarray:
    r, g, b = frame_rgb[:, :, 0], frame_rgb[:, :, 1], frame_rgb[:, :, 2]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance.astype(np.float32)

def detect_phf_conditions_v3(prev_frame: np.ndarray, curr_frame: np.ndarray,
                              delta_l_threshold=20, min_luminance_threshold=20, area_threshold=0.4) -> dict:
    result = {}
    delta = curr_frame - prev_frame  # 방향성을 유지하기 위해 abs 제거

    result['프레임간_휘도차_20이상_존재'] = np.any(np.abs(delta) >= delta_l_threshold)
    result['더_어두운프레임_최소휘도_20이하'] = min(np.min(prev_frame), np.min(curr_frame)) <= min_luminance_threshold
    result['휘도차20이상_영역_비율_40이상'] = np.sum(np.abs(delta) >= delta_l_threshold) / delta.size >= area_threshold
    result['밝아진_프레임'] = np.mean(curr_frame) - np.mean(prev_frame) >= delta_l_threshold
    result['PHF'] = all([result['프레임간_휘도차_20이상_존재'],
                         result['더_어두운프레임_최소휘도_20이하'],
                         result['휘도차20이상_영역_비율_40이상']])
    return result

def is_similar_luminance(frame1: np.ndarray, frame2: np.ndarray, threshold=4.0) -> bool:
    diff = np.abs(frame1 - frame2)
    return np.mean(diff) < threshold

def expand_phf_labels_with_luminance_increase_only_for_brightening(
    luminance_frames: list, results: list, delta_threshold: float = 0.0
) -> list:
    expanded = [res.copy() for res in results]
    for idx, res in enumerate(results):
        if not res.get("PHF"):
            continue
        if not res.get("밝아진_프레임"):  # '밝아진 프레임'이 아니면 확장하지 않음
            continue
        if (idx + 1 >= len(luminance_frames)) or (idx + 1 >= len(expanded)):
            continue

        curr_frame_lum = luminance_frames[idx]
        next_frame_lum = luminance_frames[idx + 1]
        avg_curr = np.mean(curr_frame_lum)
        avg_next = np.mean(next_frame_lum)

        if (avg_next - avg_curr) >= delta_threshold:
            expanded[idx + 1]["PHF"] = True

    return expanded

def analyze_video_for_phf_v3(video_path: str, output_csv: str = None, save_csv: bool = True):
    cap = cv2.VideoCapture(video_path)
    results = []
    luminance_frames = []
    frame_idx = 0

    if not cap.isOpened():
        print("비디오를 열 수 없습니다.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_frame_lum = None

    with tqdm(total=total_frames, desc="PHF 분석 중") as pbar:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_lum = rgb_to_luminance(frame_rgb)
            luminance_frames.append(frame_lum)

            if prev_frame_lum is not None:
                result = detect_phf_conditions_v3(prev_frame_lum, frame_lum)
                results.append(result)

            prev_frame_lum = frame_lum
            frame_idx += 1
            pbar.update(1)

    cap.release()

    results = expand_phf_labels_with_luminance_increase_only_for_brightening(luminance_frames, results, delta_threshold=0.0)

    if save_csv and output_csv:
        save_phf_results_to_csv(results, output_csv)
        print(f"PHF 결과 저장 완료: {output_csv}")

    return results

def save_phf_results_to_csv(results: list, filename='phf_detection_results.csv'):
    if not results:
        print("결과가 없습니다.")
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    keys = list(results[0].keys())
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_idx'] + keys)
        for idx, res in enumerate(results):
            row = [idx + 1] + [res[k] for k in keys]
            writer.writerow(row)

if __name__ == '__main__':
    analyze_video_for_phf_v3('data/raw/9_fireworks_01.mp4', 'data/results/type1_PHF_9_fireworks_01.csv')
"""

import cv2
import numpy as np
import csv
import os
from tqdm import tqdm
from glob import glob

# 기존 함수들 생략 (수정하지 않음)
# rgb_to_luminance, detect_phf_conditions_v3,
# is_similar_luminance, expand_phf_labels_with_luminance_increase_only_for_brightening

def analyze_video_for_phf_v3(video_path: str, output_csv: str = None, save_csv: bool = True):
    cap = cv2.VideoCapture(video_path)
    results = []
    luminance_frames = []
    frame_idx = 0

    if not cap.isOpened():
        print(f"비디오를 열 수 없습니다: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_frame_lum = None

    with tqdm(total=total_frames, desc=f"{os.path.basename(video_path)} 분석 중") as pbar:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_lum = rgb_to_luminance(frame_rgb)
            luminance_frames.append(frame_lum)

            if prev_frame_lum is not None:
                result = detect_phf_conditions_v3(prev_frame_lum, frame_lum)
                results.append(result)

            prev_frame_lum = frame_lum
            frame_idx += 1
            pbar.update(1)

    cap.release()

    results = expand_phf_labels_with_luminance_increase_only_for_brightening(luminance_frames, results, delta_threshold=0.0)

    if save_csv and output_csv:
        save_phf_frame_phfonly_to_csv(results, output_csv)
        print(f"PHF 결과 저장 완료: {output_csv}")

    return results

def save_phf_frame_phfonly_to_csv(results: list, filename='phf_minimal_results.csv'):
    if not results:
        print("결과가 없습니다.")
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_idx', 'PHF'])
        for idx, res in enumerate(results):
            writer.writerow([idx + 2, res['PHF']])  # idx+2인 이유: 첫 번째 비교는 프레임 1과 2이기 때문

if __name__ == '__main__':
    input_dir = 'data/raw'
    output_dir = 'data/results'
    os.makedirs(output_dir, exist_ok=True)

    mp4_files = glob(os.path.join(input_dir, '*.mp4'))

    for video_path in mp4_files:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_csv = os.path.join(output_dir, f"type1_PHF_{base_name}.csv")
        analyze_video_for_phf_v3(video_path, output_csv)
"""