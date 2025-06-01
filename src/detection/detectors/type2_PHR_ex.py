import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def is_saturated_red(yuv_frame):
    Y, Cr, Cb = cv2.split(yuv_frame)
    red_mask = (Cr > 150) & (Cb < 120)
    return red_mask.astype(np.uint8)

def is_black(yuv_frame):
    Y, _, _ = cv2.split(yuv_frame)
    black_mask = Y < 40
    return black_mask.astype(np.uint8)

def expand_dangerous_labels_recursively(results, base_indices, threshold=0.05):
    expanded = set(base_indices)
    queue = list(base_indices)

    while queue:
        current = queue.pop(0)

        for offset in [-1, 1]:  # 이전 및 다음 프레임 확인
            neighbor = current + offset
            if 0 <= neighbor < len(results) and neighbor not in expanded:
                r_diff = abs(results[current]['saturated_red_ratio'] - results[neighbor]['saturated_red_ratio'])
                b_diff = abs(results[current]['black_ratio'] - results[neighbor]['black_ratio'])

                if r_diff <= threshold and b_diff <= threshold:
                    expanded.add(neighbor)
                    queue.append(neighbor)

    return sorted(expanded)

def analyze_red_black_transitions(video_path, output_csv_path=None, threshold_ratio=0.25, similarity_threshold=0.30, save_csv=True):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[오류] 영상 파일을 열 수 없습니다: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = []

    prev_red_ratio = None
    prev_black_ratio = None

    for frame_idx in tqdm(range(total_frames), desc="영상 분석 중"):
        ret, frame = cap.read()
        if not ret:
            break

        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        red_mask = is_saturated_red(yuv_frame)
        black_mask = is_black(yuv_frame)

        red_ratio = np.sum(red_mask) / red_mask.size
        black_ratio = np.sum(black_mask) / black_mask.size

        is_dangerous = False
        if prev_red_ratio is not None and prev_black_ratio is not None:
            red_diff = red_ratio - prev_red_ratio
            black_diff = black_ratio - prev_black_ratio

            """
            # 조건 A: 적색 감소 + 흑색 증가
            if red_diff <= -threshold_ratio and black_diff >= threshold_ratio:
                is_dangerous = True
            """
            # 조건 B: 적색 증가 + 흑색 감소
            if red_diff >= threshold_ratio and black_diff <= -threshold_ratio:
                is_dangerous = True

        results.append({
            "frame_index": frame_idx,
            "saturated_red_ratio": round(float(red_ratio), 4),
            "black_ratio": round(float(black_ratio), 4),
            "is_dangerous": is_dangerous
        })

        prev_red_ratio = red_ratio
        prev_black_ratio = black_ratio

    cap.release()

    # 위험 프레임 인덱스 추출 및 유사 프레임 포함 확장
    dangerous_indices = [i for i, r in enumerate(results) if r["is_dangerous"]]
    expanded_indices = expand_dangerous_labels_recursively(results, dangerous_indices, threshold=similarity_threshold)

    # 결과 반영
    for i in range(len(results)):
        results[i]["is_dangerous"] = i in expanded_indices

    if save_csv and output_csv_path:
        df = pd.DataFrame(results)
        df.to_csv(output_csv_path, index=False)
        print(f"[완료] 분석 결과가 '{output_csv_path}'에 저장되었습니다.")
    
    return results

if __name__ == '__main__':
    analyze_red_black_transitions("data/raw/net_1.mp4", "data/results/type2_PHR_net1.csv")
