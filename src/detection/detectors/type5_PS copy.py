import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from type1_PHF_ex_copy import analyze_video_for_phf_v3  # PHF 검출 함수 (사용자 제공 함수)
from type2_PHR_ex import analyze_red_black_transitions  # PHR 검출 함수 (사용자 제공 함수)

def detect_ps_sequences(video_path, output_csv_path, fps=30, min_duration_sec=5, max_gap_frames=5):
    print("[1/3] PHF 분석 시작")
    phf_results = analyze_video_for_phf_v3(video_path, save_csv=False)
    print("[2/3] PHR 분석 시작")
    phr_results = analyze_red_black_transitions(video_path, save_csv=False)

    min_len = min(len(phf_results), len(phr_results))
    is_flash = np.array([
        phf_results[i]["PHF"] or phr_results[i]["is_dangerous"]
        for i in range(min_len)
    ])

    print(f"Total frames: {len(is_flash)}")
    print(f"Total flashes (True values): {np.sum(is_flash)}")

    ps_sequences = []
    start = None
    end = None
    gap_count = 0
    min_length = min_duration_sec * fps

    for i, val in enumerate(is_flash):
        print(f"Frame {i}: {val}")
        if val:
            if start is None:
                start = i
                print(f"Start of new flash sequence at frame {i}")
            end = i  # 마지막 True의 위치
            gap_count = 0
        else:
            if start is not None:
                gap_count += 1
                if gap_count > max_gap_frames:
                    if end - start + 1 >= min_length:
                        ps_sequences.append((start, end))
                        print(f"Adding sequence from frame {start} to frame {end}")
                    else:
                        print(f"Discarding sequence from frame {start} to frame {end} (too short)")
                    start = None
                    end = None
                    gap_count = 0

    # 마지막 시퀀스 처리
    if start is not None and end is not None:
        if end - start + 1 >= min_length:
            ps_sequences.append((start, end))
            print(f"Adding final sequence from frame {start} to frame {end}")
        else:
            print(f"Discarding final sequence from frame {start} to frame {end} (too short)")

    print(f"PS 시퀀스 개수: {len(ps_sequences)}")
    ps_df = pd.DataFrame(ps_sequences, columns=["start_frame", "end_frame"])
    ps_df.to_csv(output_csv_path, index=False)
    print(f"[3/3] PS 시퀀스 {len(ps_df)}개가 '{output_csv_path}'에 저장되었습니다.")
    return ps_df


# 예시 실행
if __name__ == "__main__":
    detect_ps_sequences(
        video_path="data/raw/net1_sample.mp4",
        output_csv_path="data/results/type5_PS_copy_net1sam.csv",
        fps=30,
        min_duration_sec=5
    )
