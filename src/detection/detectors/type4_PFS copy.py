import csv
import os
from type1_PHF_ex_copy import analyze_video_for_phf_v3

def detect_pfs_from_results(results: list, fps: float = 30.0) -> list:
    phf_mask = [r['PHF'] for r in results]
    pfs_sequences = []

    # Step 1: PHF 그룹 (True 연속 구간) 추출
    groups = []
    start = None
    for i, val in enumerate(phf_mask):
        if val:
            if start is None:
                start = i
        else:
            if start is not None:
                groups.append((start, i - 1))
                start = None
    if start is not None:
        groups.append((start, len(phf_mask) - 1))

    # Step 2: 연속 그룹 시퀀스 추출 (3프레임 이내 간격)
    seq = []
    for i in range(len(groups)):
        if not seq:
            seq.append(groups[i])
        else:
            prev_end = seq[-1][1]
            curr_start = groups[i][0]
            if curr_start - prev_end <= 3:
                seq.append(groups[i])
            else:
                if len(seq) >= 3:
                    # Step 3: 주파수 계산
                    seq_start = seq[0][0]
                    seq_end = seq[-1][1]
                    duration_sec = (seq_end - seq_start + 1) / fps
                    frequency = len(seq) / duration_sec if duration_sec > 0 else 0
                    if frequency > 3.0:
                        pfs_sequences.append((seq_start + 1, seq_end + 1))  # 1-based frame index
                seq = [groups[i]]

    # 마지막 시퀀스 확인
    if len(seq) >= 3:
        seq_start = seq[0][0]
        seq_end = seq[-1][1]
        duration_sec = (seq_end - seq_start + 1) / fps
        frequency = len(seq) / duration_sec if duration_sec > 0 else 0
        if frequency > 3.0:
            pfs_sequences.append((seq_start + 1, seq_end + 1))  # 1-based index

    return pfs_sequences

def save_pfs_sequences_to_csv(pfs_sequences: list, filename='pfs_sequences.csv'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence_idx', 'start_frame', 'end_frame'])
        for idx, (start, end) in enumerate(pfs_sequences):
            writer.writerow([idx + 1, start, end])
    print(f"PFS 시퀀스 저장 완료: {filename}")

results = analyze_video_for_phf_v3('data/raw/net_1.mp4', save_csv=False)
pfs_sequences = detect_pfs_from_results(results, fps=30.0)
save_pfs_sequences_to_csv(pfs_sequences, 'data/results/type4_PFS_copy_net1.csv')