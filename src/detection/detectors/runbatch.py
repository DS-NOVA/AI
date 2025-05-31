import os
import csv
import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

from type1_PHF_ex_copy import analyze_video_for_phf_v3
from type2_PHR_ex import analyze_red_black_transitions


def save_frame_results_to_csv(frame_data, filename, fieldname):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_idx', fieldname])
        for idx, val in enumerate(frame_data):
            writer.writerow([idx + 1, val])


def run_type1_batch(input_dir='data/raw', output_dir='data/results/type1'):
    os.makedirs(output_dir, exist_ok=True)
    for video_path in glob(os.path.join(input_dir, '*.mp4')):
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"[TYPE1] 처리 중: {base_name}")
        results = analyze_video_for_phf_v3(video_path, save_csv=False)
        frame_data = [r['PHF'] for r in results]
        out_csv = os.path.join(output_dir, f"type1_PHF_{base_name}.csv")
        save_frame_results_to_csv(frame_data, out_csv, 'PHF')


def run_type2_batch(input_dir='data/raw', output_dir='data/results/type2'):
    os.makedirs(output_dir, exist_ok=True)
    for video_path in glob(os.path.join(input_dir, '*.mp4')):
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"[TYPE2] 처리 중: {base_name}")
        results = analyze_red_black_transitions(video_path, save_csv=False)
        frame_data = [r['is_dangerous'] for r in results]
        out_csv = os.path.join(output_dir, f"type2_PHR_{base_name}.csv")
        save_frame_results_to_csv(frame_data, out_csv, 'PHR')


def run_type4_batch(input_dir='data/raw', output_dir='data/results/type4'):
    os.makedirs(output_dir, exist_ok=True)
    for video_path in glob(os.path.join(input_dir, '*.mp4')):
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"[TYPE4] 처리 중: {base_name}")
        results = analyze_video_for_phf_v3(video_path, save_csv=False)
        phf_mask = [r['PHF'] for r in results]

        fps = 30.0
        pfs_sequences = detect_pfs_from_results(results, fps=fps)

        max_idx = len(results)
        frame_labels = [False] * max_idx
        for s, e in pfs_sequences:
            for i in range(s-1, e):
                if i < max_idx:
                    frame_labels[i] = True

        out_csv = os.path.join(output_dir, f"type4_PFS_{base_name}.csv")
        save_frame_results_to_csv(frame_labels, out_csv, 'PFS')


def detect_pfs_from_results(results: list, fps: float = 30.0) -> list:
    phf_mask = [r['PHF'] for r in results]
    pfs_sequences = []

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
                    seq_start = seq[0][0]
                    seq_end = seq[-1][1]
                    duration_sec = (seq_end - seq_start + 1) / fps
                    frequency = len(seq) / duration_sec if duration_sec > 0 else 0
                    if frequency > 3.0:
                        pfs_sequences.append((seq_start + 1, seq_end + 1))
                seq = [groups[i]]

    if len(seq) >= 3:
        seq_start = seq[0][0]
        seq_end = seq[-1][1]
        duration_sec = (seq_end - seq_start + 1) / fps
        frequency = len(seq) / duration_sec if duration_sec > 0 else 0
        if frequency > 3.0:
            pfs_sequences.append((seq_start + 1, seq_end + 1))

    return pfs_sequences


def run_type5_batch(input_dir='data/raw', output_dir='data/results/type5'):
    os.makedirs(output_dir, exist_ok=True)
    for video_path in glob(os.path.join(input_dir, '*.mp4')):
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"[TYPE5] 처리 중: {base_name}")
        phf_results = analyze_video_for_phf_v3(video_path, save_csv=False)
        phr_results = analyze_red_black_transitions(video_path, save_csv=False)

        min_len = min(len(phf_results), len(phr_results))
        is_flash = [phf_results[i]['PHF'] or phr_results[i]['is_dangerous'] for i in range(min_len)]

        out_csv = os.path.join(output_dir, f"type5_PS_{base_name}.csv")
        save_frame_results_to_csv(is_flash, out_csv, 'PS')


if __name__ == '__main__':
    run_type1_batch()
    run_type2_batch()
    run_type4_batch()
    run_type5_batch()