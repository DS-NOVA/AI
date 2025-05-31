import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

CLIP_LENGTH = 16
FRAME_SIZE = (224, 224)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_clips_from_video(video_path, output_subdir):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        return

    video_name = video_path.stem
    frames = []
    clip_index = 0

    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, FRAME_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        if len(frames) == CLIP_LENGTH:
            clip = np.stack(frames)
            save_path = output_subdir / f"{video_name}_clip_{clip_index:04d}.npy"
            np.save(str(save_path), clip)
            clip_index += 1
            frames = []

    
    if 0 < len(frames) < CLIP_LENGTH:
        pad_count = CLIP_LENGTH - len(frames)
        pad_frame = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), np.nan, dtype=np.float32)
        frames.extend([pad_frame] * pad_count)
        clip = np.stack(frames)
        save_path = output_subdir / f"{video_name}_clip_{clip_index:04d}.npy"
        np.save(str(save_path), clip)

    cap.release()


def process_all_videos():
    video_files = list(RAW_DIR.glob("*.mp4"))

    for video_file in tqdm(video_files, desc="Processing videos"):
        output_subdir = OUTPUT_DIR / video_file.stem
        output_subdir.mkdir(parents=True, exist_ok=True)
        extract_clips_from_video(video_file, output_subdir)

if __name__ == "__main__":
    process_all_videos()