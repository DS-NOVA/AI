import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import timm
from torchvision import transforms

CLIP_DIR = Path(__file__).resolve().parents[2] / "data" / "generated"
FEATURE_DIR = Path(__file__).resolve().parents[2] / "data" / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('vit_base_patch16_224.mae', pretrained=True)
model.to(device)
model.eval()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def preprocess(clip):
    clip = clip.astype(np.float32) / 255.0
    clip = np.transpose(clip, (0, 3, 1, 2))
    return [(torch.tensor(frame).float() - torch.tensor(mean).view(3,1,1)) / torch.tensor(std).view(3,1,1)
            for frame in clip]

def extract_feature_from_clip(clip):
    processed_frames = preprocess(clip)
    features = []

    for frame in processed_frames:
        if torch.isnan(frame).any():
            continue  # NaN 프레임은 무시
        with torch.no_grad():
            input_tensor = frame.unsqueeze(0).to(device)
            output = model.forward_features(input_tensor)
            cls_token = output[:, 0, :]
            features.append(cls_token.squeeze(0).cpu().numpy())

    return np.stack(features) if features else np.empty((0, 768), dtype=np.float32)

def main():
    clip_files = list(CLIP_DIR.rglob("*.npy"))
    for clip_path in tqdm(clip_files, desc="Extracting features"):
        clip = np.load(clip_path)
        try:
            feature = extract_feature_from_clip(clip)
            out_path = FEATURE_DIR / clip_path.relative_to(CLIP_DIR)
            out_path = out_path.with_suffix(".npy")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), feature)
        except Exception as e:
            print(f"[ERROR] {clip_path.name}: {e}")

if __name__ == "__main__":
    main()
