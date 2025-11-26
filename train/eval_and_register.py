
import argparse, glob, os
import torch
import rasterio
import numpy as np
import mlflow
from train.model.siamese_unet import SiameseUNet

import sys
def iou_score(pred, target, thr=0.5):
    p = (pred>thr).astype(np.uint8)
    t = (target>0.5).astype(np.uint8)
    inter = (p & t).sum()
    union = (p | t).sum()
    return float(inter)/float(union) if union>0 else 1.0


def evaluate(model_path, data_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseUNet(in_ch=6).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    files = sorted(glob.glob(os.path.join(data_dir, '*_before.tif')))
    scores = []
    for bf in files[:50]:
        af = bf.replace('_before.tif','_after.tif')
        mf = bf.replace('_before.tif','_mask.tif')
        with rasterio.open(bf) as ds:
            b = ds.read().astype('float32')/10000.0
        with rasterio.open(af) as ds:
            a = ds.read().astype('float32')/10000.0
        with rasterio.open(mf) as ds:
            m = ds.read(1).astype('float32')
        bi = torch.from_numpy(b).unsqueeze(0).to(device)
        ai = torch.from_numpy(a).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(bi, ai).squeeze(0).squeeze(0).cpu().numpy()
        scores.append(iou_score(out, m))
    return float(np.mean(scores)), len(scores)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', required=True)
    p.add_argument('--data-dir', default='data/chips')
    args = p.parse_args()
    mlflow.set_experiment('eval')
    with mlflow.start_run():
        mean_iou, n = evaluate(args.model_path, args.data_dir)
        mlflow.log_metric('mean_iou', mean_iou)
        print(f'[OK] evaluated {n} samples; mean_iou={mean_iou:.4f}')


if __name__ == "__main__" and __package__ is None:
    # ensure repo root is on sys.path so `import train` works
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)