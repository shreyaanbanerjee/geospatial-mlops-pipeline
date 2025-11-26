import numpy as np, glob, rasterio, torch
from train.model.siamese_unet import SiameseUNet
files = sorted(glob.glob('../data/chips/*_before.tif'))[:50]
model = SiameseUNet(in_ch=6)
model.load_state_dict(torch.load('runs/model_fused.pth', map_location='cpu'))
model.eval()
rows=[]
for bf in files:
    af=bf.replace('_before.tif','_after.tif'); mf=bf.replace('_before.tif','_mask.tif')
    with rasterio.open(bf) as ds: b = ds.read().astype('float32')/10000.0
    with rasterio.open(af) as ds: a = ds.read().astype('float32')/10000.0
    with rasterio.open(mf) as ds: m = ds.read(1).astype('float32')
    bi=torch.from_numpy(b).unsqueeze(0); ai=torch.from_numpy(a).unsqueeze(0)
    with torch.no_grad():
        out = model(bi, ai).squeeze(0).squeeze(0).numpy()
    thr = 0.5
    p = (out>thr).astype(np.uint8); t = (m>0.5).astype(np.uint8)
    inter=(p & t).sum(); union=(p | t).sum()
    iou=float(inter)/float(union) if union>0 else 1.0
    rows.append((bf, iou))
import csv
with open('runs/eval_report.csv','w') as f:
    w=csv.writer(f); w.writerow(['before_file','iou']); w.writerows(rows)
print('Saved runs/eval_report.csv')