#!/usr/bin/env python3
"""
Simple FastAPI server that accepts two uploaded GeoTIFFs (before/after) and returns a prediction mask (.npy)
This is a minimal working server to verify end-to-end functionality.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn, tempfile, os, numpy as np, rasterio, torch
from train.model.siamese_unet import SiameseUNet

app = FastAPI(title="Geospatial Change Detection API")

MODEL_PATH = os.environ.get("MODEL_PATH", "runs/model_inference.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseUNet(in_ch=6).to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

@app.get("/health")
async def health():
    return {"status":"ok", "model_loaded": os.path.exists(MODEL_PATH)}

def read_tiff_bytes(data: bytes):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    tmp.write(data)
    tmp.close()
    return tmp.name

@app.post("/predict")
async def predict(before: UploadFile = File(...), after: UploadFile = File(...)):
    before_bytes = await before.read()
    after_bytes = await after.read()
    bpath = read_tiff_bytes(before_bytes)
    apath = read_tiff_bytes(after_bytes)
    try:
        with rasterio.open(bpath) as ds:
            b = ds.read().astype('float32')/10000.0
        with rasterio.open(apath) as ds:
            a = ds.read().astype('float32')/10000.0
        tb = torch.from_numpy(b).unsqueeze(0).to(DEVICE)
        ta = torch.from_numpy(a).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(tb, ta)
            prob = out.sigmoid().squeeze(0).cpu().numpy()
        # return a small npy as proof-of-concept (client can save & view)
        outfile = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
        np.save(outfile, prob)
        return {"result_npy": outfile.name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(bpath); os.unlink(apath)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
