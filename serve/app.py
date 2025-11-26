import os
import time
import tempfile
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional

from serve.utils import (
    sliding_windows,
    stitch_tiles,
    read_geotiff,
    write_geotiff,
)

app = FastAPI(title="Deforestation Change Detection API")

# defaults
MODEL_PATH = os.getenv("MODEL_PATH", "runs/model_fused.pth")

# select device
DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps") if getattr(torch, "has_mps", False) and torch.has_mps else torch.device("cpu")
)

_model = None

def load_model():
    """Lazy-load model only once."""
    global _model
    if _model is None:
        from train.model.siamese_unet import SiameseUNet
        model = SiameseUNet(in_ch=6)

        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)

        model.to(DEVICE)
        model.eval()
        _model = model
    return _model


class PredictResult(BaseModel):
    out_path: str
    runtime_s: float


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_exists": os.path.exists(MODEL_PATH)
    }


@app.post("/predict", response_model=PredictResult)
async def predict(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    tile_size: int = 256,
    overlap: int = 32,
    threshold: float = 0.5,
):
    start = time.time()

    # Save temporary uploaded files
    try:
        tfb = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        tfb.write(await before.read())
        tfb.flush()

        tfa = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        tfa.write(await after.read())
        tfa.flush()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Load geotiffs
    b_arr, b_meta = read_geotiff(tfb.name)
    a_arr, a_meta = read_geotiff(tfa.name)

    if b_arr.shape[1:] != a_arr.shape[1:]:
        raise HTTPException(status_code=400, detail="Before/after spatial dimensions differ")

    H, W = b_arr.shape[1:]

    model = load_model()

    windows = []
    outputs = []

    # sliding window
    for window, tile_chw in sliding_windows(b_arr, tile_size, overlap):
        r, c, h, w = window

        # prepare matching after-tile
        tile_b = b_arr[:, r:r+h, c:c+w]
        tile_a = a_arr[:, r:r+h, c:c+w]

        # pad if needed
        if tile_b.shape[1] < tile_size or tile_b.shape[2] < tile_size:
            pad_b = np.zeros((b_arr.shape[0], tile_size, tile_size), dtype=tile_b.dtype)
            pad_b[:, :tile_b.shape[1], :tile_b.shape[2]] = tile_b
            tile_b = pad_b

            pad_a = np.zeros((a_arr.shape[0], tile_size, tile_size), dtype=tile_a.dtype)
            pad_a[:, :tile_a.shape[1], :tile_a.shape[2]] = tile_a
            tile_a = pad_a

        tb = torch.from_numpy(tile_b).unsqueeze(0).to(DEVICE)
        ta = torch.from_numpy(tile_a).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(tb, ta)  # (1,1,tile,tile)
            out = out.squeeze(0).squeeze(0).cpu().numpy()
            out_cropped = out[:h, :w]

        windows.append(window)
        outputs.append(out_cropped)

    # Stitch tiles
    stitched = stitch_tiles(windows, outputs, (H, W))

    # Apply threshold (0/255 mask)
    mask = (stitched > threshold).astype("uint8") * 255

    # save output
    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/pred_{int(time.time())}.tif"

    write_geotiff(out_path, stitched, b_meta)

    runtime = time.time() - start

    # cleanup
    try:
        os.unlink(tfb.name)
        os.unlink(tfa.name)
    except:
        pass

    return {
        "out_path": out_path,
        "runtime_s": runtime
    }


if __name__ == "__main__":
    uvicorn.run("serve.app:app", host="0.0.0.0", port=8000, reload=False)
