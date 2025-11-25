import argparse, glob, os
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
import mlflow
from model.siamese_unet import SiameseUNet

class ChipDataset(Dataset):
    def __init__(self, folder):
        self.before = sorted(glob.glob(os.path.join(folder, "*_before.tif")))
    def __len__(self): return len(self.before)
    def __getitem__(self, idx):
        bfile = self.before[idx]
        afile = bfile.replace("_before.tif", "_after.tif")
        mfile = bfile.replace("_before.tif", "_mask.tif")
        with rasterio.open(bfile) as ds:
            b = ds.read().astype('float32')/10000.0
        with rasterio.open(afile) as ds:
            a = ds.read().astype('float32')/10000.0
        with rasterio.open(mfile) as ds:
            m = ds.read(1).astype('float32')
        return torch.from_numpy(b), torch.from_numpy(a), torch.from_numpy(m).unsqueeze(0)

def train_loop(args):
    ds = ChipDataset(args.data_dir)
    print(f"DEBUG: dataset length = {len(ds)}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch, "has_mps", False) and torch.has_mps else "cpu"))
    print("Using device:", device)
    model = SiameseUNet(in_ch=6).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run():
        mlflow.log_params({"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr})
        for ep in range(args.epochs):
            model.train()
            epoch_loss = 0.0
            for i, (b,a,m) in enumerate(dl):
                b = b.to(device); a = a.to(device); m = m.to(device)
                out = model(b, a)
                loss = bce(out, m)
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += loss.item()
                if i % 10 == 0:
                    print(f"ep={ep} step={i} loss={loss.item():.4f}")
            avg = epoch_loss / max(1, len(dl))
            print(f"Epoch {ep} loss {avg:.4f}")
            mlflow.log_metric("train_loss", avg, step=ep)
        # save model
        os.makedirs("runs", exist_ok=True)
        model_path = "runs/model_inference.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        print("[OK] model saved to", model_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/chips")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--experiment", default="deforestation_demo")
    args = p.parse_args()
    train_loop(args)