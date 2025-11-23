import glob, os, torch
from train.model.siamese_unet import SiameseUNet

# find a chip triplet
before = sorted(glob.glob('data/chips/*_before.tif'))[:1]
if not before:
    print("NO CHIPS FOUND in data/chips")
    raise SystemExit(1)
bfile = before[0]
afile = bfile.replace('_before.tif','_after.tif')
mfile = bfile.replace('_before.tif','_mask.tif')
print("Using files:", bfile, afile, mfile)

# load with rasterio (lazy import inside)
import rasterio, numpy as np
def read_tif(p):
    with rasterio.open(p) as ds:
        arr = ds.read().astype('float32')/10000.0
    return arr

b_arr = read_tif(bfile)
a_arr = read_tif(afile)
# add batch dim and convert to torch
b = torch.from_numpy(b_arr).unsqueeze(0)    # (1,C,H,W)
a = torch.from_numpy(a_arr).unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = SiameseUNet(in_ch=b.shape[1])
model = model.to(device)
b = b.to(device)
a = a.to(device)

# run encode_single for both branches and print shapes
print("=== Encoding BEFORE ===")
b1,b2,b3,b4 = model.encode_single(b)
print("b1", tuple(b1.shape))
print("b2", tuple(b2.shape))
print("b3", tuple(b3.shape))
print("b4", tuple(b4.shape))

print("=== Encoding AFTER ===")
a1,a2,a3,a4 = model.encode_single(a)
print("a1", tuple(a1.shape))
print("a2", tuple(a2.shape))
print("a3", tuple(a3.shape))
print("a4", tuple(a4.shape))

# concatenated skips:
c1 = torch.cat([b1,a1], dim=1)
c2 = torch.cat([b2,a2], dim=1)
c3 = torch.cat([b3,a3], dim=1)
c4 = torch.cat([b4,a4], dim=1)
print("c1", tuple(c1.shape))
print("c2", tuple(c2.shape))
print("c3", tuple(c3.shape))
print("c4", tuple(c4.shape))

# bottleneck
bt = model.bottleneck(c4)
print("bottleneck output", tuple(bt.shape))

# step through decoder, printing shapes before each concat
x = bt
print("\\n=== Decoder steps ===")
for i, (up, skip) in enumerate([(model.up4, c3), (model.up3, c2), (model.up2, c1), (model.up1, c1)]):
    print(f\"Before up{i+1}: x={tuple(x.shape)} skip={tuple(skip.shape)}\")
    # up module will attempt to upsample and concat, so call its up only to inspect sizes
    x_up = up.up(x)
    print(f\"After convtranspose: x_up={tuple(x_up.shape)} (will concat with skip)\")
    # if spatial mismatch, report it
    if x_up.shape[2:] != skip.shape[2:]:
        print(f\"SPATIAL MISMATCH at up{i+1}: x_up.shape={x_up.shape}, skip.shape={skip.shape}\")
    # perform actual forward for progress
    try:
        x = up(x, skip)
    except Exception as e:
        print(f\"ERROR during up{i+1} forward: {e}\") 
        raise
    print(f\"After up{i+1} conv output: x={tuple(x.shape)}\\n\")

print("PASS: decoder completed, final shape:", tuple(x.shape))
