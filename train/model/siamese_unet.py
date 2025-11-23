"""
Siamese U-Net for change detection (fixed decoder depth).
Input: before/after chips (B, C, H, W)
This variant uses 3 downsampling (enc2/enc3/enc4) and 3 upsampling steps,
so the final output spatial size matches the input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        """in_ch: channels of input to be upsampled (from previous layer)
        skip_ch: channels of the skip connection tensor (already concatenated b+a)
        out_ch: desired output channels after conv
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if skip.shape[2:] != x.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class SiameseUNet(nn.Module):
    def __init__(self, in_ch=6, base=32):
        super().__init__()
        # encoder (apply to before and after separately)
        self.enc1 = ConvBlock(in_ch, base)         # spatial: H
        self.enc2 = Down(base, base*2)             # spatial: H/2
        self.enc3 = Down(base*2, base*4)           # spatial: H/4
        self.enc4 = Down(base*4, base*8)           # spatial: H/8

        # bottleneck operates on concatenated deepest features (b4 + a4) -> channels = base*8*2
        self.bottleneck = ConvBlock(base*8*2, base*16)

        # DECODER (3 up steps to return to input resolution)
        # skip channel counts (concatenated b + a):
        #  c3 = base*4*2, c2 = base*2*2, c1 = base*1*2
        # up: (in_ch_from_prev, skip_ch, out_ch)
        self.up3 = Up(base*16, base*4*2, base*8)  # 32 -> 64 spatial
        self.up2 = Up(base*8,  base*2*2, base*4)  # 64 -> 128 spatial
        self.up1 = Up(base*4,  base*1*2, base*2)  # 128 -> 256 spatial

        # final head: accepts base*2 channels and outputs 1 channel mask
        self.final = nn.Conv2d(base*2, 1, kernel_size=1)

    def encode_single(self, x):
        e1 = self.enc1(x)   # base channels, H
        e2 = self.enc2(e1)  # base*2 channels, H/2
        e3 = self.enc3(e2)  # base*4 channels, H/4
        e4 = self.enc4(e3)  # base*8 channels, H/8
        return e1, e2, e3, e4

    def forward(self, before, after):
        # before/after: (B, C, H, W)
        b1,b2,b3,b4 = self.encode_single(before)
        a1,a2,a3,a4 = self.encode_single(after)

        # concatenated skips
        c4 = torch.cat([b4, a4], dim=1)  # base*8*2
        c3 = torch.cat([b3, a3], dim=1)  # base*4*2
        c2 = torch.cat([b2, a2], dim=1)  # base*2*2
        c1 = torch.cat([b1, a1], dim=1)  # base*1*2

        bt = self.bottleneck(c4)         # spatial H/8

        x = self.up3(bt, c3)             # H/4
        x = self.up2(x, c2)              # H/2
        x = self.up1(x, c1)              # H

        out = self.final(x)              # (B,1,H,W)
        return out
