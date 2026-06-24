#!/usr/bin/env python3
"""Predict one cropped symbol image with a checkpoint from train_math_symbol_classifier.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import transforms


class SmallSymbolCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def normalize_symbol_image(image: Image.Image, img_size: int = 64, pad: int = 8) -> Image.Image:
    gray = image.convert("L")
    arr = np.array(gray)
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if float(np.median(border)) < 127.0:
        gray = ImageOps.invert(gray)
        arr = np.array(gray)
    ink = arr < 245
    if int(ink.sum()) > 4:
        ys, xs = np.where(ink)
        gray = gray.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    w, h = gray.size
    side = max(w, h) + 2 * pad
    canvas = Image.new("L", (side, side), 255)
    canvas.paste(gray, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((img_size, img_size), Image.Resampling.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--checkpoint", default="weights/math_symbol_cnn.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    class_names = ckpt["class_names"]
    class_to_symbol = ckpt["class_to_symbol"]
    img_size = int(ckpt.get("img_size", 64))

    model = SmallSymbolCNN(num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device)
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    image = normalize_symbol_image(Image.open(args.image), img_size=img_size)
    tensor = transform(image).unsqueeze(0).to(args.device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    idx = int(torch.argmax(probs).item())
    class_name = class_names[idx]
    symbol = class_to_symbol[class_name]
    print({
        "class": class_name,
        "symbol": symbol,
        "confidence": float(probs[idx].item()),
    })

    topk = min(5, len(class_names))
    values, indices = torch.topk(probs, k=topk)
    for value, index in zip(values.tolist(), indices.tolist()):
        name = class_names[int(index)]
        print(f"{class_to_symbol[name]}\t{name}\t{value:.4f}")


if __name__ == "__main__":
    main()
