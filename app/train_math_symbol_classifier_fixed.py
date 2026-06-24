#!/usr/bin/env python3
"""
Train a small classifier for handwritten digits and math symbols:
0 1 2 3 4 5 6 7 8 9 + - x : = ,

It automatically builds a dataset from:
1) MNIST digits via torchvision download=True.
2) Basic Handwritten Math Symbols Dataset from GitHub.
3) Synthetic font-generated samples for missing/weak classes.
4) Optional real samples exported from your canvas.

Recommended first run:
python train_math_symbol_classifier.py --rebuild --epochs 15 --device auto

After dataset has already been built:
python train_math_symbol_classifier.py --epochs 15 --device auto

Optional real data folder:
real_symbols/
  0/
  1/
  ...
  plus/
  minus/
  mul/
  colon/
  equal/
  comma/

Then:
python train_math_symbol_classifier.py --rebuild --real-data-dir real_symbols --epochs 20
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


CLASS_NAMES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "plus", "minus", "mul", "colon", "equal", "comma",
]

CLASS_TO_SYMBOL = {
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "plus": "+",
    "minus": "-",
    "mul": "x",
    "colon": ":",
    "equal": "=",
    "comma": ",",
}

BHMSDS_URL = "https://github.com/wblachowski/bhmsds/archive/refs/heads/master.zip"

# Names used in wblachowski/bhmsds symbols folder.
BHMSDS_NAME_TO_CLASS = {
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "plus": "plus",
    "minus": "minus",
    "x": "mul",
    "asterisk": "mul",
    "times": "mul",
    "star": "mul",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class BuildStats:
    mnist: int = 0
    bhmsds: int = 0
    synthetic: int = 0
    real: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def class_dir(root: Path, class_name: str) -> Path:
    path = root / class_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_symbol_image(image: Image.Image, img_size: int = 64, pad: int = 8) -> Image.Image:
    """Convert to white background, dark ink, square padded grayscale image."""
    gray = image.convert("L")
    arr = np.array(gray)
    if arr.size == 0:
        return Image.new("L", (img_size, img_size), 255)

    # If border is dark, invert. This fixes MNIST black background.
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if float(np.median(border)) < 127.0:
        gray = ImageOps.invert(gray)
        arr = np.array(gray)

    ink = arr < 245
    if int(ink.sum()) > 4:
        ys, xs = np.where(ink)
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        gray = gray.crop((x1, y1, x2, y2))

    w, h = gray.size
    side = max(w, h) + 2 * pad
    canvas = Image.new("L", (side, side), 255)
    canvas.paste(gray, ((side - w) // 2, (side - h) // 2))
    canvas = canvas.resize((img_size, img_size), Image.Resampling.LANCZOS)
    return canvas


def save_normalized(image: Image.Image, output_path: Path, img_size: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = normalize_symbol_image(image, img_size=img_size)
    image.save(output_path)


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, output_path)


def build_from_mnist(output_root: Path, raw_root: Path, limit_per_digit: int, img_size: int) -> int:
    if limit_per_digit <= 0:
        return 0

    print("Loading/downloading MNIST digits with torchvision...")
    mnist = datasets.MNIST(root=str(raw_root), train=True, download=True)
    per_digit_count = {str(i): 0 for i in range(10)}
    total = 0

    for image, label in mnist:
        class_name = str(int(label))
        if per_digit_count[class_name] >= limit_per_digit:
            continue
        out = class_dir(output_root, class_name) / f"mnist_{per_digit_count[class_name]:05d}.png"
        save_normalized(image, out, img_size=img_size)
        per_digit_count[class_name] += 1
        total += 1
        if all(count >= limit_per_digit for count in per_digit_count.values()):
            break

    print("MNIST samples:", per_digit_count)
    return total


def bhmsds_symbol_name(path: Path) -> str:
    # Files are usually like plus-0123.png, slash-1484.png, etc.
    stem = path.stem.lower()
    return stem.split("-", 1)[0]


def build_from_bhmsds(output_root: Path, raw_root: Path, limit_per_class: int, img_size: int) -> int:
    if limit_per_class <= 0:
        return 0

    downloads = raw_root / "downloads"
    zip_path = downloads / "bhmsds-master.zip"
    extract_dir = raw_root / "bhmsds"
    download_file(BHMSDS_URL, zip_path)

    if not extract_dir.exists():
        print("Extracting BHMSDS...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    symbol_dirs = list(extract_dir.glob("**/symbols"))
    if not symbol_dirs:
        print("WARNING: BHMSDS symbols folder was not found.")
        return 0

    symbols_dir = symbol_dirs[0]
    counts = {name: 0 for name in CLASS_NAMES}
    total = 0

    for path in sorted(symbols_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        symbol_name = bhmsds_symbol_name(path)
        class_name = BHMSDS_NAME_TO_CLASS.get(symbol_name)
        if class_name is None:
            continue
        if counts[class_name] >= limit_per_class:
            continue
        try:
            image = Image.open(path)
        except Exception:
            continue
        out = class_dir(output_root, class_name) / f"bhmsds_{counts[class_name]:05d}.png"
        save_normalized(image, out, img_size=img_size)
        counts[class_name] += 1
        total += 1

    print("BHMSDS samples:", {k: v for k, v in counts.items() if v})
    return total


def find_font_paths() -> list[Path]:
    candidates: list[Path] = []
    roots = [
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
        Path("C:/Windows/Fonts"),
        Path("/System/Library/Fonts"),
        Path("/Library/Fonts"),
    ]
    for root in roots:
        if not root.exists():
            continue
        for ext in ("*.ttf", "*.otf", "*.ttc"):
            candidates.extend(root.rglob(ext))
    return candidates


def load_random_font(font_paths: list[Path], size: int) -> ImageFont.ImageFont:
    if font_paths:
        path = random.choice(font_paths)
        try:
            return ImageFont.truetype(str(path), size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_rough_symbol(symbol: str, img_size: int, font_paths: list[Path]) -> Image.Image:
    # Font-based sample with position, rotation, blur, thickness, and noise changes.
    canvas_size = img_size + 24
    image = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(image)

    font_size = random.randint(int(img_size * 0.50), int(img_size * 0.86))
    font = load_random_font(font_paths, font_size)

    bbox = draw.textbbox((0, 0), symbol, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (canvas_size - tw) // 2 + random.randint(-8, 8)
    y = (canvas_size - th) // 2 + random.randint(-8, 8)

    ink = random.randint(0, 55)
    # Draw several nearby copies to simulate stroke thickness.
    thickness = random.choice([0, 0, 1, 1, 2])
    for dx in range(-thickness, thickness + 1):
        for dy in range(-thickness, thickness + 1):
            if dx * dx + dy * dy <= thickness * thickness + 1:
                draw.text((x + dx, y + dy), symbol, fill=ink, font=font)

    angle = random.uniform(-18, 18)
    image = image.rotate(angle, fillcolor=255, resample=Image.Resampling.BICUBIC)

    if random.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.15, 0.75)))

    arr = np.array(image).astype(np.int16)
    noise = np.random.normal(0, random.uniform(2, 10), size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr, mode="L")

    return normalize_symbol_image(image, img_size=img_size)


def build_synthetic(output_root: Path, samples_per_class: int, img_size: int) -> int:
    if samples_per_class <= 0:
        return 0

    print("Generating synthetic samples...")
    font_paths = find_font_paths()
    print(f"Found {len(font_paths)} font files.")
    total = 0

    for class_name in CLASS_NAMES:
        symbol = CLASS_TO_SYMBOL[class_name]
        out_dir = class_dir(output_root, class_name)
        for i in range(samples_per_class):
            image = draw_rough_symbol(symbol, img_size=img_size, font_paths=font_paths)
            image.save(out_dir / f"synthetic_{i:05d}.png")
            total += 1

    return total


def normalize_real_folder_name(name: str) -> str | None:
    value = name.strip().lower()
    aliases = {
        "+": "plus",
        "plus": "plus",
        "cong": "plus",
        "-": "minus",
        "minus": "minus",
        "tru": "minus",
        "x": "mul",
        "*": "mul",
        "mul": "mul",
        "times": "mul",
        "multiply": "mul",
        ":": "colon",
        "colon": "colon",
        "=": "equal",
        "equal": "equal",
        ",": "comma",
        "comma": "comma",
    }
    if value in CLASS_NAMES:
        return value
    if value in aliases:
        return aliases[value]
    if value.isdigit() and value in CLASS_NAMES:
        return value
    return None


def iter_images(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def build_from_real_data(output_root: Path, real_data_dir: Path | None, img_size: int) -> int:
    if real_data_dir is None:
        return 0
    if not real_data_dir.exists():
        print(f"WARNING: real data folder not found: {real_data_dir}")
        return 0

    print(f"Importing real samples from {real_data_dir}...")
    total = 0
    per_class = {name: 0 for name in CLASS_NAMES}

    for child in sorted(real_data_dir.iterdir()):
        if not child.is_dir():
            continue
        class_name = normalize_real_folder_name(child.name)
        if class_name is None:
            print(f"Skipping unknown real-data class folder: {child.name}")
            continue
        for path in iter_images(child):
            try:
                image = Image.open(path)
            except Exception:
                continue
            idx = per_class[class_name]
            out = class_dir(output_root, class_name) / f"real_{idx:05d}.png"
            save_normalized(image, out, img_size=img_size)
            per_class[class_name] += 1
            total += 1

    print("Real samples:", {k: v for k, v in per_class.items() if v})
    return total


def build_dataset(args: argparse.Namespace) -> Path:
    data_dir = Path(args.data_dir)
    raw_root = data_dir / "raw"
    combined_root = data_dir / "combined"

    if args.rebuild:
        ensure_clean_dir(combined_root)
    else:
        combined_root.mkdir(parents=True, exist_ok=True)
        for name in CLASS_NAMES:
            class_dir(combined_root, name)

    stats = BuildStats()
    if args.rebuild or not any(combined_root.glob("*/*.png")):
        stats.mnist = build_from_mnist(
            output_root=combined_root,
            raw_root=raw_root,
            limit_per_digit=args.mnist_per_digit,
            img_size=args.img_size,
        )
        stats.bhmsds = build_from_bhmsds(
            output_root=combined_root,
            raw_root=raw_root,
            limit_per_class=args.bhmsds_per_class,
            img_size=args.img_size,
        )
        stats.synthetic = build_synthetic(
            output_root=combined_root,
            samples_per_class=args.synthetic_per_class,
            img_size=args.img_size,
        )
        stats.real = build_from_real_data(
            output_root=combined_root,
            real_data_dir=Path(args.real_data_dir) if args.real_data_dir else None,
            img_size=args.img_size,
        )
        print("Dataset build stats:", stats)
    else:
        print(f"Using existing dataset: {combined_root}")

    counts = {}
    for name in CLASS_NAMES:
        counts[name] = len(list((combined_root / name).glob("*.png")))
    print("Class counts:")
    for name in CLASS_NAMES:
        print(f"  {name:>5s}: {counts[name]}")

    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        raise RuntimeError(f"Empty classes: {empty}. Add data or increase synthetic_per_class.")

    return combined_root


class SmallSymbolCNN(nn.Module):
    def __init__(self, num_classes: int = len(CLASS_NAMES)) -> None:
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


class RemappedImageFolder(datasets.ImageFolder):
    """
    torchvision.datasets.ImageFolder always sorts folders alphabetically.

    Folder order becomes:
    0 1 2 3 4 5 6 7 8 9 colon comma equal minus mul plus

    But the model/checkpoint must use CLASS_NAMES order:
    0 1 2 3 4 5 6 7 8 9 plus minus mul colon equal comma

    This class remaps ImageFolder's folder index to the target index in CLASS_NAMES.
    """

    def __init__(self, root: str, transform=None):
        super().__init__(root, transform=transform)
        missing = [name for name in CLASS_NAMES if name not in self.class_to_idx]
        if missing:
            raise RuntimeError(f"Missing class folders: {missing}")

        self.folder_idx_to_target_idx = {
            folder_idx: CLASS_NAMES.index(class_name)
            for class_name, folder_idx in self.class_to_idx.items()
            if class_name in CLASS_NAMES
        }

        # Print once per dataset creation so the mapping is visible.
        print("ImageFolder classes:", self.classes)
        print("Model classes:", CLASS_NAMES)

    def __getitem__(self, index: int):
        image, folder_target = super().__getitem__(index)
        target = self.folder_idx_to_target_idx[folder_target]
        return image, target


def make_loaders(dataset_root: Path, img_size: int, batch_size: int, val_ratio: float, seed: int):
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(
            degrees=14,
            translate=(0.10, 0.10),
            scale=(0.82, 1.12),
            shear=8,
            fill=255,
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    full_for_split = RemappedImageFolder(str(dataset_root), transform=train_transform)

    val_size = max(1, int(len(full_for_split) * val_ratio))
    train_size = len(full_for_split) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_for_split, [train_size, val_size], generator=generator)

    # random_split shares the same dataset object. Use a second ImageFolder for validation.
    full_val = RemappedImageFolder(str(dataset_root), transform=val_transform)
    val_ds.dataset = full_val

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == labels).float().mean().item())


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    per_class_correct = torch.zeros(len(CLASS_NAMES), dtype=torch.long)
    per_class_count = torch.zeros(len(CLASS_NAMES), dtype=torch.long)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            total_loss += float(loss.item()) * images.size(0)
            total_correct += int((preds == labels).sum().item())
            total_count += int(images.size(0))
            for cls in range(len(CLASS_NAMES)):
                mask = labels == cls
                per_class_count[cls] += int(mask.sum().item())
                per_class_correct[cls] += int((preds[mask] == cls).sum().item())

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    class_acc = {}
    for idx, name in enumerate(CLASS_NAMES):
        count = int(per_class_count[idx].item())
        corr = int(per_class_correct[idx].item())
        class_acc[name] = None if count == 0 else corr / count
    return avg_loss, avg_acc, class_acc


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    dataset_root = build_dataset(args)
    device = choose_device(args.device)
    print("Device:", device)

    train_loader, val_loader = make_loaders(
        dataset_root=dataset_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    model = SmallSymbolCNN(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_acc = -1.0
    export_path = Path(args.export)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        running_loss = 0.0
        running_acc = 0.0
        steps = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            running_acc += accuracy_from_logits(logits.detach(), labels)
            steps += 1

        scheduler.step()
        train_loss = running_loss / max(steps, 1)
        train_acc = running_acc / max(steps, 1)
        val_loss, val_acc, class_acc = evaluate(model, val_loader, device, criterion)
        elapsed = time.time() - start

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"time={elapsed:.1f}s"
        )

        # Print weakest classes sometimes to catch symbol issues early.
        if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
            known_acc = {k: v for k, v in class_acc.items() if v is not None}
            weakest = sorted(known_acc.items(), key=lambda kv: kv[1])[:5]
            print("  Weakest classes:", {k: round(v, 4) for k, v in weakest})

        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                "model_state": model.state_dict(),
                "class_names": CLASS_NAMES,
                "class_to_symbol": CLASS_TO_SYMBOL,
                "img_size": args.img_size,
                "val_acc": val_acc,
                "epoch": epoch,
            }
            torch.save(checkpoint, export_path)
            with open(export_path.with_suffix(".labels.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "class_names": CLASS_NAMES,
                        "class_to_symbol": CLASS_TO_SYMBOL,
                        "img_size": args.img_size,
                        "best_val_acc": val_acc,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"  Saved best checkpoint to {export_path}")

    print("Best val acc:", best_acc)
    print("Exported:", export_path)
    print("Labels:", export_path.with_suffix(".labels.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="math_symbol_data")
    parser.add_argument("--real-data-dir", default="")
    parser.add_argument("--export", default="weights/math_symbol_cnn.pt")
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--mnist-per-digit", type=int, default=3000)
    parser.add_argument("--bhmsds-per-class", type=int, default=1500)
    parser.add_argument("--synthetic-per-class", type=int, default=1200)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        train(parse_args())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise
