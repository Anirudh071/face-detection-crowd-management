# scripts/make_sample_ellipselist.py
import os
from PIL import Image

IMG_ROOT = "data/fddb/originalPics"
OUT_FILE = "data/fddb/ellipseList.txt"
max_images = 40  # sample size

paths = []
for root, _, files in os.walk(IMG_ROOT):
    for f in files:
        if f.lower().endswith(".jpg"):
            rel = os.path.relpath(os.path.join(root, f), IMG_ROOT).replace("\\", "/")
            paths.append(rel[:-4])  # remove extension

paths = sorted(paths)[:max_images]

lines = []
for p in paths:
    img_path = os.path.join(IMG_ROOT, p + ".jpg")
    try:
        w, h = Image.open(img_path).size
    except Exception as e:
        continue
    cx = w / 2.0
    cy = h / 2.0
    a = w / 5.0
    b = h / 5.0
    angle = 0.0

    lines.append(p)
    lines.append("1")
    lines.append(f"{a:.6f} {b:.6f} {angle:.6f} {cy:.6f} {cx:.6f} 1")

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w", encoding="utf8") as f:
    f.write("\n".join(lines))

print(f"Sample ellipseList.txt generated with {len(paths)} entries -> {OUT_FILE}")
