import os, json, math
from PIL import Image

FDDB_BASE = "data/fddb"
IMAGES_DIR = os.path.join(FDDB_BASE, "originalPics")
ELLIPSE_FILE = os.path.join(FDDB_BASE, "ellipseList.txt")
OUT_ANN = os.path.join("data", "annotations", "fddb_coco.json")

def ellipse_to_bbox(cx, cy, a, b, theta):
    pts = []
    for i in range(36):
        t = 2 * math.pi * i / 36.0
        x = cx + a*math.cos(t)*math.cos(theta) - b*math.sin(t)*math.sin(theta)
        y = cy + a*math.cos(t)*math.sin(theta) + b*math.sin(t)*math.cos(theta)
        pts.append((x, y))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def parse_ellipse_list(ellipse_file):
    with open(ellipse_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    imgs = []
    while i < len(lines):
        img_rel = lines[i]; i += 1
        n = int(lines[i]); i += 1
        faces = []
        for _ in range(n):
            a, b, angle, cy, cx, _ = map(float, lines[i].split()[:6])
            i += 1
            bbox = ellipse_to_bbox(cx, cy, a, b, angle)
            faces.append(bbox)
        imgs.append((img_rel, faces))
    return imgs

def build_coco(imgs_list):
    coco = {"images": [], "annotations": [], "categories": [{"id":1, "name":"face"}]}
    ann_id = 1
    img_id = 1
    for img_rel, faces in imgs_list:
        img_path = os.path.join(IMAGES_DIR, img_rel + ".jpg")
        if not os.path.exists(img_path):
            continue
        w, h = Image.open(img_path).size
        coco["images"].append({
            "id": img_id,
            "file_name": os.path.join("originalPics", img_rel + ".jpg"),
            "width": w,
            "height": h
        })
        for bbox in faces:
            x, y, wbox, hbox = bbox
            x, y = max(0, x), max(0, y)
            wbox, hbox = max(1, wbox), max(1, hbox)
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x, y, wbox, hbox],
                "area": wbox*hbox,
                "iscrowd": 0
            })
            ann_id += 1
        img_id += 1
    return coco

if __name__ == "__main__":
    assert os.path.exists(ELLIPSE_FILE), "ellipseList.txt not found!"
    imgs = parse_ellipse_list(ELLIPSE_FILE)
    coco = build_coco(imgs)
    os.makedirs("data/annotations", exist_ok=True)
    with open(OUT_ANN, "w") as f:
        json.dump(coco, f)
    print("Wrote COCO JSON to", OUT_ANN)
