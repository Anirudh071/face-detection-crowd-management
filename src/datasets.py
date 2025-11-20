import os
import torch
import torch.utils.data
import torchvision.transforms as T
from PIL import Image
import json

class CocoFaceDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.annotation_file = annotation_file
        self.transforms = transforms

        with open(annotation_file, 'r') as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]

        # Build index: image_id -> list of annotations
        self.ann_map = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.ann_map:
                self.ann_map[img_id] = []
            self.ann_map[img_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.root, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        # Load annotations
        anns = self.ann_map.get(img_info["id"], [])
        boxes = []
        labels = []
        for ann in anns:
            boxes.append(ann["bbox"])
            labels.append(1)   # Only one class: face

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_info["id"]])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target
