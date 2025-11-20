# src/inference.py
import torch
from PIL import Image, ImageDraw
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import os

def get_model(checkpoint_path, device):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    return model

def predict_and_save(img_path, model, device, out_path, score_thr=0.5):
    img = Image.open(img_path).convert("RGB")
    tensor = F.to_tensor(img).to(device)
    with torch.no_grad():
        outputs = model([tensor])[0]
    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    keep = scores >= score_thr
    boxes = boxes[keep]; scores = scores[keep]

    draw = ImageDraw.Draw(img)
    for (x1,y1,x2,y2), s in zip(boxes, scores):
        draw.rectangle([x1,y1,x2,y2], outline="red", width=2)
        draw.text((x1, max(0,y1-12)), f"{s:.2f}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path)
    print("Saved:", out_path)

if __name__ == "__main__":
    import sys
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = "outputs/checkpoints/facercnn.pth"
    model = get_model(ckpt, device)
    # replace the image path below with a real image from data/fddb/originalPics
    sample = "data/fddb/originalPics/2002/07/19/big/img_158.jpg"
    predict_and_save(sample, model, device, "outputs/sample_pred.jpg", score_thr=0.4)
