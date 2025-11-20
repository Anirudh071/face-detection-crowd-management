import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from src.datasets import CocoFaceDataset
from src.transforms import get_transform
from src.utils import collate_fn

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = CocoFaceDataset("data/fddb", "data/annotations/fddb_coco.json", get_transform(train=True))
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    for epoch in range(2):
        print(f"\nEpoch {epoch+1}")
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print("Loss:", float(losses))

    torch.save(model.state_dict(), "outputs/checkpoints/facercnn.pth")
    print("Training complete → Model saved.")

if __name__ == "__main__":
    main()
