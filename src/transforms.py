import torchvision.transforms as T

def get_transform(train=True):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)
