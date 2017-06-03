from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.serialization import load_lua

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
    return transforms.ToTensor()(img)

def cv_loader(path):
    import cv2
    import numpy as np
    img = cv2.imread(path)
    img = Image.fromarray(img).convert('RGB')
    return transforms.ToTensor()(img)

# path = '/local/common-data/imagenet_2012/images/val/n04447861/ILSVRC2012_val_00028306.JPEG'
path = 'lena.png'
img = pil_loader(path)
img_cv = cv_loader(path)

img_t7 = load_lua('img.t7')

print('PIL vs t7', torch.dist(img, img_t7))
print('cv2 vs t7', torch.dist(img_cv, img_t7))
