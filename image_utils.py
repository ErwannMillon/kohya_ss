import numpy as np
import torch
from PIL import Image, ImageOps


def load_image(image_path):
    i = Image.open(image_path)
    i = ImageOps.exif_transpose(i)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    if 'A' in i.getbands():
        mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
        mask = 1. - torch.from_numpy(mask)
    else:
        mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
    return (image, mask)