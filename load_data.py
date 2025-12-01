import os
import numpy as np
from PIL import Image

def load_dataset(root="data/tgs/train", size=(128, 128), limit=None):
    images_dir = os.path.join(root, "images")
    masks_dir = os.path.join(root, "masks")

    image_files = sorted(os.listdir(images_dir))

    if limit:
        image_files = image_files[:limit]

    images = []
    masks = []

    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)
        mask_path = os.path.join(masks_dir, img_name.replace(".png", "_mask.png"))

        image = Image.open(img_path).convert("L").resize(size)
        mask = Image.open(mask_path).convert("L").resize(size)

        images.append(np.array(image) / 255.0)
        masks.append((np.array(mask) > 127).astype(np.float32))

    images = np.expand_dims(np.array(images), axis=1)
    masks = np.expand_dims(np.array(masks), axis=1)

    return images, masks
