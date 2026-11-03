import numpy as np
from PIL import Image
from torch import Tensor


def images_to_pillow(images: Tensor | list[Tensor]) -> list[Image.Image]:
    pillow_images = []
    for _bn, image in enumerate(images):
        i = 255.0 * image.cpu().numpy()
        pillow_images.append(Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)))
    return pillow_images
