from packet import input, weight

import cv2
import numpy as np
from numpy.typing import NDArray
from time import sleep

ROI_ORIGIN:  tuple[int, int] = (768, 1024)
ROI_SHAPE:   tuple[int, int] = (1536, 2048)
ARRAY_SHAPE: tuple[int, int] = (48, 64)

def process(image: NDArray) -> tuple[NDArray[np.float32], tuple[int, int, int, int]]:
    row, col = ROI_ORIGIN
    image_height, image_width = ROI_SHAPE
    array_height, array_width = ARRAY_SHAPE
    roi = image[
        row : row + image_height,
        col : col + image_width
    ]
    array = roi.reshape(
        array_height, image_height // array_height,
        array_width,  image_width  // array_width
    ).mean(axis=(1, 3), dtype=np.float32)
    return array, (*ROI_ORIGIN, *ROI_SHAPE)

def main_target():
    sleep(10)
    input.set((np.random.rand(32, 32) * 255).astype(np.uint8))
    weight.set((np.random.rand(32, 32) * 255).astype(np.uint8))
    sleep(10)
    input.set((np.random.rand(32, 32) * 255).astype(np.uint8))
    sleep(10)
    weight.set((np.random.rand(32, 32) * 255).astype(np.uint8))
    ...
