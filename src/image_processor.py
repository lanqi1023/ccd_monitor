import cv2
import numpy as np
from numpy.typing import NDArray

class Image_Processor:
    def __init__(
        self,
        begin: tuple[int, int] = (0, 0),
        image_size: int = 1600,
        array_size: int = 32,
    ):
        self.begin = begin
        self.image_size = image_size
        self.array_size = array_size

    def __call__(self, bayer: NDArray) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        if bayer.dtype == np.uint8:
            image = cv2.cvtColor(bayer, cv2.COLOR_BAYER_RGGB2GRAY)
        else:
            image = cv2.cvtColor((bayer >> 4).astype(np.uint8), cv2.COLOR_BAYER_RGGB2GRAY)
        y, x = self.begin
        roi = image[
            y : y + self.image_size,
            x : x + self.image_size
        ]
        block_size = self.image_size // self.array_size
        array = roi.reshape(
            self.array_size, block_size,
            self.array_size, block_size,
        ).mean(axis=(1, 3), dtype=np.float32)
        return image, array
