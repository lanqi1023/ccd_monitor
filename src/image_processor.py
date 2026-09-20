import cv2
import numpy as np
from numpy.typing import NDArray

class Image_Processor:
    def __init__(
        self,
        roi_origin:  tuple[int, int] = (768, 1024),
        roi_shape:   tuple[int, int] = (1536, 2048),
        array_shape: tuple[int, int] = (48, 64)
    ):
        self.roi_origin = roi_origin
        self.roi_shape  = roi_shape
        self.array_shape = array_shape

    def __call__(self, bayer: NDArray) -> tuple[NDArray[np.uint8], NDArray[np.float32], tuple[int, int, int, int]]:
        if bayer.dtype == np.uint8:
            image = cv2.cvtColor(bayer, cv2.COLOR_BAYER_RGGB2GRAY)
        else:
            image = cv2.cvtColor((bayer >> 4).astype(np.uint8), cv2.COLOR_BAYER_RGGB2GRAY)
        row, col = self.roi_origin
        image_height, image_width = self.roi_shape
        array_height, array_width = self.array_shape
        roi = image[
            row : row + image_height,
            col : col + image_width
        ]
        array = roi.reshape(
            array_height, image_height // array_height,
            array_width,  image_width  // array_width
        ).mean(axis=(1, 3), dtype=np.float32)
        return image, array, (*self.roi_origin, *self.roi_shape)
