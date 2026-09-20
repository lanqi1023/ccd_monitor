from image_processor import Image_Processor

import time
from ccd import CCD
from numpy.typing import NDArray
from typing import Callable, Optional

class CCD_Manager:
    def __init__(self, process: Optional[Callable[[NDArray], tuple[NDArray, NDArray, tuple[int, int, int, int]]]] = None, fps: float = 10):
        self.__camera = CCD()
        self.__process = process or Image_Processor()
        self.__interval  = 1.0 / fps
        self.__last_time = 0.0

    def __publish(self, bayer: NDArray, on_publish: Callable[[NDArray, NDArray, tuple[int, int, int, int]], None]) -> None:
        now = time.monotonic()
        if now - self.__last_time >= self.__interval:
            self.__last_time = now
            image, array, roi_info = self.__process(bayer)
            on_publish(image, array, roi_info)

    def start(self, on_publish: Callable[[NDArray, NDArray, tuple[int, int, int, int]], None]) -> None:
        if self.__camera.open():
            self.__camera.frame_rate = 10
            self.__camera.process    = CCD.PROCESS_NONE
            self.__last_time         = 0.0
            self.__camera.begin_thread(on_capture = self.__publish, on_publish = on_publish)

    def stop(self) -> None:
        self.__camera.end_thread()
        self.__camera.close()
