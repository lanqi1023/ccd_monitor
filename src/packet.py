import cv2
import logging
from numpy.typing import NDArray
from struct import Struct
from threading import Lock
from time import time
from typing import Optional

log = logging.getLogger(__name__)

class __Packet:
    def __init__(self):
        self.__lock     = Lock()
        self.__frame_id = -1
        self.__data: Optional[bytes] = None

    def get(self) -> tuple[int, Optional[bytes]]:
        with self.__lock:
            return self.__frame_id, self.__data

class __Input_Packet(__Packet):
    def __init__(self, JPEG_QUALITY: int = 90):
        super().__init__()
        self.JPEG_QUALITY = JPEG_QUALITY

    def set(self, input: NDArray) -> None:
        success, code_array = cv2.imencode('.jpg', input, [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])
        if not success:
            log.error('JPEG encoding failed')
            return
        with self._Packet__lock:
            self._Packet__frame_id += 1
            self._Packet__data = code_array.tobytes()

class __Output_Packet(__Packet):
    def __init__(self, JPEG_QUALITY: int = 80):
        super().__init__()
        self.HEADER       = Struct('<IdHHHHHHI')
        self.JPEG_QUALITY = JPEG_QUALITY

    def set(self, input: tuple[NDArray, NDArray, tuple[int, int, int, int]]) -> None:
        image, array, roi_info = input
        success, code_array = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])
        if not success:
            log.error('JPEG encoding failed')
            return
        code_byte = code_array.tobytes()
        with self._Packet__lock:
            self._Packet__frame_id += 1
            self._Packet__data = self.HEADER.pack(
                self._Packet__frame_id, time() * 1000, *roi_info, *array.shape, len(code_byte)
            ) + array.astype('<f4').tobytes() + code_byte

input  = __Input_Packet()
weight = __Input_Packet()
output = __Output_Packet()
