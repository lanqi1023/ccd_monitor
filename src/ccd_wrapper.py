from ccd import CCD

from numpy.typing import NDArray
from time import monotonic
from typing import Callable

class CCD_Wrapper:
    def __init__(
        self,
        process: Callable[[NDArray], tuple[NDArray, NDArray, tuple[int, int, int, int]]],
        publish: Callable[[tuple[NDArray, NDArray, tuple[int, int, int, int]]], None],
        frame_rate: float = 10
    ):
        self.__ccd  = CCD()
        self.__open = False
        self.__process    = process
        self.__publish    = publish
        self.__frame_rate = frame_rate
        self.__last_time = 0.0

    def __on_capture(self, bayer: NDArray) -> None:
        now = monotonic()
        if now - self.__last_time >= 1.0 / self.__frame_rate:
            self.__last_time = now
            self.__publish(self.__process(bayer))

    def start(self) -> None:
        if self.__ccd.open():
            self.__ccd.frame_rate = self.__frame_rate
            self.__ccd.process    = CCD.PROCESS_NONE
            self.__last_time      = 0.0
            if self.__ccd.begin_thread(on_capture = self.__on_capture):
                self.__open = True

    def stop(self) -> None:
        if self.__open:
            self.__ccd.end_thread()
            self.__ccd.close()
            self.__open = False
