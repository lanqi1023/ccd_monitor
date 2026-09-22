from ccd import CCD
from main import ROI_ORIGIN, ROI_SHAPE, process
from packet import output

from math import isfinite
from numpy.typing import NDArray
from threading import RLock
from time import monotonic
from typing import Callable, Literal, TypedDict

class Status(TypedDict):
    opened:     bool
    running:    bool
    format:     Literal['BayerRG8', 'BayerRG12']
    size:       tuple[int, int]
    exposure:   float
    frame_rate: float

class Config(TypedDict, total = False):
    format:     Literal['BayerRG8', 'BayerRG12']
    size:       tuple[int, int]
    exposure:   float
    frame_rate: float

class __CCD_Wrapper:
    def __init__(
        self,
        process: Callable[[NDArray], tuple[NDArray, tuple[int, int, int, int]]],
        publish: Callable[[NDArray, NDArray, tuple[int, int, int, int]], None]
    ):
        self.__ccd  = CCD()
        self.__lock = RLock()
        self.__opened  = False
        self.__process = process
        self.__publish = publish
        self.__last_time = 0.0

        self.__format     = 'BayerRG8'
        self.__size       = (3072, 4096)
        self.__exposure   = 10000
        self.__frame_rate = 10

    def __on_capture(self, image: NDArray) -> None:
        now = monotonic()
        if now - self.__last_time >= 1.0 / self.__frame_rate:
            self.__last_time = now
            self.__publish(image, *self.__process(image))

    @staticmethod
    def __validate(config: Config) -> Config:
        validated = Config()

        if 'format' in config:
            if config['format'] in ('BayerRG8', 'BayerRG12'):
                validated['format'] = config['format']
            else:
                raise ValueError('invalid format')

        if 'size' in config:
            if (isinstance(config['size'], tuple)
                and len(config['size']) == 2
            ):
                height, width = config['size']
                roi_row, roi_col = ROI_ORIGIN
                roi_height, roi_width = ROI_SHAPE
                if (isinstance(height, int) and isinstance(width, int)
                    and 16 <= height <= 3072 and 16 <= width <= 4096
                    and height % 16 == 0 and width % 16 == 0
                    and height >= roi_row + roi_height and width >= roi_col + roi_width
                ):
                    validated['size'] = config['size']
                else:
                    raise ValueError('invalid size')
            else:
                raise TypeError('invalid size type')

        if 'exposure' in config:
            if (isinstance(config['exposure'], (int, float))
                and 37 <= config['exposure'] <= 1_000_000
            ):
                validated['exposure'] = config['exposure']
            else:
                raise ValueError('invalid exposure')

        if 'frame_rate' in config:
            if (isinstance(config['frame_rate'], (int, float))
                and not isinstance(config['frame_rate'], bool)
                and config['frame_rate'] > 0
                and isfinite(config['frame_rate'])
            ):
                validated['frame_rate'] = config['frame_rate']
            else:
                raise ValueError('invalid frame_rate')

        return validated

    def status(self) -> Status:
        with self.__lock:
            if self.__opened:
                return {
                    'opened':     True,
                    'running':    self.__ccd.is_thread_running,
                    'format':     self.__ccd.format     or self.__format,
                    'size':       self.__ccd.size       or self.__size,
                    'exposure':   self.__ccd.exposure   or self.__exposure,
                    'frame_rate': self.__ccd.frame_rate or self.__frame_rate
                }
            else:
                return {
                    'opened':     False,
                    'running':    False,
                    'format':     self.__format,
                    'size':       self.__size,
                    'exposure':   self.__exposure,
                    'frame_rate': self.__frame_rate
                }

    def start(self) -> Status:
        with self.__lock:
            if self.__opened or self.__ccd.open():
                self.__opened = True
                if not self.__ccd.is_thread_running:
                    self.__ccd.format     = self.__format
                    self.__ccd.size       = self.__size
                    self.__ccd.exposure   = self.__exposure
                    self.__ccd.frame_rate = self.__frame_rate
                    if self.__format == 'BayerRG8':
                        self.__ccd.process = CCD.PROCESS_RG8_GRAY
                    elif self.__format == 'BayerRG12':
                        self.__ccd.process = CCD.PROCESS_RG12_GRAY
                    else:
                        self.__ccd.process = CCD.PROCESS_NONE
                    self.__last_time = 0.0
                    if not self.__ccd.begin_thread(on_capture = self.__on_capture):
                        success = self.__ccd.close()
                        self.__opened = False
                        if not success:
                            raise RuntimeError('capture thread startup failed, then camera close failed')
                        raise RuntimeError('capture thread startup failed')
            else:
                raise RuntimeError('camera open failed')
            return self.status()

    def pause(self) -> Status:
        with self.__lock:
            if not self.__ccd.end_thread():
                raise RuntimeError('capture thread stop timed out')
            return self.status()

    def stop(self) -> Status:
        with self.__lock:
            if not self.__ccd.end_thread():
                raise RuntimeError('capture thread stop timed out')
            if self.__opened:
                success = self.__ccd.close()
                self.__opened = False
                if not success:
                    raise RuntimeError('camera close failed')
            return self.status()

    def update(self, config: Config) -> Status:
        config = self.__validate(config)
        with self.__lock:
            if not config:
                return self.status()
            else:
                restart_flag = self.__ccd.is_thread_running
                if restart_flag:
                    self.pause()
                for name, value in config.items():
                    setattr(self, f'_CCD_Wrapper__{name}', value)
                if restart_flag:
                    self.start()
                elif self.__opened:
                    for name, value in config.items():
                        setattr(self.__ccd, name, value)
                        if name == 'format':
                            if value == 'BayerRG8':
                                self.__ccd.process = CCD.PROCESS_RG8_GRAY
                            elif value == 'BayerRG12':
                                self.__ccd.process = CCD.PROCESS_RG12_GRAY
                            else:
                                self.__ccd.process = CCD.PROCESS_NONE
                return self.status()

ccd_wrapper = __CCD_Wrapper(process = process, publish = output.set)
