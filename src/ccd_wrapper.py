from ccd import CCD
from packet import output

import cv2
import numpy as np
from dataclasses import dataclass, asdict
from math import isfinite
from numpy.typing import NDArray
from threading import RLock
from time import monotonic
from typing import Callable, Literal

@dataclass
class Status():
    opened:     bool
    running:    bool
    format:     Literal['BayerRG8', 'BayerRG12']
    size:       tuple[int, int]
    exposure:   float
    frame_rate: float

    roi_display: bool
    roi_origin:  tuple[int, int]
    roi_size:    tuple[int, int]
    array_size:  tuple[int, int]

@dataclass
class Config():
    format:     Literal['BayerRG8', 'BayerRG12'] | None = None
    size:       tuple[int, int]                  | None = None
    exposure:   float                            | None = None
    frame_rate: float                            | None = None

    roi_display: bool            | None = None
    roi_origin:  tuple[int, int] | None = None
    roi_size:    tuple[int, int] | None = None
    array_size:  tuple[int, int] | None = None

CAMERA_FIELDS = {
    'format',
    'size',
    'exposure',
    'frame_rate'
}

class __CCD_Wrapper:
    def __init__(self, publish: Callable[[NDArray, NDArray], None]):
        self.__ccd  = CCD()
        self.__lock = RLock()
        self.__publish = publish
        self.__last_time = 0.0

        self.__opened = False
        self.__config = Config(
            format     = 'BayerRG8',
            size       = (3072, 4096),
            exposure   = 10000,
            frame_rate = 10,

            roi_display = True,
            roi_origin  = (768, 1024),
            roi_size    = (1536, 2048),
            array_size  = (48, 64)
        )

    def __process(self, bayer: NDArray) -> tuple[NDArray, NDArray]:
        bgr  = cv2.cvtColor(bayer, cv2.COLOR_BAYER_RGGB2BGR)
        gray = cv2.cvtColor(bayer, cv2.COLOR_BAYER_RGGB2GRAY)

        g = np.empty((bayer.shape[0], bayer.shape[1] // 2), dtype = np.uint8)
        g[0::2] = bayer[0::2, 1::2]
        g[1::2] = bayer[1::2, 0::2]

        # get roi
        roi_row,    roi_col   = self.__config.roi_origin
        roi_height, roi_width = self.__config.roi_size
        roi = gray[
            roi_row : roi_row + roi_height,
            roi_col : roi_col + roi_width
        ]

        # get array
        array_height, array_width = self.__config.array_size
        array = roi.reshape(
            array_height, roi_height // array_height,
            array_width,  roi_width  // array_width
        ).mean(axis=(1, 3), dtype=np.float32)

        # plot roi
        if self.__config.roi_display:
            cv2.rectangle(bgr, (roi_col, roi_row), (roi_col + roi_width - 1, roi_row + roi_height - 1), [255, 255, 255], 2)
            cv2.rectangle(gray, (roi_col, roi_row), (roi_col + roi_width - 1, roi_row + roi_height - 1), 255, 2)
        return bgr, array

    def __on_capture(self, image: NDArray) -> None:
        now = monotonic()
        if now - self.__last_time >= 1.0 / self.__config.frame_rate:
            self.__last_time = now
            self.__publish(*self.__process(image))

    def __validate(self, config: Config) -> Config:
        validated = Config()

        if config.format is not None:
            if config.format in ('BayerRG8', 'BayerRG12'):
                if config.format != self.__config.format:
                    validated.format = config.format
            else:
                raise ValueError('invalid format')

        if config.exposure is not None:
            if (isinstance(config.exposure, (int, float))
                and 37 <= config.exposure <= 1_000_000
            ):
                if config.exposure != self.__config.exposure:
                    validated.exposure = config.exposure
            else:
                raise ValueError('invalid exposure')

        if config.frame_rate is not None:
            if (isinstance(config.frame_rate, (int, float))
                and not isinstance(config.frame_rate, bool)
                and config.frame_rate > 0
                and isfinite(config.frame_rate)
            ):
                if config.frame_rate != self.__config.frame_rate:
                    validated.frame_rate = config.frame_rate
            else:
                raise ValueError('invalid frame_rate')

        if config.roi_display is not None:
            if isinstance(config.roi_display, bool):
                if config.roi_display != self.__config.roi_display:
                    validated.roi_display = config.roi_display
            else:
                raise TypeError('invalid roi_display type')

        image_height, image_width = self.__config.size
        roi_row,      roi_col     = self.__config.roi_origin
        roi_height,   roi_width   = self.__config.roi_size
        array_height, array_width = self.__config.array_size

        if config.size is not None:
            if (isinstance(config.size, tuple)
                and len(config.size) == 2
            ):
                height, width = config.size
                if (isinstance(height, int) and isinstance(width, int)
                    and 16 <= height <= 3072 and 16 <= width <= 4096
                    and height % 16 == 0 and width % 16 == 0
                ):
                    image_height, image_width = height, width
                else:
                    raise ValueError('invalid size')
            else:
                raise TypeError('invalid size type')

        if config.roi_origin is not None:
            if (isinstance(config.roi_origin, tuple)
                and len(config.roi_origin) == 2
            ):
                row, col = config.roi_origin
                if (isinstance(row, int) and isinstance(col, int)
                    and not isinstance(row, bool) and not isinstance(col, bool)
                    and 0 <= row < 3072 and 0 <= col < 4096
                ):
                    roi_row, roi_col = row, col
                else:
                    raise ValueError('invalid roi_origin')
            else:
                raise TypeError('invalid roi_origin type')

        if config.roi_size is not None:
            if (isinstance(config.roi_size, tuple)
                and len(config.roi_size) == 2
            ):
                height, width = config.roi_size
                if (isinstance(height, int) and isinstance(width, int)
                    and not isinstance(height, bool) and not isinstance(width, bool)
                    and 0 < height <= 3072 and 0 < width <= 4096
                ):
                    roi_height, roi_width = height, width
                else:
                    raise ValueError('invalid roi_size')
            else:
                raise TypeError('invalid roi_size type')

        if config.array_size is not None:
            if (isinstance(config.array_size, tuple)
                and len(config.array_size) == 2
            ):
                height, width = config.array_size
                if (isinstance(height, int) and isinstance(width, int)
                    and not isinstance(height, bool) and not isinstance(width, bool)
                    and 0 < height <= 3072 and 0 < width <= 4096
                ):
                    array_height, array_width = height, width
                else:
                    raise ValueError('invalid array_size')
            else:
                raise TypeError('invalid array_size type')

        if (roi_row + roi_height <= image_height and roi_col + roi_width <= image_width
            and array_height <= roi_height and array_width <= roi_width
        ):
            roi_height, roi_width = (roi_height // array_height) * array_height, (roi_width // array_width) * array_width
            if (image_height, image_width) != self.__config.size:
                validated.size = image_height, image_width
            if (roi_row, roi_col) != self.__config.roi_origin:
                validated.roi_origin = roi_row, roi_col
            if (roi_height, roi_width) != self.__config.roi_size:
                validated.roi_size = roi_height, roi_width
            if (array_height, array_width) != self.__config.array_size:
                validated.array_size = array_height, array_width
        else:
            raise ValueError('invalid image_size / roi_size / array_size')

        return validated

    def status(self) -> Status:
        with self.__lock:
            if self.__opened:
                return Status(
                    opened     = True,
                    running    = self.__ccd.is_thread_running,
                    format     = self.__ccd.format     or self.__config.format,
                    size       = self.__ccd.size       or self.__config.size,
                    exposure   = self.__ccd.exposure   or self.__config.exposure,
                    frame_rate = self.__ccd.frame_rate or self.__config.frame_rate,

                    roi_display = self.__config.roi_display,
                    roi_origin  = self.__config.roi_origin,
                    roi_size    = self.__config.roi_size,
                    array_size  = self.__config.array_size
                )
            else:
                return Status(
                    opened     = False,
                    running    = False,
                    format     = self.__config.format,
                    size       = self.__config.size,
                    exposure   = self.__config.exposure,
                    frame_rate = self.__config.frame_rate,

                    roi_display = self.__config.roi_display,
                    roi_origin  = self.__config.roi_origin,
                    roi_size    = self.__config.roi_size,
                    array_size  = self.__config.array_size
                )

    def start(self) -> Status:
        with self.__lock:
            if self.__opened or self.__ccd.open():
                self.__opened = True
                if not self.__ccd.is_thread_running:
                    self.__ccd.format     = self.__config.format
                    self.__ccd.size       = self.__config.size
                    self.__ccd.exposure   = self.__config.exposure
                    self.__ccd.frame_rate = self.__config.frame_rate
                    self.__ccd.process    = CCD.PROCESS_NONE
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
        with self.__lock:
            config_dict = {
                k: v for k, v in asdict(self.__validate(config)).items() if v is not None
            }
            if not config_dict:
                return self.status()
            else:
                restart_flag = self.__ccd.is_thread_running
                if restart_flag:
                    self.pause()
                for k, v in config_dict.items():
                    setattr(self.__config, k, v)
                if restart_flag:
                    self.start()
                elif self.__opened:
                    for k, v in config_dict.items():
                        if k in CAMERA_FIELDS:
                            setattr(self.__ccd, k, v)
                return self.status()

ccd_wrapper = __CCD_Wrapper(publish = output.set)
