import os
import sys
from pathlib import Path
GALAXY_SDK_DEVELOPMENT = os.getenv('GALAXY_SDK_DEVELOPMENT')
if not GALAXY_SDK_DEVELOPMENT:
    raise RuntimeError('GALAXY_SDK_DEVELOPMENT is not configured')
PATH_TO_GXIPY = Path(GALAXY_SDK_DEVELOPMENT) / 'Samples' / 'Python'
if not PATH_TO_GXIPY.is_dir():
    raise RuntimeError(f'gxipy path not found: {PATH_TO_GXIPY}')
sys.path.insert(0, str(PATH_TO_GXIPY))

import gxipy

import cv2
import logging
import numpy as np
from numpy.typing import NDArray
from threading import Thread, Event, Lock
from typing import Callable, Iterator

class CCD:
    '''
    Daheng camera controller through gxipy SDK
    ---

    0. Config logger and open camera:

        logging.basicConfig(
            level  = logging.INFO,
            format = '[%(levelname).1s] %(message)s'
        )
        ccd = CCD()
        if ccd.open():
            ...
            ccd.close()

    1. Iterator mode:

        images = ccd.get_iterator()
        try:
            for image in images:
                ...
        finally:
            images.close()

    Pass an integer to ``get_iterator(N = ...)`` to acquire a finite number of frames

    2. Thread mode:

        ccd.begin_thread(on_capture) # callback is optional
        image = ccd.thread_image     # thread-safe latest frame
        ccd.end_thread()

    3. Get / Set camera parameters
        ccd.format     = 'BayerRG8' # or 'BayerRG12'
        ccd.size       = (3072, 4096) # (height, width)
        ccd.exposure   = 1000 # us
        ccd.frame_rate = 10 # fps
        ccd.process    = CCD.PROCESS_RG8 # or others
    '''

    PROCESS_NONE      = lambda bayer: bayer if (bayer.dtype == np.uint8) else (bayer >> 4).astype(np.uint8)
    PROCESS_RG8       = lambda bayer: cv2.cvtColor(bayer, cv2.COLOR_BAYER_RGGB2BGR)
    PROCESS_RG12      = lambda bayer: cv2.cvtColor((bayer >> 4).astype(np.uint8), cv2.COLOR_BAYER_RGGB2BGR)
    PROCESS_RG8_GRAY  = lambda bayer: cv2.cvtColor(bayer, cv2.COLOR_BAYER_RGGB2GRAY)
    PROCESS_RG12_GRAY = lambda bayer: cv2.cvtColor((bayer >> 4).astype(np.uint8), cv2.COLOR_BAYER_RGGB2GRAY)

    def __init__(self):
        self.process: Callable[[NDArray], NDArray] = CCD.PROCESS_RG8
        self.__manager: gxipy.DeviceManager  | None = None
        self.__camera:  gxipy.Device         | None = None
        self.__feature: gxipy.FeatureControl | None = None
        self.__thread:       Thread  = None
        self.__begin_event:  Event   = Event()
        self.__end_event:    Event   = Event()
        self.__begin_flag:   bool    = False
        self.__image:        NDArray = None
        self.__image_lock:   Lock    = Lock()
        self.__capture_lock: Lock    = Lock()
        self.log: logging.Logger = logging.getLogger(__name__)

    def open(self) -> bool:
        try:
            self.__manager = gxipy.DeviceManager()
            device_num, device_list = self.__manager.update_all_device_list()
            if device_num > 0:
                self.__camera  = self.__manager.open_device_by_index(1)
                self.__feature = self.__camera.get_remote_device_feature_control()
                self.__feature.get_enum_feature('UserSetSelector').set('Default')
                self.__feature.get_command_feature('UserSetLoad').send_command()
                self.log.info(f"open camera {device_list[0]['model_name']} on {device_list[0]['ip']}")
                return True
            else:
                self.log.error('camera open failed: no camera detected')
                return False
        except Exception as e:
            if self.__camera is not None:
                try:
                    self.log.error(f'camera initialization failed: {e}, closing...')
                    self.__camera.close_device()
                except Exception as close_error:
                    self.log.error(f'camera close failed: {close_error}')
            self.log.error(f'camera open failed: {e}')
            self.__manager = None
            self.__camera  = None
            self.__feature = None
            return False

    def close(self) -> bool:
        success = True
        try:
            if self.__camera is not None:
                self.__camera.close_device()
                self.log.info('camera closed')
        except Exception as e:
            success = False
            self.log.error(f'camera close failed: {e}')
        finally:
            self.__manager = None
            self.__camera  = None
            self.__feature = None
        return success

    @property
    def format(self) -> str | None:
        '''Pixel format, ``BayerRG8`` or ``BayerRG12``.'''
        try:
            return self.__feature.get_enum_feature('PixelFormat').get()[1]
        except Exception as e:
            self.log.error(f'pixel format read failed: {e}')
            return None

    @format.setter
    def format(self, format: str) -> None:
        try:
            self.__feature.get_enum_feature('PixelFormat').set(format)
        except Exception as e:
            raise RuntimeError(f'set format to {format} failed') from e

    @property
    def size(self) -> tuple[int, int] | None:
        '''Image size ``(height, width)``, 16 ≤ height ≤ 3072, 16 ≤ width ≤ 4096; both must be divisible by 16.
        '''
        try:
            return self.__feature.get_int_feature('Height').get(), self.__feature.get_int_feature('Width').get()
        except Exception as e:
            self.log.error(f'camera size read failed: {e}')
            return None

    @size.setter
    def size(self, size: tuple[int, int]) -> None:
        try:
            self.__feature.get_int_feature('Height').set(size[0])
            self.__feature.get_int_feature('Width').set(size[1])
        except Exception as e:
            raise RuntimeError(f'set size to {size} failed') from e

    @property
    def exposure(self) -> float | None:
        '''Exposure time in microseconds, 37us ≤ exposure ≤ 1,000,000us.'''
        try:
            return self.__feature.get_float_feature('ExposureTime').get()
        except Exception as e:
            self.log.error(f'exposure read failed: {e}')
            return None

    @exposure.setter
    def exposure(self, exposure: float) -> None:
        try:
            self.__feature.get_float_feature('ExposureTime').set(float(exposure))
        except Exception as e:
            raise RuntimeError(f'set exposure to {exposure} failed') from e

    @property
    def frame_rate(self) -> float | None:
        '''Current acquisition frame rate.'''
        try:
            return self.__feature.get_float_feature('CurrentAcquisitionFrameRate').get()
        except Exception as e:
            self.log.error(f'frame rate read failed: {e}')
            return None

    @frame_rate.setter
    def frame_rate(self, frame_rate: float) -> None:
        try:
            self.__feature.get_enum_feature('AcquisitionFrameRateMode').set('On')
            self.__feature.get_float_feature('AcquisitionFrameRate').set(float(frame_rate))
        except Exception as e:
            raise RuntimeError(f'set frame_rate to {frame_rate} failed') from e

    def __grab(self) -> NDArray[np.uint8 | np.uint16] | None:
        buf = None
        try:
            buf = self.__camera.data_stream[0].dq_buf(timeout = 1000)
            image = buf.get_numpy_array().copy()
        except Exception as e:
            image = None
            self.log.error(f'frame acquisition failed: {e}')
        finally:
            if buf is not None:
                self.__camera.data_stream[0].q_buf(buf)
        return image

    def get_iterator(self, N: int | None = None, flush: bool = False) -> Iterator[NDArray[np.uint8 | np.uint16]]:
        '''Yield ``N`` processed frames, or indefinitely when ``N`` is ``None``.

        ``flush`` discards queued frames before each capture.
        '''
        if not self.__capture_lock.acquire(blocking = False):
            raise RuntimeError('camera is already capturing')
        image       = None
        stream_flag = False
        try:
            self.__camera.stream_on()
            stream_flag = True
            n = 0
            while N is None or n < N:
                if flush:
                    self.__camera.data_stream[0].flush_queue()
                image = self.__grab()
                if image is not None:
                    yield self.process(image)
                    n += 1
        finally:
            try:
                if stream_flag:
                    self.__camera.stream_off()
            except Exception as e:
                self.log.error(f'camera stream stop failed: {e}')
            finally:
                self.__capture_lock.release()

    def __target(self, on_capture: Callable[..., None] | None, **kwargs) -> None:
        lock_flag   = False
        stream_flag = False
        try:
            if not self.__capture_lock.acquire(blocking = False):
                raise RuntimeError('camera is already capturing')
            lock_flag = True
            self.__camera.stream_on()
            stream_flag = True
            self.__begin_flag = True
            self.__begin_event.set()

            while not self.__end_event.is_set():
                image = self.__grab()
                if image is not None:
                    image = self.process(image)
                    with self.__image_lock:
                        self.__image = image
                    if on_capture is not None:
                        on_capture(image, **kwargs)
        except Exception as e:
            self.log.error(f'capture thread stopped unexpectedly: {e}')
        finally:
            self.__begin_event.set()
            try:
                if stream_flag:
                    self.__camera.stream_off()
            except Exception as e:
                self.log.error(f'camera stream stop failed: {e}')
            finally:
                if lock_flag:
                    self.__capture_lock.release()

    @property
    def thread_image(self) -> NDArray | None:
        '''Return the latest processed frame published by the capture thread.'''
        with self.__image_lock:
            return self.__image

    @property
    def is_thread_running(self) -> bool:
        '''Whether the background capture thread is currently alive.'''
        return self.__thread is not None and self.__thread.is_alive()

    def begin_thread(self, on_capture: Callable[..., None] | None = None, **kwargs) -> bool:
        '''Start background capture, optionally invoking ``on_capture`` per frame.'''
        if self.__thread is not None:
            if self.__thread.is_alive():
                self.log.error('a capture thread already exists')
                return False
            else:
                self.__thread.join()
        with self.__image_lock:
            self.__image = None
        self.__begin_event.clear()
        self.__end_event.clear()
        self.__begin_flag = False
        self.__thread = Thread(
            target = self.__target,
            kwargs = {'on_capture': on_capture, **kwargs},
            daemon = True
        )
        try:
            self.__thread.start()
        except Exception as e:
            self.__thread = None
            self.log.error(f'capture thread start failed: {e}')
            return False

        self.__begin_event.wait()
        if self.__begin_flag:
            self.log.info('capture thread started')
            return True
        else:
            self.__thread.join()
            self.__thread = None
            return False

    def end_thread(self, timeout: float = 3.0) -> bool:
        '''Request background capture to stop and wait for the thread to exit.'''
        if self.__thread is None:
            return True
        elif self.__thread.is_alive():
            self.__end_event.set()
            self.__thread.join(timeout)

        if self.__thread.is_alive():
            self.log.error(f'capture thread did not stop within {timeout}s')
            return False
        else:
            self.__thread = None
            self.log.info('capture thread stopped')
            return True

    def list_feature(self) -> None:
        '''Print all implemented features and their current values or ranges.'''
        for name, feature in vars(self.__camera).items():
            if isinstance(feature, gxipy.Feature):
                if feature.is_implemented():
                    if isinstance(feature, gxipy.IntFeature):
                        print(name, '= '
                            f"{feature.get() if feature.is_readable() else '?'} "
                            f"{feature.get_range().get('unit, ', '')}"
                            f"in [{feature.get_range().get('min')}, "
                            f"{feature.get_range().get('max')}]")
                    if isinstance(feature, gxipy.FloatFeature):
                        print(name, '= '
                            f"{feature.get() if feature.is_readable() else '?'} "
                            f"{feature.get_range().get('unit, ', '')}"
                            f"in [{feature.get_range().get('min')}, "
                            f"{feature.get_range().get('max')}]")
                    elif isinstance(feature, gxipy.EnumFeature):
                        print(name, '= '
                            f"{feature.get()[1] if feature.is_readable() else '?'} "
                            f"in {list(feature.get_range())}")
                    elif isinstance(feature, gxipy.BoolFeature):
                        print(name, '= '
                            f"{feature.get() if feature.is_readable() else '?'}")
                    elif isinstance(feature, gxipy.StringFeature):
                        print(name, '= '
                            f"'{feature.get() if feature.is_readable() else '?'}'")
                    elif isinstance(feature, gxipy.BufferFeature):
                        print('(buffer)', name)

if __name__ == '__main__':
    logging.basicConfig(
        level  = logging.INFO,
        format = '[%(levelname).1s] %(message)s'
    )

    cv2.namedWindow('ccd', cv2.WINDOW_NORMAL)
    ccd = CCD()
    if ccd.open():
        try:
            # ccd.list_feature()

            for image in ccd.get_iterator(3):
                cv2.imshow('ccd', image)
                cv2.waitKey(0)

            images = ccd.get_iterator()
            try:
                while (cv2.waitKey(1) != 27):
                    cv2.imshow('ccd', image)
                    image = next(images)
            finally:
                images.close()

            ccd.begin_thread()
            while True:
                image = ccd.thread_image
                if image is not None:
                    cv2.imshow('ccd', image)
                    if (cv2.waitKey(1) == 27):
                        break
        finally:
            if ccd.end_thread():
                ccd.close()
            cv2.destroyWindow('ccd')
