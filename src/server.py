from ccd_manager import CCD_Manager

import asyncio
import cv2
import logging
import struct
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from numpy.typing import NDArray
from pathlib import Path
from threading import Lock
from typing import Optional

log = logging.getLogger()

class Packet:
    def __init__(self):
        self.HEADER = struct.Struct('<IdHHI')
        self.__lock     = Lock()
        self.__frame_id = -1
        self.__data     = None

    def get(self) -> tuple[int, Optional[bytes]]:
        with self.__lock:
            return self.__frame_id, self.__data

    def set(self, image: NDArray, array: NDArray, JPEG_QUALITY: int = 80) -> None:
        success, code_array = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not success:
            log.error('JPEG encoding failed')
            return
        code_byte = code_array.tobytes()
        with self.__lock:
            self.__frame_id += 1
            self.__data = self.HEADER.pack(
                self.__frame_id, time.time() * 1000, *array.shape, len(code_byte)
            ) + array.astype('<f4').tobytes() + code_byte

packet      = Packet()
ccd_manager = CCD_Manager()

@asynccontextmanager
async def lifespan(_app: FastAPI):
    ccd_manager.start(on_publish = packet.set)
    try:
        yield
    finally:
        ccd_manager.stop()
app = FastAPI(lifespan = lifespan)

@app.get('/')
async def index():
    return FileResponse(Path(__file__).parent.parent / 'index.html')

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    log.info(f'web client connected')
    last_frame_id = -1
    try:
        while True:
            frame_id, data = packet.get()
            if frame_id != last_frame_id:
                await websocket.send_bytes(data)
                last_frame_id = frame_id
            await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.info(f'web client disconnected: {e}')
    finally:
        log.info(f'web client closed')

if __name__ == '__main__':
    import uvicorn
    logging.basicConfig(
        level  = logging.INFO,
        format = '[%(levelname).1s] %(message)s',
    )
    uvicorn.run(app, host = '127.0.0.1', port = 8000)
