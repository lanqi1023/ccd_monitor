from ccd_wrapper import Config, Status, ccd_wrapper
from main import main_target
from packet import input, weight, output

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from threading import Thread

HTML_DIR = Path(__file__).parent.parent / 'html'

@asynccontextmanager
async def lifespan(_app: FastAPI):
    main_thread = Thread(target = main_target, daemon = True)
    main_thread.start()
    try:
        ccd_wrapper.start()
        yield
    finally:
        try:
            ccd_wrapper.stop()
        except RuntimeError:
            logging.exception('camera shutdown failed')
app = FastAPI(lifespan = lifespan)
app.mount('/static', StaticFiles(directory = HTML_DIR / 'static'), name = 'static')

@app.get('/')
async def index():
    return FileResponse(HTML_DIR / 'index.html')

@app.get('/api/camera', response_model = Status)
def camera_status():
    return ccd_wrapper.status()

@app.post('/api/camera/start', response_model = Status)
def camera_start():
    try:
        return ccd_wrapper.start()
    except RuntimeError as e:
        raise HTTPException(status_code = 503, detail = str(e)) from e

@app.post('/api/camera/pause', response_model = Status)
def camera_pause():
    try:
        return ccd_wrapper.pause()
    except RuntimeError as e:
        raise HTTPException(status_code = 409, detail = str(e)) from e

@app.post('/api/camera/stop', response_model = Status)
def camera_stop():
    try:
        return ccd_wrapper.stop()
    except RuntimeError as e:
        raise HTTPException(status_code = 409, detail = str(e)) from e

@app.patch('/api/camera/config', response_model = Status)
def camera_update(config: Config):
    try:
        return ccd_wrapper.update(config)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code = 422, detail = str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code = 409, detail = str(e)) from e

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    async def send_frames():
        last_frame_id = -1
        while True:
            frame_id, data = output.get()
            if frame_id != last_frame_id:
                await websocket.send_bytes(data)
                last_frame_id = frame_id
            await asyncio.sleep(0.01)

    async def wait_for_disconnect():
        while True:
            message = await websocket.receive()
            if message['type'] == 'websocket.disconnect':
                return

    await websocket.accept()
    sender   = asyncio.create_task(send_frames())
    receiver = asyncio.create_task(wait_for_disconnect())

    try:
        await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        sender.cancel()
        receiver.cancel()
        await asyncio.gather(sender, receiver, return_exceptions = True)

def image_response(frame_id: int, image: bytes | None, current_id: int) -> Response:
    headers = {
        'Cache-Control': 'no-store',
        'X-Frame-Id': str(frame_id)
    }
    if image is None:
        return Response(status_code = 404, headers = headers)
    elif frame_id == current_id:
        return Response(status_code = 204, headers = headers)
    else:
        return Response(content = image, media_type = 'image/jpeg', headers = headers)

@app.get('/api/input.jpg')
def api_input(current_id: int = -1):
    return image_response(*input.get(), current_id)

@app.get('/api/weight.jpg')
def api_weight(current_id: int = -1):
    return image_response(*weight.get(), current_id)

@app.get('/input')
@app.get('/weight')
async def image_page():
    return FileResponse(HTML_DIR / 'image.html')

if __name__ == '__main__':
    import uvicorn
    logging.basicConfig(
        level  = logging.INFO,
        format = '[%(levelname).1s] %(message)s'
    )
    uvicorn.run(app, host = '0.0.0.0', port = 8000)
