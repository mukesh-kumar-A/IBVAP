import json
import threading
import asyncio
from fastapi import WebSocket


class WebSocketManager:
    """
    Thread-safe WebSocket manager.
    Handles connections from FastAPI event loop and allows non-blocking
    broadcasts from arbitrary background threads.
    """
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self._lock:
            self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def _async_send_to_all(self, payload: str):
        with self._lock:
            conns = list(self.active_connections)

        stale = []
        for connection in conns:
            try:
                await connection.send_text(payload)
            except Exception:
                stale.append(connection)

        if stale:
            with self._lock:
                for s in stale:
                    if s in self.active_connections:
                        self.active_connections.remove(s)

    def broadcast_from_thread(self, data: dict):
        if not self.active_connections or self._loop is None or self._loop.is_closed():
            return
        payload = json.dumps(data)
        asyncio.run_coroutine_threadsafe(self._async_send_to_all(payload), self._loop)


ws_manager = WebSocketManager()