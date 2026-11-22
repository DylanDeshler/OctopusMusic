"""WebSocket link to the GPU server, with reconnect and RTT measurement."""

from __future__ import annotations

import asyncio
import json
import time

import aiohttp

from octomusic import protocol


class GPULink:
    def __init__(self, url: str, on_message, on_audio):
        self.url = url
        self.on_message = on_message      # f(dict)
        self.on_audio = on_audio          # f(seq, start, sr, pcm)
        self.ws = None
        self.state = "disconnected"
        self.error = ""
        self.rtt_ms = None
        self.server_info = None
        self.bytes_in = 0
        self._task = None

    def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._task = None
        self.state = "disconnected"

    async def send(self, obj) -> bool:
        if self.ws is None or self.ws.closed:
            return False
        try:
            await self.ws.send_str(protocol.dumps(obj))
            return True
        except (ConnectionResetError, RuntimeError, aiohttp.ClientError):
            return False

    async def _pinger(self):
        while True:
            await self.send({"type": "ping", "t": time.time()})
            await asyncio.sleep(2.0)

    async def _run(self):
        backoff = 1.0
        while True:
            self.state = "connecting"
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.ws_connect(self.url, heartbeat=20,
                                            max_msg_size=64 << 20) as ws:
                        self.ws, self.state, self.error, backoff = ws, "connected", "", 1.0
                        await self.send({"type": "hello"})
                        pinger = asyncio.create_task(self._pinger())
                        try:
                            async for msg in ws:
                                if msg.type == aiohttp.WSMsgType.BINARY:
                                    self.bytes_in += len(msg.data)
                                    self.on_audio(*protocol.decode_audio(msg.data))
                                elif msg.type == aiohttp.WSMsgType.TEXT:
                                    self.bytes_in += len(msg.data)
                                    m = json.loads(msg.data)
                                    if m.get("type") == "pong":
                                        self.rtt_ms = (time.time() - m["t"]) * 1000
                                        continue
                                    if m.get("type") == "server_info":
                                        self.server_info = m
                                    self.on_message(m)
                        finally:
                            pinger.cancel()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self.error = f"{type(e).__name__}: {e}"
            self.ws = None
            self.state = "disconnected"
            self.on_message({"type": "link", "state": "disconnected", "error": self.error})
            await asyncio.sleep(backoff)
            backoff = min(10.0, backoff * 1.7)

    def info(self):
        return {"url": self.url, "state": self.state, "error": self.error,
                "rtt_ms": None if self.rtt_ms is None else round(self.rtt_ms, 1),
                "kb_in": self.bytes_in // 1024}
