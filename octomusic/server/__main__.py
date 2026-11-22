"""GPU server entrypoint.

    NMG_PATH=~/Neural_Music_Generator python -m octomusic.server
    NMG_PATH=... python -m octomusic.server --mock        # no GPU, fake audio

Binds to 127.0.0.1 by default: reach it from the laptop through
    ssh -N -L 8765:localhost:8765 user@gpu-box
so nothing is exposed publicly. If you do bind 0.0.0.0, set --token.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from aiohttp import WSMsgType, web

from octomusic import protocol
from octomusic.server.engine import Engine, Session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=protocol.DEFAULT_PORT)
    ap.add_argument("--mock", action="store_true",
                    help="use NMG's MockBackend: no torch, no GPU")
    ap.add_argument("--token", default="", help="shared secret (optional)")
    args = ap.parse_args()

    engine = Engine(mock=args.mock)
    current = {"session": None}

    async def ws_handler(request):
        if args.token and request.query.get("token") != args.token:
            raise web.HTTPForbidden()
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=256 << 20)
        await ws.prepare(request)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def send(obj):                       # called from the generation thread
            loop.call_soon_threadsafe(q.put_nowait, obj)

        async def writer():
            while True:
                obj = await q.get()
                if obj is None:
                    return
                try:
                    if isinstance(obj, tuple) and obj[0] == "audio":
                        _, seq, start, chunk, sr = obj
                        await ws.send_bytes(protocol.encode_audio(seq, start, chunk, sr))
                    else:
                        await ws.send_str(protocol.dumps(obj))
                except (ConnectionResetError, RuntimeError):
                    return

        wtask = asyncio.create_task(writer())
        # One live session per server: a new client takes over the GPU.
        if current["session"] is not None:
            await asyncio.to_thread(current["session"].stop)
        sess = Session(engine, send)
        current["session"] = sess
        send({"type": "server_info", **engine.info()})
        print(f"[server] client connected from {request.remote}", flush=True)
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    m = json.loads(msg.data)
                except ValueError:
                    continue
                t = m.get("type")
                if t == "dials":
                    if sess.mode == "online" or sess.state in ("idle", "done", "error"):
                        sess.cond.push(m.get("t", time.time()), m.get("values", {}),
                                       m.get("chroma"))
                elif t == "playback":
                    sess.on_playback(m.get("played", 0))
                elif t == "anchor":
                    sess.on_anchor(m["sample"], m["t"])
                elif t == "params":
                    sess.set_params(m.get("params", {}))
                elif t == "start":
                    if sess.mode == "offline":      # live frames start fresh
                        sess.cond = type(sess.cond)(snap_bpm=engine.B.snap_bpm)
                        sess._style_key = None
                    await asyncio.to_thread(sess.start, m.get("params", {}), "online",
                                            m.get("t_start"))
                elif t == "offline_start":
                    await asyncio.to_thread(sess.start, m.get("params", {}), "offline",
                                            None, m.get("frames", []))
                elif t == "stop":
                    await asyncio.to_thread(sess.stop)
                elif t == "ping":
                    send({"type": "pong", "t": m.get("t"), "server_t": time.time()})
        finally:
            await asyncio.to_thread(sess.stop)
            if current["session"] is sess:
                current["session"] = None
            q.put_nowait(None)
            await wtask
            print("[server] client disconnected", flush=True)
        return ws

    async def health(_):
        return web.json_response({"ok": True, **{k: v for k, v in engine.info().items()
                                                 if k != "defaults"}})

    app = web.Application()
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/health", health)
    print(f"[server] listening on ws://{args.host}:{args.port}/ws", flush=True)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
