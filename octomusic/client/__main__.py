"""Laptop entrypoint.

    # 1. tunnel to the GPU box (leave running)
    ssh -N -L 8765:localhost:8765 user@gpu-box
    # 2. start the client, open http://localhost:8080
    python -m octomusic.client --source octopus          # Bluetooth
    python -m octomusic.client --source octopus-usb      # USB-C (octomusic firmware)
    python -m octomusic.client --source synthetic          # no headset
    python -m octomusic.client --source replay --replay recordings/20261122-170000
"""

from __future__ import annotations

import argparse
import os
import webbrowser

from aiohttp import web

from octomusic import protocol
from octomusic.client.app import App, build_web


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=f"ws://localhost:{protocol.DEFAULT_PORT}/ws",
                    help="GPU server WebSocket URL (default: through the SSH tunnel)")
    ap.add_argument("--token", default=os.environ.get("OCTOMUSIC_TOKEN", ""))
    ap.add_argument("--source", choices=["octopus", "octopus-usb", "synthetic", "replay"],
                    default="synthetic")
    ap.add_argument("--replay", help="recording folder or eeg.csv for --source replay")
    ap.add_argument("--ble-address", help="skip scanning; connect to this address/UUID")
    ap.add_argument("--usb-port", help="serial port for --source octopus-usb (auto-detected)")
    ap.add_argument("--audio-device", help="sounddevice output device name or index")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    url = args.server + (("&" if "?" in args.server else "?") + f"token={args.token}"
                         if args.token else "")
    dev = args.audio_device
    if dev is not None and dev.isdigit():
        dev = int(dev)
    app = App(url, audio_device=dev, source=args.source, replay_path=args.replay,
              ble_address=args.usb_port if args.source == "octopus-usb" else args.ble_address)
    if not args.no_browser:
        webbrowser.open(f"http://localhost:{args.port}")
    print(f"[client] UI on http://localhost:{args.port}   server {args.server}")
    web.run_app(build_web(app), host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
