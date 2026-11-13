from __future__ import annotations

import argparse
import os

import uvicorn

from wm_platform.api_app import create_app

app = create_app()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanplate API process")
    parser.add_argument("--host", default=os.getenv("DWM_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("DWM_API_PORT", "8000")))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
