from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from wm_platform.api_app import create_app

app = create_app()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanplate API process")
    parser.add_argument("--host", default=os.getenv("DWM_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("DWM_API_PORT", "8000")))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    uvicorn.run("apps.api.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
