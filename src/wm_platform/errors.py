from __future__ import annotations

from collections.abc import Mapping


class AppError(Exception):
    def __init__(
        self,
        error_code: str,
        error_message: str,
        status_code: int = 400,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(error_message)
        self.error_code = error_code
        self.error_message = error_message
        self.status_code = status_code
        self.headers = dict(headers or {})


def error_payload(error_code: str, error_message: str) -> dict[str, str]:
    return {
        "error_code": error_code,
        "error_message": error_message,
    }
