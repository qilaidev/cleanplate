from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

from wm_platform.config import Settings
from wm_platform.errors import AppError


def _is_forbidden_address(address: ipaddress._BaseAddress) -> bool:
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def _resolved_addresses(host: str) -> set[ipaddress._BaseAddress]:
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return set()

    addresses: set[ipaddress._BaseAddress] = set()
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        try:
            addresses.add(ipaddress.ip_address(sockaddr[0]))
        except ValueError:
            continue
    return addresses


def validate_callback_url(raw_url: str, settings: Settings) -> str:
    url = raw_url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise AppError("VALIDATION_ERROR", "callback_url must use http or https", 400)
    if not parsed.netloc or not parsed.hostname:
        raise AppError("VALIDATION_ERROR", "callback_url must include a host", 400)
    if settings.allow_private_callback_urls:
        return url

    host = parsed.hostname.strip().lower()
    if host == "localhost" or host.endswith(".localhost"):
        raise AppError("VALIDATION_ERROR", "callback_url host is not allowed", 400)

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        resolved_addresses = _resolved_addresses(host)
        if any(_is_forbidden_address(item) for item in resolved_addresses):
            raise AppError("VALIDATION_ERROR", "callback_url host is not allowed", 400) from None
        return url

    if _is_forbidden_address(address):
        raise AppError("VALIDATION_ERROR", "callback_url host is not allowed", 400)
    return url
