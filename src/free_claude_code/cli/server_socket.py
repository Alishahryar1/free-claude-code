"""Retained listening sockets shared by terminal and desktop supervision."""

import errno
import os
import socket


class ServerSockets:
    """Own the actual listeners, including before Uvicorn adopts them."""

    def __init__(self, sockets: list[socket.socket]) -> None:
        self.sockets = sockets

    @classmethod
    def reserve(cls, host: str, port: int) -> ServerSockets:
        sockets: list[socket.socket] = []
        owner = cls(sockets)
        try:
            addresses = dict.fromkeys(
                socket.getaddrinfo(
                    host or None,
                    port,
                    socket.AF_UNSPEC,
                    socket.SOCK_STREAM,
                    0,
                    socket.AI_PASSIVE,
                )
            )
            for family, kind, protocol, _name, address in addresses:
                try:
                    listener = socket.socket(family, kind, protocol)
                except OSError as exc:
                    if exc.errno in {errno.EAFNOSUPPORT, errno.EPROTONOSUPPORT}:
                        continue
                    raise
                sockets.append(listener)
                if os.name == "nt":
                    listener.setsockopt(
                        socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1
                    )
                elif os.name == "posix":
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if family == socket.AF_INET6:
                    listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                try:
                    listener.bind(address)
                except OSError as exc:
                    if exc.errno == errno.EADDRNOTAVAIL:
                        sockets.remove(listener)
                        listener.close()
                        continue
                    raise
                listener.listen(2048)
                listener.setblocking(False)
            if not sockets:
                raise OSError(f"No usable listening address for {host}:{port}")
            return owner
        except BaseException:
            owner.close()
            raise

    def close(self) -> None:
        for listener in self.sockets:
            listener.close()
        self.sockets.clear()

    def __enter__(self) -> ServerSockets:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
