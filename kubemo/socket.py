from typing import Union
from socket import socket


class Socket:

    def __init__(self, sock: socket) -> None:
        self.sock = sock
        
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def read(self, n: int = -1) -> bytes:
        if n == -1:
            return self.readall()

        b = bytes()
        while n > 0:
            packet = self.sock.recv(n)
            if len(packet) == 0:
                raise ConnectionError('disconnected')
            
            b += packet
            n -= len(packet)
        return b

    def readall(self) -> bytes:
        _size = 4096
        b = bytes()
        while True:
            packet = self.sock.recv(_size)
            if len(packet) < _size:
                break
            b += packet
        return b

    def write(self, b: Union[bytes, bytearray]) -> int:
        self.sock.sendall(b)
        return len(b)

    def close(self):
        self.sock.close()