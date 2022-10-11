from typing import BinaryIO, Tuple, Union, Optional
from urllib.parse import urlparse
from socket import AF_INET, AF_UNIX, SOCK_STREAM, socket
from time import time_ns
from io import BytesIO
from .socket import Socket
from .errors import BatchShapeError, MessageError, ProtocolVersionError, InvocationError, ParseAddressError
from .serialize import Dynamic, Input, Output
from .protocol import *

import struct


class Client:

    def __init__(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        '''Closes all connections in the pool
        '''
        pass

    
    def __connect(self, target: str) -> Socket:
        # todo: pool
        network, address = _parse_url(target)

        if network == 'tcp':
            af, sk = AF_INET, SOCK_STREAM
        elif network == 'unix':
            af, sk = AF_UNIX, SOCK_STREAM
        else:
            raise ValueError(f'network {network} is not supported yet')

        sock = socket(af, sk)
        sock.connect(address)
        return Socket(sock)


    def __do(self, conn: Socket, kind: int, reader: Optional[BinaryIO] = None) -> BinaryIO:
        '''Sends a request to a server and receives response from it.
        '''
        request_payload = None
        request_payload_size = 0

        if reader:
            request_payload = reader.read()
            request_payload_size = len(request_payload)

        with conn:
            request_header = struct.pack(FIXED_HEADER_FMT, PROTOCOL_VERSION, kind, MESSAGE_REQUEST, RESERVED_BYTE, request_payload_size)
            conn.write(request_header + request_payload if request_payload else request_header)

            response_header = conn.read(FIXED_HEADER_LEN)
            version, kind, subtype, _, remaining = struct.unpack(FIXED_HEADER_FMT, response_header)

            if version != PROTOCOL_VERSION:
                raise ProtocolVersionError

            if kind == MESSAGE_ERROR:
                conn.read(remaining) # discard the payload if there is any
                raise InvocationError(f'responded with error code: {subtype}')

            if subtype != MESSAGE_RESPONSE:
                raise MessageError

            response_payload = conn.read(remaining)
            return BytesIO(response_payload)
    

    def inference(self, target: str, batch: Tuple[Tuple[Input, ...], ...]) -> Tuple[Tuple[Output, ...], ...]:
        conn = self.__connect(target)

        batch_size_req = len(batch)
        n_input = len(batch[0])
        for inputs in batch[1:]:
            if n_input != len(inputs):
                raise BatchShapeError('inconsistent number of inputs')


        header = struct.pack(INFERENCE_HEADER_FMT, n_input, 0, batch_size_req)
        buffer = BytesIO(header)
        buffer.seek(len(header))

        for inputs in batch:
            for input in inputs:
                input_type = input.kind
                input_data = input.reader.read()
                input_tl = struct.pack('!2L', input_type, len(input_data))
                buffer.write(input_tl + input_data)

        buffer.seek(0)
        reader = self.__do(conn, MESSAGE_INFERENCE, buffer)
        n_input, n_output, batch_size_res = struct.unpack(INFERENCE_HEADER_FMT, reader.read(4))

        if batch_size_req != batch_size_res:
            raise BatchShapeError(f'batch sent has {batch_size_req} inputs, but the received batch has {batch_size_res} outputs')

        return tuple(tuple(Dynamic(*decode_single_input(reader)) for _ in range(n_output)) for _ in range(batch_size_res))


    def ping(self, target: str) -> int:
        conn = self.__connect(target)
        start = time_ns()
        self.__do(conn, MESSAGE_PING)
        return time_ns() - start



def _parse_url(s: str) -> Tuple[str, Union[Tuple[str, int],str]]:
    u = urlparse(s)
    if u.scheme == 'unix':
        addr = u.path
        if u.hostname:
            addr = u.hostname + addr
        return u.scheme, addr
    elif u.scheme == 'tcp':
        return u.scheme, (u.hostname, u.port)
    
    raise ParseAddressError