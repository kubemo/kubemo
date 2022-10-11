from typing import BinaryIO, Tuple, Union, Optional
from socket import AF_INET, AF_UNIX, SOCK_STREAM, socket
from time import time_ns
from io import BytesIO, SEEK_END
from .socket import Socket
from .errors import BatchShapeError, MessageError, ProtocolVersionError, InvocationError
from .serialize import Input, Output
from .protocol import *

import struct


class Connection:

    def __init__(self, network: str, address: Union[Tuple[str, int], str]) -> None:
        '''Creates a connection
        '''
        self.network = network
        self.address = address

        if self.network == 'tcp':
            af, sk = AF_INET, SOCK_STREAM
        elif self.network == 'unix':
            af, sk = AF_UNIX, SOCK_STREAM
        else:
            raise ValueError(f'network {self.network} is not supported yet')

        sock = socket(af, sk)
        sock.connect(self.address)
        self.socket = Socket(sock)


    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.close()

    def close(self):
        self.socket.close()


    def __do(self, kind: int, reader: Optional[BinaryIO]) -> BinaryIO:
        '''Sends a request to server and receives response from it.
        
        '''
        payload_size = 0
        if reader:
            response_payload = reader.read()
            payload_size = len(response_payload)

        request_header = struct.pack(FIXED_HEADER_FMT, PROTOCOL_VERSION, kind, MESSAGE_REQUEST, RESERVED_BYTE, payload_size)
        self.socket.write(request_header + response_payload)

        response_header = self.socket.read(FIXED_HEADER_LEN)
        version, kind, subtype, _, remaining = struct.unpack(FIXED_HEADER_FMT, response_header)

        if version != PROTOCOL_VERSION:
            raise ProtocolVersionError

        if kind == MESSAGE_ERROR:
            self.socket.read(remaining) # discard the payload if there is any
            raise InvocationError(f'responded with error code: {subtype}')

        if subtype != MESSAGE_RESPONSE:
            raise MessageError

        response_payload = self.socket.read(remaining)
        return BytesIO(response_payload)



    def ping(self) -> int:
        start = time_ns()
        self.__do(MESSAGE_PING)
        return time_ns() - start


    def inference(self, batch: Tuple[Tuple[Input, ...], ...]) -> Tuple[Tuple[Output, ...], ...]:
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
        reader = self.__do(MESSAGE_INFERENCE, buffer)
        n_input, n_output, batch_size_res = struct.unpack(INFERENCE_HEADER_FMT, reader.read(4))

        if batch_size_req != batch_size_res:
            raise BatchShapeError(f'batch sent has {batch_size_req} inputs, but the received batch has {batch_size_res} outputs')

        return tuple(tuple(Output(*self.__decode_single_input(reader)) for _ in range(n_output)) for _ in range(batch_size_res))


    def __decode_single_input(self, reader: BinaryIO) -> Tuple[int, BinaryIO]:
        input_type, input_size = struct.unpack('!2L', reader.read(8))
        return input_type, BytesIO(reader.read(input_size))


class Client:

    def __init__(self) -> None:
        pass

    
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    
    def connect(self, network: str, address: Union[Tuple[str, int], str]) -> Connection:
        return Connection(network, address) # todo: pool


