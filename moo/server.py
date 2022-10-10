# from concurrent.futures import ThreadPoolExecutor
# from .proto.inference_pb2_grpc import InferenceServicer, add_InferenceServicer_to_server
# from .proto.inference_pb2 import PredictRequest,  PredictResponse, Output
# from grpc import ServicerContext, StatusCode
# import grpc

# from .inference import Inference
# from .template import Input


# # todo: remove
# class Server:
#     '''An interface abstracting a server to serve an Inference object.
#     '''

#     def __init__(self, model: Inference) -> None:
#         self.model = model

#     def run(self, address: str) -> None:
#         '''Runs the server itself.
#         '''
#         server = grpc.server(ThreadPoolExecutor(max_workers=10))
#         add_InferenceServicer_to_server(Servicer(self.model), server)
#         server.add_insecure_port(address)
#         server.start()
#         server.wait_for_termination()



# class Servicer(InferenceServicer):

#     def __init__(self, model: Inference) -> None:
#         self.model = model

#     def Predict(self, req: PredictRequest, ctx: ServicerContext) -> PredictResponse:
#         try:
#             batch_input = [Input(i.body) for i in req.batch]
#             batch_output = self.model(batch_input)
#             return PredictResponse(batch=[Output(type=o.type, body=o.encode()) for o in batch_output])

#         except Exception as e:
#             ctx.abort(StatusCode.INTERNAL, str(e))


'''
        Model Invocation Protocol

0         7         15        23         31
+---------+----------+---------+----------+       ---
| version |   kind   | subtype | reserved |        ^      
+---------+---------+---------+-----------+   fixed header 
|             remaining size              |        v
+---------+----------+--------------------+       ---
| n-input | n-output |     batch size     |        ^
+---------+----------+--------------------+        |
|                  ...                    |        |
+-----------------------------------------+        |
|            input/output size            |  variable payload (on *kind* == 1)
+-----------------------------------------+        |
|            input/output data            |        |
+-----------------------------------------+        |
|                  ...                    |        v
+-----------------------------------------+       ---

MIP, aka model invocation protocol, is used for ML model invocation. The first
8 octets form the fixed-length header which includes:
**version**        specifies MIP version to use.
**kind**           specifies message type (for multi-endpoint invocation)
**subtype**        specifies request, response or a concrete error type.
**reserved**       reserved for later usage.
**remaining size** specifies how many bytes of payload to read.

When *kind* equals to 1, which indicates an Inferencing call, the following 4
octets are required to include:
**n-input**    specifies the number of the model's inputs.
**n-output**   specifies the number of the model's outputs.
**batch size** specifies the batch size of the following input/output.

The rest of the message uses a *size-value* format to curry a batch of input
or output.
'''

from typing import Callable, Tuple, Dict, Union
from socket import AF_INET, AF_UNIX, SOCK_STREAM, socket
from .inference import Inference
from .template import Input, Output
from .protocol import *

import struct
import logging
import os


Handler = Callable[[socket], bytes]


class PingHandler(Handler):

    def __call__(self, conn: socket) -> bytes:
        return bytes()


class InferenceHandler(Handler):

    def __init__(self, model: Inference) -> None:
        self.model = model

    def __call__(self, conn: socket) -> bytes:
        header = conn.recv(4)
        n_input, n_output, batch_size = struct.unpack('!2BH', header)
        batch_input = tuple(tuple(Input(self.__decode_single_input(conn)) for _ in range(n_input)) for _ in range(batch_size))
        batch_output = self.model(*batch_input)
        return header + bytes(self.__encode_single_output(output) for output in batch_output)


    def __decode_single_input(self, conn: socket) -> bytes:
        input_size, = struct.unpack('!L', conn.recv(4))
        return conn.recv(input_size)
                
    def __encode_single_output(self, output: Output) -> bytes:
        output_data = output.encode()
        output_size = struct.pack('!L', len(output_data))
        return output_size + output_data



class Server:

    def __init__(self, 
                network: str, 
                address: Union[Tuple[str, int], str],
                register_ping_handler: bool = True
        ) -> None:
        '''Creates a server with given options
        '''
        self.network = network
        self.address = address

        if self.network == 'tcp':
            af, sk = AF_INET, SOCK_STREAM
        elif self.network == 'unix':
            af, sk = AF_UNIX, SOCK_STREAM
        else:
            raise ValueError(f'network {self.network} is not supported yet')

        self.socket = socket(af, sk)
        self.socket.bind(self.address)
        self.socket.listen(1)

        self.handlers: Dict[int, Handler] = {}
        if register_ping_handler:
            self.handle(MESSAGE_PING, PingHandler())


    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self) -> None:
        self.socket.close()
        if self.network == 'unix':
            os.remove(self.address)


    def handle(self, kind: int, handler: Handler) -> None:
        self.handlers[kind] = handler
    

    def serve(self) -> None:
        while True:
            conn, addr = self.socket.accept()
            conn.setblocking(True)

            with conn:
                logging.info(f'accepted connection from: {addr}')
                self.dispatch(conn)


    def dispatch(self, conn: socket) -> None:
        fixed_header = conn.recv(FIXED_HEADER_LEN)
        version, kind, subtype, reserved, remaining = struct.unpack(FIXED_HEADER_FMT, fixed_header)

        if version != PROTOCOL_VERSION:
            return self.__respond_with_error(conn, ERROR_PROTOCOL)

        if subtype != MESSAGE_REQUEST:
            return self.__respond_with_error(conn, ERROR_SUBTYPE)

        if kind not in self.handlers:
            return self.__respond_with_error(conn, ERROR_METHOD)

        try:
            response = self.handlers[kind](conn)
            return self.__respond(conn, kind, response)
        except MemoryError:
            return self.__respond_with_error(conn, ERROR_MEMORY)
        except Exception as e:
            logging.error('error handing request: %s', e)
            return self.__respond_with_error(conn, ERROR_INTERNAL)

    
    def __respond_with_error(self, conn: socket, status: int) -> None:
        fixed_header = struct.pack(FIXED_HEADER_FMT, PROTOCOL_VERSION, MESSAGE_ERROR, status, 0, RESERVED_BYTE)
        return conn.sendall(fixed_header)


    def __respond(self, conn: socket, kind: int, b: bytes) -> None:
        fixed_header = struct.pack(FIXED_HEADER_FMT, PROTOCOL_VERSION, kind, MESSAGE_RESPONSE, len(b), RESERVED_BYTE)
        return conn.sendall(fixed_header + b)


    