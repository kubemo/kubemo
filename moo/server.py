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


from typing import Tuple, Dict, Union, BinaryIO
from socket import AF_INET, AF_UNIX, SOCK_STREAM, socket
from io import BytesIO
from .inference import Inference
from .template import Input, Output
from .errors import BatchShapeError
from .sock import Sock
from .protocol import *

import struct
import logging
import os


class Handler:

    def handle(self, conn: Sock) -> bytes:
        raise NotImplementedError

    def kind(self) -> int:
        raise NotImplementedError


class PingHandler(Handler):

    def handle(self, conn: Sock) -> bytes:
        return b''

    def kind(self) -> int:
        return MESSAGE_PING


class InferenceHandler(Handler):

    def __init__(self, model: Inference) -> None:
        self.model = model

    def handle(self, conn: Sock) -> bytes:
        request_header = conn.read(4)
        n_input, _, batch_size = struct.unpack(INFERENCE_HEADER_FMT, request_header)
        batch_input = tuple(tuple(Input(*self.__decode_single_input(conn)) for _ in range(n_input)) for _ in range(batch_size))
        batch_output = self.model(*batch_input)

        n_output = len(batch_input[0])
        for outputs in batch_output[1:]:
            if n_output != len(outputs):
                raise BatchShapeError
        
        batch_size = len(batch_output)
        response_header = struct.pack(INFERENCE_HEADER_FMT, n_input, n_output, batch_size)
        return response_header + bytes(bytes(self.__encode_single_output(output) for output in outputs) for outputs in batch_output)


    def __decode_single_input(self, conn: Sock) -> Tuple[int, BinaryIO]:
        input_type, input_size = struct.unpack('!2L', conn.read(8))
        return input_type, BytesIO(conn.read(input_size))
                
    def __encode_single_output(self, output: Output) -> bytes:
        output_type = output.kind
        output_data = output.encode()
        output_tl = struct.pack('!2L', output_type, len(output_data))
        return output_tl + output_data

    def kind(self) -> int:
        return MESSAGE_INFERENCE


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
            self.handle(PingHandler())


    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self) -> None:
        self.socket.close()
        if self.network == 'unix':
            os.remove(self.address)


    def handle(self, handler: Handler) -> None:
        self.handlers[handler.kind()] = handler
    

    def serve(self) -> None:
        while True:
            conn, addr = self.socket.accept()
            conn.setblocking(True)

            with conn:
                logging.info(f'accepted connection from: {addr}')
                self.dispatch(Sock(conn))


    def dispatch(self, conn: Sock) -> None:
        try:
            fixed_header = conn.read(FIXED_HEADER_LEN)
            version, kind, subtype, _, _ = struct.unpack(FIXED_HEADER_FMT, fixed_header)

            if version != PROTOCOL_VERSION:
                return self.__respond_with_error(conn, ERROR_PROTOCOL)

            if subtype != MESSAGE_REQUEST:
                return self.__respond_with_error(conn, ERROR_SUBTYPE)

            if kind not in self.handlers:
                return self.__respond_with_error(conn, ERROR_METHOD)

            response = self.handlers[kind].handle(conn)
            return self.__respond(conn, kind, response)
            
        except ConnectionError:
            logging.warn('client disconnected')
            return
        except MemoryError:
            return self.__respond_with_error(conn, ERROR_MEMORY)
        except BatchShapeError:
            return self.__respond_with_error(conn, ERROR_SHAPE)
        except Exception as e:
            logging.error('error handing request: %s', e)
            return self.__respond_with_error(conn, ERROR_INTERNAL)

    
    def __respond_with_error(self, conn: Sock, status: int) -> None:
        fixed_header = struct.pack(FIXED_HEADER_FMT, PROTOCOL_VERSION, MESSAGE_ERROR, status, 0, RESERVED_BYTE)
        return conn.write(fixed_header)


    def __respond(self, conn: Sock, kind: int, b: bytes) -> None:
        fixed_header = struct.pack(FIXED_HEADER_FMT, PROTOCOL_VERSION, kind, MESSAGE_RESPONSE, len(b), RESERVED_BYTE)
        return conn.write(fixed_header + b)


    

