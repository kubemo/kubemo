from concurrent.futures import ThreadPoolExecutor
from .proto.inference_pb2_grpc import InferenceServicer, add_InferenceServicer_to_server
from .proto.inference_pb2 import PredictRequest,  PredictResponse, Output
from grpc import ServicerContext, StatusCode
import grpc

from .inference import Inference
from .template import Input


class Server:
    '''An interface abstracting a server to serve an Inference object.
    '''

    def __init__(self, model: Inference) -> None:
        self.model = model

    def run(self, address: str) -> None:
        '''Runs the server itself.
        '''
        server = grpc.server(ThreadPoolExecutor(max_workers=10))
        add_InferenceServicer_to_server(Servicer(self.model), server)
        server.add_insecure_port(address)
        server.start()
        server.wait_for_termination()



class Servicer(InferenceServicer):

    def __init__(self, model: Inference) -> None:
        self.model = model

    def Predict(self, req: PredictRequest, ctx: ServicerContext) -> PredictResponse:
        try:
            batch_input = [Input(i.body) for i in req.batch]
            batch_output = self.model(batch_input)
            return PredictResponse(batch=[Output(type=o.type, body=o.encode()) for o in batch_output])

        except Exception as e:
            ctx.abort(StatusCode.INTERNAL, str(e))

