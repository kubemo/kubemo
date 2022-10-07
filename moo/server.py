from concurrent.futures import ThreadPoolExecutor
from .proto.inference_pb2_grpc import InferenceServicer, add_InferenceServicer_to_server
from .proto.inference_pb2 import ScaleRequest, ScaleResponse, PredictRequest,  PredictResponse
from grpc import ServicerContext, StatusCode
import grpc

from .inference import Inference
from .template import Input, ImageInput, TextInput


class Server:
    '''An interface abstracting a server to serve an Inference object.
    '''

    def __init__(self, inference: Inference) -> None:
        self.inference = inference


    def run(self, address: str) -> None:
        '''Starts the server itself.

        Raises:
        NotImplementedError: An error occurred when this method is not implemented
            by subclasses.
        '''
        server = grpc.server(ThreadPoolExecutor(max_workers=10))
        add_InferenceServicer_to_server(Servicer(self.inference), server)
        server.add_insecure_port(address)
        server.start()
        server.wait_for_termination()



class Servicer(InferenceServicer):

    def __init__(self, inference: Inference) -> None:
        self.inference = inference

    def Scale(self, req: ScaleRequest, ctx: ServicerContext) -> ScaleResponse:
        return ScaleResponse()

    def Predict(self, req: PredictRequest, ctx: ServicerContext) -> PredictResponse:
        try:
            input = from_request(req)
            output = self.inference(input)
            return PredictResponse(type=output.type, body=output.encode())
        except Exception as e:
            ctx.abort(StatusCode.INVALID_ARGUMENT, str(e))




def from_request(req: PredictRequest) -> Input:
    if req.type == 'image':
        return ImageInput(req.body)
    elif req.type == 'text':
        return TextInput(req.body)
    else:
        raise KeyError(f'unspported content type: {req.type}')