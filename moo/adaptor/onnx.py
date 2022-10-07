from numpy import ndarray
from moo import Input, Output, Inference as BaseInference
from onnxruntime import InferenceSession


class Inference(BaseInference):

    def __init__(self, path: str) -> None:
        self.session = InferenceSession(path)

    def __del__(self):
        del self.session
    
    def forward(self, x: ndarray, **kargs) -> ndarray:
        input_name = kargs.get('onnx_input_name') or 'x'
        return self.session.run(None, {input_name: x})

    def preprocess(self, input: Input) -> ndarray:
        raise NotImplementedError

    def postprocess(self, output: ndarray) -> Output:
        raise NotImplementedError
    