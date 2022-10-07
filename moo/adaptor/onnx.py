from typing import Dict, Union, Any
from onnxruntime import InferenceSession
from numpy import ndarray
from PIL.Image import Image
from moo import BaseInference


class Inference(BaseInference):

    def __init__(self, path: str) -> None:
        self.session = InferenceSession(path)

    def __del__(self):
        del self.session
    
    def forward(self, x: ndarray, **kargs) -> ndarray:
        input_name = kargs.get('onnx_input_name') or 'x'
        return self.session.run(None, {input_name: x})

    def preprocess(self, input: Union[Image, str, bytearray]) -> ndarray:
        raise NotImplementedError

    def postprocess(self, output: ndarray) -> Union[str, Dict[str, Any]]:
        raise NotImplementedError
    