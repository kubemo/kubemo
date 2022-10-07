from typing import Dict, Union, Any
from torch import load, Tensor
from PIL.Image import Image
from moo import BaseInference



class Inference(BaseInference):

    def __init__(self, path: str) -> None:
        self.model = load(path)

    def __del__(self) -> None:
        del self.model

    def preprocess(self, input: Union[Image, str, bytearray]) -> Tensor:
        raise NotImplementedError

    def forward(self, input: Tensor, **kargs) -> Tensor:
        return self.model(input)

    def postprocess(self, output: Tensor) -> Union[str, Dict[str, Any]]:
        raise NotImplementedError
