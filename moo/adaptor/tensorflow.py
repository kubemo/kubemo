from typing import Dict, Union, Any
from tensorflow import Tensor
from keras.models import load_model
from PIL.Image import Image
from moo import BaseInference


class Inference(BaseInference):

    def __init__(self, path: str) -> None:
        self.model = load_model(path)

    def __del__(self) -> None:
        del self.model

    def forward(self, x: Tensor, **kargs) -> Tensor:
        return self.model(x)

    def preprocess(self, input: Union[Image, str, bytearray]) -> Tensor:
        raise NotImplementedError

    def postprocess(self, output: Tensor) -> Union[str, Dict[str, Any]]:
        raise NotImplementedError
