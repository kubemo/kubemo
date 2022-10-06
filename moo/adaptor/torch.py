from typing import Dict, Union, Any
from torch import load, Tensor
from PIL.Image import Image

from moo import BaseInference



class TorchInference(BaseInference):

    def __init__(self) -> None:
        self.model = None

    def load(self, path: str) -> None:
        self.model = load(path)

    def unload(self) -> None:
        del self.model

    def preprocess(self, input: Union[Image, str, bytearray]) -> Tensor:
        raise NotImplementedError

    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)

    def postprocess(self, output: Tensor) -> Union[str, Dict[str, Any]]:
        raise NotImplementedError
