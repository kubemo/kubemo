from typing import List
from moo.template import Input, Output
from torch import load, Tensor, cat
from moo import Inference as BaseInference



class Inference(BaseInference):

    def __init__(self, path: str) -> None:
        self.model = load(path)

    def __del__(self) -> None:
        del self.model

    def preprocess(self, input: Input) -> Tensor:
        raise NotImplementedError

    def forward(self, input: Tensor, **kargs) -> Tensor:
        return self.model(input)

    def postprocess(self, output: Tensor) -> Output:
        raise NotImplementedError

    def concat(self, batch: List[Tensor]) -> Tensor:
        return cat(batch)