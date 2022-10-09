from typing import Tuple
from moo.template import Input, Output
from torch import load, Tensor, cat
from moo import Inference as BaseInference



class Inference(BaseInference):

    def __init__(self, path: str, input_names: Tuple[str, ...], output_names: Tuple[str, ...]) -> None:
        super().__init__(path, input_names, output_names)
        self.model = load(path)

    def __del__(self) -> None:
        del self.model

    def preprocess(self, inputs: Tuple[Input, ...]) -> Tuple[Tensor, ...]:
        raise NotImplementedError

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        y = self.model(*inputs)
        return y if isinstance(y, tuple) else y,

    def postprocess(self, outputs: Tuple[Tensor, ...]) -> Output:
        raise NotImplementedError

    def concat(self, batch: Tuple[Tuple[Tensor, ...], ...]) -> Tuple[Tensor, ...]:
        return tuple(cat(x) for x in zip(*batch))