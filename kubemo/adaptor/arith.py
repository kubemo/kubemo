from abc import abstractmethod
from typing import Tuple
from kubemo import Inference as BaseInference
from numpy import ndarray, concatenate
from kubemo.serialize import Input, Output

class Inference(BaseInference[ndarray]):

    def concat(self, batch: Tuple[Tuple[ndarray, ...], ...]) -> Tuple[ndarray, ...]:
        return tuple(concatenate(x) for x in zip(*batch))

    @abstractmethod
    def preprocess(self, inputs: Tuple[Input, ...]) -> Tuple[ndarray, ...]: ...

    @abstractmethod
    def forward(self, inputs: Tuple[ndarray, ...]) -> Tuple[ndarray, ...]: ...

    @abstractmethod
    def postprocess(self, outputs: Tuple[ndarray, ...]) -> Tuple[Output, ...]: ...

    