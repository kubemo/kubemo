from abc import abstractmethod
from typing import Tuple, Optional
from tensorflow import Tensor, concat
from keras.models import load_model
from kubemo import Input, Output, Inference as BaseInference


class Inference(BaseInference[Tensor]):

    def __init__(self, path: str, input_names: Optional[Tuple[str, ...]] = None, output_names: Optional[Tuple[str, ...]] = None) -> None:
        super().__init__(input_names, output_names)
        self.model = load_model(path)

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        y = self.model(*inputs)
        return y if isinstance(y, tuple) else y, 

    def concat(self, batch: Tuple[Tuple[Tensor, ...], ...]) -> Tuple[Tensor, ...]:
        return tuple(concat(x, axis=0) for x in zip(*batch))

    @abstractmethod
    def preprocess(self, inputs: Tuple[Input, ...]) -> Tuple[Tensor, ...]: ...

    @abstractmethod
    def postprocess(self, outputs: Tuple[Tensor, ...]) -> Tuple[Output, ...]: ...
