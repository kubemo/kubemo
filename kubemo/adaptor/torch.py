from abc import abstractmethod
from typing import Tuple, Optional
from torch import load, Tensor, cat, cuda, device
from kubemo import Inference as BaseInference
from kubemo.serialize import Input, Output


class Inference(BaseInference[Tensor]):

    def __init__(self, 
                path: str,
                device_id: Optional[int] = None,
                input_names: Optional[Tuple[str, ...]] = None,
                output_names: Optional[Tuple[str, ...]] = None
    ) -> None:
        super().__init__(device_id, input_names, output_names)
        # self.device = device(device_id) if device_id and cuda.is_available() else 'cpu'
        self.model = load(path)

    def __del__(self) -> None:
        del self.model

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        y = self.model(*inputs)
        return y if isinstance(y, tuple) else y,

    def concat(self, batch: Tuple[Tuple[Tensor, ...], ...]) -> Tuple[Tensor, ...]:
        return tuple(cat(x) for x in zip(*batch))

    @abstractmethod
    def preprocess(self, inputs: Tuple[Input, ...]) -> Tuple[Tensor, ...]: ...

    @abstractmethod
    def postprocess(self, outputs: Tuple[Tensor, ...]) -> Tuple[Output, ...]: ...

    