from abc import abstractmethod
from typing import Tuple, Optional
from numpy import ndarray, concatenate
from kubemo import Input, Output, Inference as BaseInference
from onnxruntime import InferenceSession


class Inference(BaseInference[ndarray]):

    def __init__(self, 
        path: str, 
        device_id: Optional[int] = None,
        input_names: Optional[Tuple[str, ...]] = None, 
        output_names: Optional[Tuple[str, ...]] = None,
    ) -> None:

        super().__init__(device_id,input_names, output_names)
        self.session = InferenceSession(path)

    def __del__(self) -> None:
        del self.session

    def forward(self, inputs: Tuple[ndarray, ...]) -> Tuple[ndarray, ...]:
        return self.session.run(
            output_names=self.output_names,
            input_feed={k: v for k, v in zip(self.input_names, inputs)}
        )

    def concat(self, batch: Tuple[Tuple[ndarray, ...], ...]) -> Tuple[ndarray, ...]:
        return tuple(concatenate(x) for x in zip(*batch))

    @abstractmethod
    def preprocess(self, inputs: Tuple[Input, ...]) -> Tuple[ndarray, ...]: ...

    @abstractmethod
    def postprocess(self, outputs: Tuple[ndarray, ...]) -> Tuple[Output, ...]: ...

    
