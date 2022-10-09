from typing import Tuple
from numpy import ndarray, concatenate
from moo import Input, Output, Inference as BaseInference
from onnxruntime import InferenceSession


class Inference(BaseInference):

    def __init__(self, path: str, input_names: Tuple[str, ...], output_names: Tuple[str, ...]) -> None:
        super().__init__(path, input_names, output_names)
        self.session = InferenceSession(path)

    def __del__(self):
        del self.session
    
    def forward(self, inputs: Tuple[ndarray, ...]) -> Tuple[ndarray, ...]:
        return self.session.run(
            output_names=self.output_names,
            input_feed={k: v for k, v in zip(self.input_names, inputs)}
        )

    def preprocess(self, inputs: Tuple[Input, ...]) -> Tuple[ndarray, ...]:
        raise NotImplementedError

    def postprocess(self, outputs: Tuple[ndarray, ...]) -> Output:
        raise NotImplementedError

    def concat(self, batch: Tuple[Tuple[ndarray, ...], ...]) -> Tuple[ndarray, ...]:
        return tuple(concatenate(x) for x in zip(*batch))
