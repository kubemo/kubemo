from tensorflow import Tensor
from keras.models import load_model
from moo import Input, Output, Inference as BaseInference


class Inference(BaseInference):

    def __init__(self, path: str) -> None:
        self.model = load_model(path)

    def __del__(self) -> None:
        del self.model

    def forward(self, x: Tensor, **kargs) -> Tensor:
        return self.model(x)

    def preprocess(self, input: Input) -> Tensor:
        raise NotImplementedError

    def postprocess(self, output: Tensor) -> Output:
        raise NotImplementedError
