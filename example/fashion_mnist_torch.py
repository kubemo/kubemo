from moo.adaptor.torch import Inference
from moo.template import ImageInput, JsonOutput
from PIL import Image
from torchvision.transforms import Compose, PILToTensor, Grayscale, Resize, ConvertImageDtype
from torch.nn import Linear, ReLU, Flatten, Softmax, Sequential, Module

import torch


# model definition
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = Flatten()
        self.linear_relu_stack = Sequential(
            Linear(28*28, 512),
            ReLU(),
            Linear(512, 512),
            ReLU(),
            Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# pre-process composer
compose = Compose([
    PILToTensor(),
    Grayscale(),
    Resize((28, 28)),
    ConvertImageDtype(torch.float)
])


# human-readable labels
labels = ("T-Shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot")


# inferencing lifecycle
class FashionMNIST(Inference):

    def preprocess(self, input: ImageInput):
        return compose(input.as_image())

    def postprocess(self, y) -> JsonOutput:
        y = Softmax(1)(y)[0]
        values, indices = y.topk(3)
        result = {labels[i]: v.item() for i, v in zip(indices, values)}
        return JsonOutput(result)


# invocation test
if __name__ == '__main__':
    # load the input image
    img = Image.open('example/ankle-boot.jpg')
    # load the saved model 
    f = FashionMNIST('example/fashion_mnist.pt')
    # invoke
    print(f(img))
    