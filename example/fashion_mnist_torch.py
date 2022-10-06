from moo.adaptor import TorchInference
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
class FashionMNIST(TorchInference):

    def preprocess(self, input):
        if not isinstance(input, Image.Image):
            raise TypeError('input is not a PIL.Image')
        return compose(input)

    def forward(self, x):
        return self.model(x)

    def postprocess(self, y):
        y = Softmax(1)(y)[0]
        values, indices = y.topk(3)
        return {labels[i]: v.item() for i, v in zip(indices, values)}


# invocation test
if __name__ == '__main__':
    # load the input image
    img = Image.open('example/ankle-boot.jpg')
    # instaniate 
    f = FashionMNIST()
    # load
    f.load('example/fashion_mnist.pt')
    # invoke
    print(f(img))
    