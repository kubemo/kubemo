from moo.adaptor.torch import Inference
from moo.template import Input, JsonOutput
from torchvision.transforms import Compose, PILToTensor, Grayscale, Resize, ConvertImageDtype
from torch.nn import Linear, ReLU, Flatten, Sequential, Module
from torch.nn.functional import softmax

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

    def preprocess(self, input: Input):
        return compose(input.as_image())

    def postprocess(self, y) -> JsonOutput:
        y = softmax(y, 0)
        values, indices = y.topk(3)
        result = {labels[i]: v.item() for i, v in zip(indices, values)}
        return JsonOutput(result)


# invocation test
if __name__ == '__main__':
    images = ['example/ankle-boot.jpg', 'example/t-shirt.jpg']

    # load the saved model 
    model = FashionMNIST('example/fashion_mnist.pt')
    batch_input = []

    # load inputs
    for p in images:
        with open(p, 'rb') as f:
            batch_input.append(Input(f.read()))

    # call the model
    batch_output = model(batch_input)

    # print outputs
    for y in batch_output:
        print(y.encode())
        
        
    