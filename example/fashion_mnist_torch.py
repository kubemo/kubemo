from typing import Tuple
from torchvision.transforms import Compose, PILToTensor, Grayscale, Resize, ConvertImageDtype
from torch.nn import Linear, ReLU, Flatten, Sequential, Module
from torch.nn.functional import softmax
from torch import Tensor, float
from kubemo.adaptor.torch import Inference
from kubemo.serialize import Json, Image, Output


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
    ConvertImageDtype(float)
])


# human-readable labels
labels = ("T-Shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot")


# inferencing lifecycle
class FashionMNIST(Inference):

    def preprocess(self, inputs: Tuple[Image, ...]) -> Tuple[Tensor, ...]:
        return compose(inputs[0].decode()), 

    def postprocess(self, outputs: Tuple[Tensor, ...]) -> Tuple[Output,]:
        y = softmax(outputs[0], 0)
        values, indices = y.topk(3)
        result = {labels[i]: v.item() for i, v in zip(indices, values)}
        return Json(result),


# invocation test
if __name__ == '__main__':
    # use two images as a batch of inputs each of which is an image
    images = ['example/ankle-boot.jpg', 'example/t-shirt.jpg']

    # load the saved model 
    model = FashionMNIST(
        path='example/fashion_mnist.pt',
        input_names=('x', ),
        output_names=None,
    )

    # create a batch of inputs using the two images just selected
    batch_input = []
    for i in images:
        inputs = (Image(i), ) # single input
        batch_input.append(inputs)

    # call the model with the batch of inputs just created
    batch_output = model(*batch_input)

    # print the batch of outputs respectively
    for k, y in zip(images, batch_output):
        print(f'{k} => {y[0]}')
        
        
    