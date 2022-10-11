from typing import Tuple
from kubemo.adaptor.tensorflow import Inference
from kubemo.algorithm import softmax
from kubemo.serialize import Json, Image, Output
from tensorflow import Tensor, convert_to_tensor

import numpy


# human-readable labels
labels = ("T-Shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot")


# inferencing lifecycle
class FashionMNIST(Inference):

    def preprocess(self, inputs: Tuple[Image, ...]) -> Tuple[Tensor, ...]:
        img = inputs[0].decode()
        img = img.resize((28, 28)).convert('L')
        x = numpy.array(img, dtype=numpy.float32).reshape(1, 28, 28) / 255.
        return convert_to_tensor(x), 


    def postprocess(self, outputs: Tuple[Tensor, ...]) -> Tuple[Output,]:
        k = 3
        y = outputs[0]
        y = softmax(y.numpy())
        topk = y.argsort()[-k:][::-1]
        return Json({labels[i]: y[i] for i in topk}),


# invocation test
if __name__ == '__main__':
    # use two images as a batch of inputs each of which is an image
    images = ['example/ankle-boot.jpg', 'example/t-shirt.jpg']

    # load the saved model 
    model = FashionMNIST('example/fashion_mnist.h5')

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
