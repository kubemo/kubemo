from typing import Tuple
from kubemo.adaptor.onnx import Inference
from kubemo.algorithm import softmax
from kubemo.serialize import Input, Json, Image

import numpy


# human-readable labels
labels = ("T-Shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot")


# inferencing lifecycle
class FashionMNIST(Inference):

    def preprocess(self, inputs: Tuple[Input, ...]) -> Tuple[numpy.ndarray, ...]:
        img = inputs[0].image()
        img = img.resize((28, 28)).convert('L')
        x = numpy.array(img, dtype=numpy.float32).reshape(1, 28, 28) / 255.
        return x, 


    def postprocess(self, outputs: Tuple[numpy.ndarray, ...]) -> Tuple[Json,]:
        k = 3
        y = softmax(outputs[0])
        topk = y.argsort()[-k:][::-1]
        result = {labels[i]: y[i] for i in topk}
        return Json(result), 
        

# invocation test
if __name__ == '__main__':

    images = ['example/ankle-boot.jpg', 'example/t-shirt.jpg']

    # load the saved model 
    model = FashionMNIST(
        path='example/fashion_mnist.onnx',
        input_names=('x', ),
        output_names=None,
    )

    # load inputs
    batch_input = []
    for i in images:
        inputs = (Image(i), ) # single input
        batch_input.append(inputs)

    # call the model with a batch of inputs
    batch_output = model(*batch_input)

    # print outputs
    for k, y in zip(images, batch_output):
        print(f'{k} => {y[0]}')