from typing import Tuple
from moo.adaptor.tensorflow import Inference
from moo.algorithm import softmax
from moo.template import Input, JsonOutput
from tensorflow import Tensor, convert_to_tensor

import numpy


# human-readable labels
labels = ("T-Shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot")


# inferencing lifecycle
class FashionMNIST(Inference):

    def preprocess(self, inputs: Tuple[Input, ...]) -> Tuple[Tensor, ...]:
        img = inputs[0].as_image()
        img = img.resize((28, 28)).convert('L')
        x = numpy.array(img, dtype=numpy.float32).reshape(1, 28, 28) / 255.
        return convert_to_tensor(x), 


    def postprocess(self, outputs: Tuple[Tensor, ...]) -> JsonOutput:
        k = 3
        y = outputs[0]
        y = softmax(y.numpy())
        topk = y.argsort()[-k:][::-1]
        return JsonOutput({labels[i]: y[i] for i in topk})


# invocation test
if __name__ == '__main__':
    images = ['example/ankle-boot.jpg', 'example/t-shirt.jpg']

    # load the saved model 
    model = FashionMNIST(
        path='example/fashion_mnist.h5',
        input_names=('x', ),
        output_names=None,
    )

    # load inputs
    batch_input = []
    for p in images:
        with open(p, 'rb') as f:
            inputs = (Input(f.read()), ) # single input
            batch_input.append(inputs)

    # call the model with a batch of inputs
    batch_output = model(*batch_input)

    # print outputs
    for k, y in zip(images, batch_output):
        print(f'{k} => {y}')
    
