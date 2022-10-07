from moo.adaptor.tensorflow import Inference
from moo.algorithm import softmax
from moo.template import Input, JsonOutput
import numpy


# human-readable labels
labels = ("T-Shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot")


# inferencing lifecycle
class FashionMNIST(Inference):

    def preprocess(self, input: Input) -> numpy.ndarray:
        img = input.as_image()
        img = img.resize((28, 28)).convert('L')
        return numpy.array(img, dtype=numpy.float32).reshape(1, 28, 28) / 255.


    def postprocess(self, y) -> JsonOutput:
        k = 3
        y =  softmax(y.numpy().flatten())
        topk = y.argsort()[-k:][::-1]
        return JsonOutput({labels[i]: y[i] for i in topk})


# invocation test
if __name__ == '__main__':
    images = ['example/ankle-boot.jpg', 'example/t-shirt.jpg']

    # load the saved model 
    model = FashionMNIST('example/fashion_mnist.h5')
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
    
