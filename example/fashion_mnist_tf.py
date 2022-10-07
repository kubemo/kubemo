from moo.adaptor.tensorflow import Inference
from moo.algorithm import softmax
from moo.template import ImageInput, DictOutput
from PIL import Image
import numpy


# human-readable labels
labels = ("T-Shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot")


# inferencing lifecycle
class FashionMNIST(Inference):

    def preprocess(self, input: ImageInput) -> numpy.ndarray:
        img = input.as_image()
        img = img.resize((28, 28)).convert('L')
        return numpy.array(img, dtype=numpy.float32).reshape(1, 28, 28) / 255.


    def postprocess(self, y) -> DictOutput:
        k = 3
        y =  softmax(y.numpy().flatten())
        topk = y.argsort()[-k:][::-1]
        return DictOutput({labels[i]: y[i] for i in topk})


# invocation test
if __name__ == '__main__':
    # load the input image
    img = Image.open('example/ankle-boot.jpg')
    # load the saved model 
    f = FashionMNIST('example/fashion_mnist.h5')
    # invoke
    print(f(img))