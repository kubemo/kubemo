from fashion_mnist_onnx import FashionMNIST
from moo.server import Server

if __name__ == '__main__':
    address = '[::]:50051'
    inference = FashionMNIST('example/fashion_mnist.onnx')
    server = Server(inference, address)
    server.run()