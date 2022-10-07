from fashion_mnist_onnx import FashionMNIST
from moo.server import Server

if __name__ == '__main__':
    address = '[::]:50051'
    model = FashionMNIST('example/fashion_mnist.onnx')
    server = Server(model)
    server.run(address)