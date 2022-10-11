from kubemo import Server, InferenceHandler
from fashion_mnist_onnx import FashionMNIST

import logging

if __name__ == '__main__':
    # init logger using base config
    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(levelname)-5s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.DEBUG
    )

    # load the saved model 
    model = FashionMNIST(
        path='example/fashion_mnist.onnx',
        input_names=('x', ),
    )

    network = 'unix'
    address = 'test/kubemo.sock'

    # create a server and register an inference handler
    with Server(network, address) as server:

        server.handle(InferenceHandler(model))

        try:
            logging.info(f'server listening at {network}://{address}')
            server.serve()
        except KeyboardInterrupt:
            logging.info('server stopped')

    

