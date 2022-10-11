from kubemo.client import Client
from kubemo import Input, IMAGE

network = 'unix'
address = 'test/kubemo.sock'

images = []

with Client() as client:
    with client.connect(network, address) as conn:
        batch_input = (
            (Input(IMAGE, open('example/t-shirt.jpg', 'rb')), ),
            (Input(IMAGE, open('example/ankle-boot.jpg', 'rb')), ),
        )
        batch_output = conn.inference(batch_input)
        for outputs in batch_output:
            print(outputs[0].json())