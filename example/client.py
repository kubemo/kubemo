from kubemo import Image, Client

network = 'unix'
address = 'test/kubemo.sock'

with Client() as client:
    with client.connect(network, address) as conn:
        batch_input = (
            (Image('example/t-shirt.jpg'), ),
            (Image('example/ankle-boot.jpg'), ),
        )
        batch_output = conn.inference(batch_input)
        for outputs in batch_output:
            print(outputs[0].json())