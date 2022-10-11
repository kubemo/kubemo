from kubemo import Image, Client

endpoint = 'unix://test/kubemo.sock'

with Client() as client:
    # ping
    rrt = client.ping(endpoint)
    print(f'RRT: {rrt}ns')

    # inference
    batch_input = (
        (Image('example/t-shirt.jpg'), ),
        (Image('example/ankle-boot.jpg'), ),
    )
    batch_output = client.inference(endpoint, batch_input)
    for outputs in batch_output:
        print(outputs[0].decode())