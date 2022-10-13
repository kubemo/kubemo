## Protocol

There are some overheads in the existing communication protocols like HTTP or gRPC if used by a client to invoke a machine-learning model from a server since a model can just process a batch of inputs at a time. Concurrent invocation to a model is made possible only by deploying more than just one instance of the model. So we developed a protocol named MIP which is not only lightweight compared with other general ones, but also efficient enough to curry a batch of inputs/outputs.

A MIP packet consists of an 8-byte long fixed header and a variable payload. Here is what it looks like.

```
        Model Invocation Protocol

0         7         15        23         31
+---------+----------+---------+----------+       ---
| version |   kind   | subtype | reserved |        ^      
+---------+---------+----------+----------+   fixed header 
|             remaining size              |        v
+---------+----------+--------------------+       ---
| n-input | n-output |     batch size     |        ^
+---------+----------+--------------------+        |
|                   ...                   |        |
+-----------------------------------------+        |
|            input/output type            |        |
+-----------------------------------------+  variable payload (on *kind* set to 2)
|            input/output size            |        |
+-----------------------------------------+        |
|            input/output data            |        |
+-----------------------------------------+        |
|                   ...                   |        v
+-----------------------------------------+       ---
```

The first 8 octets form the fixed-length header which includes:

- **version**        specifies MIP version to use.
- **kind**           specifies the message type (for multi-endpoint invocation)
- **subtype**        specifies request, response or a concrete error type.
- **reserved**       reserved for later usage.
- **remaining size** specifies how many bytes the variable payload takes up.


The implementation of MIP by KubeMo can be found in files named [server.py](https://github.com/kubemo/kubemo/blob/main/kubemo/server.py), [client.py](https://github.com/kubemo/kubemo/blob/main/kubemo/client.py) and [protocol.py](https://github.com/kubemo/kubemo/blob/main/kubemo/protocol.py).


### Inference

When *kind* equals to 2, which indicates an inference invocation, the following 4 octets are required to include:

- **n-input**    specifies the number of the model's inputs.
- **n-output**   specifies the number of the model's outputs.
- **batch size** specifies the batch size of the following input/output.

The rest of the packet is a series of *TLV* formatted binaries currying a flattened batch of inputs or outputs. So the receiver of the packet needs to turn the flattened batch into, say, a two-dimmensional array according to *n-input*, *n-output* and *batch size*. And the sender oof the packet needs to encode the batch input/output into a flattened array of binaries.

The sender also needs to set *subtype* to 0 to tell that the packet is a request sent to a receiver who in return sets *subtype* to 1 which indicates a response packet.

### Ping

Besides an inference call, you can send a ping packet to check out if a server is available, with *kind* set to 1 and *subtype* set to 0. And the server needs to respond a packet with *kind* set to 1 and *subtype* set to 1. There is no variable payload for a ping packet.


### Errors

No request can always make a successful call to a server. The server may send back a response currying an error by setting *kind* to 0 and *subtype* to whatever the concrete error is. Currently, there are six types of errors in the design.

- *ERROR_PROTOCOL* used when receiving an invalid version.
- *ERROR_SUBTYPE* used when receiving an invalid subtype. e.g. the packet received by a server should always have *subtype* set to 0.
- *ERROR_METHOD* used when *kind* is not what the server has implemented. e.g. inference, ping
- *ERROR_MEMORY* used when the server runs out of memory.
- *ERROR_SHAPE* used when variable payload does not match its header.
- *ERROR_INTERNAL* used when other errors occur.
