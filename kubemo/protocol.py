
'''
        Model Invocation Protocol

0         7         15        23         31
+---------+----------+---------+----------+       ---
| version |   kind   | subtype | reserved |        ^      
+---------+---------+---------+-----------+   fixed header 
|             remaining size              |        v
+---------+----------+--------------------+       ---
| n-input | n-output |     batch size     |        ^
+---------+----------+--------------------+        |
|                   ...                   |        |
+-----------------------------------------+        |
|            input/output type            |        |
+-----------------------------------------+  variable payload (on *kind* == 1)
|            input/output size            |        |
+-----------------------------------------+        |
|            input/output data            |        |
+-----------------------------------------+        |
|                   ...                   |        v
+-----------------------------------------+       ---

MIP, aka model invocation protocol, is used for ML model invocation. The first
8 octets form the fixed-length header which includes:
**version**        specifies MIP version to use.
**kind**           specifies message type (for multi-endpoint invocation)
**subtype**        specifies request, response or a concrete error type.
**reserved**       reserved for later usage.
**remaining size** specifies how many bytes of payload to read.

When *kind* equals to 1, which indicates an Inferencing call, the following 4
octets are required to include:
**n-input**    specifies the number of the model's inputs.
**n-output**   specifies the number of the model's outputs.
**batch size** specifies the batch size of the following input/output.

The rest of the message uses a series of *TLV* formats to curry a batch of input
or output.
'''
from typing import BinaryIO, Tuple
from io import BytesIO
import struct

FIXED_HEADER_FMT = '!4BL'
FIXED_HEADER_LEN = 8
PROTOCOL_VERSION = 0
RESERVED_BYTE = 0

MESSAGE_ERROR = 0
MESSAGE_PING = 1
MESSAGE_INFERENCE = 2

MESSAGE_REQUEST = 0
MESSAGE_RESPONSE = 1

ERROR_PROTOCOL = 0
ERROR_SUBTYPE = 1
ERROR_METHOD = 2
ERROR_MEMORY = 3
ERROR_SHAPE = 4
ERROR_INTERNAL = 5

INFERENCE_HEADER_FMT = '!2BH'
INFERENCE_TL_FMT = '!2L'

TEXT = 1
JSON = 2
IMAGE = 3


def decode_single_input(reader: BinaryIO) -> Tuple[int, BinaryIO]:
    input_type, input_size = struct.unpack('!2L', reader.read(8))
    return input_type, BytesIO(reader.read(input_size))
