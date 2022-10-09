from typing import Union, Dict, List
from io import BytesIO
from PIL import Image

import json
import numpy


class Input:

    def __init__(self, raw: bytes) -> None:
        '''A dynamic input

        Args:
        raw: Bytes containing input data.
        '''
        self.raw = raw


    @property
    def type(self) -> str:
        return 'dynamic'

    def as_image(self) -> Image:
        '''
        Returns the underlying input data as a PIL image.

        Raises: exceptions occur if the underlying input data is not an image.
        '''
        return Image.open(BytesIO(self.raw))

    def as_str(self) -> str:
        '''
        Returns the underlying input data as a string.

        Raises: exceptions occur if the underlying input data is not a string.
        '''
        return self.raw.decode()


class Output:

    @property
    def type(self) -> str:
        raise NotImplementedError

    def encode(self) -> bytes:
        raise NotImplementedError


class TextOutput(Output):

    def __init__(self, s: str) -> None:
        self.s = s

    def encode(self) -> bytes:
        return self.s.encode()

    @property
    def type(self) -> str:
        return 'text'

    def __repr__(self) -> str:
        return self.s


class JsonOutput(Output):

    def __init__(self, d: Union[Dict, List]) -> None:
        self.d = d

    def encode(self) -> bytes:
        return json.dumps(self.d, cls=JsonEncoder).encode()

    @property
    def type(self) -> str:
        return 'json'

    def __repr__(self) -> str:
        return self.d.__repr__()


class ImageOutput(Output):

    def __init__(self, o: Union[str, bytes, Image.Image]) -> None:
        '''An image output

        Args:
        o: A union typed object that can be one of:
            1. A string refering to the local path to an image.
            2. A serial of bytes of an image.
            3. A PIL image object containing an image.
        '''
        self.o = o

    def encode(self) -> bytes:
        '''Serializes the underlying image.

        Returns: Entire bytes of the underlying image.

        Raises:
        TypeError: An error occurred when the underlying data is not one of the
            required types.
        '''
        if isinstance(self.o, str):
            with open(self.o, 'rb') as f:
                return f.read()
        elif isinstance(self.o, Image.Image):
            buf = BytesIO()
            self.o.save(buf, format='PNG')
            return buf.getvalue()
        elif isinstance(self.o, bytes):
            return self.o
        else:
            raise TypeError(f'invalid image output content: {type(self.o)}')
    
    @property
    def type(self) -> str:
        return 'image'

    def __repr__(self) -> str:
        return self.o.__repr__()


class JsonEncoder(json.JSONEncoder):
    '''A numpy-compatible JSON encoder.

    This class is taken from https://bobbyhadz.com/blog/python-typeerror-object-of-type-float32-is-not-json-serializable
    '''
    def default(self, o):
        if isinstance(o, numpy.integer):
            return int(o)
        if isinstance(o, numpy.floating):
            return float(o)
        if isinstance(o, numpy.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)
