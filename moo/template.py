from typing import Union, Dict, List, BinaryIO
from io import BytesIO
from PIL import Image
from .protocol import INFERENCE_IMAGE, INFERENCE_TEXT, INFERENCE_JSON

import json
import numpy


class Input:

    def __init__(self, kind: int, reader: BinaryIO) -> None:
        '''A dynamic input

        Args:
        kind: An integer specifying input type.
        reader: A BineryIO for reading the input data.

        Example:

        >>> from moo import IMAGE, Input
        >>>
        >>> x = Input(IMAGE, open('foo.jpg', 'rb'))
        >>> img = x.as_image()
        >>> img.show()
        '''
        self.kind = kind
        self.reader = reader


    def as_image(self) -> Image:
        '''
        Returns the underlying input data as a PIL image.

        Raises: exceptions occur if the underlying input data is not an image.
        '''
        return Image.open(self.reader)

    def as_str(self) -> str:
        '''
        Returns the underlying input data as a string.

        Raises: exceptions occur if the underlying input data is not a string.
        '''
        return self.reader.read().decode()


class Output:

    @property
    def kind(self) -> int:
        raise NotImplementedError

    def encode(self) -> bytes:
        raise NotImplementedError


class TextOutput(Output):

    def __init__(self, s: str) -> None:
        self.s = s

    def encode(self) -> bytes:
        return self.s.encode()

    @property
    def kind(self) -> int:
        return INFERENCE_TEXT

    def __repr__(self) -> str:
        return self.s


class JsonOutput(Output):

    def __init__(self, d: Union[Dict, List]) -> None:
        self.d = d

    def encode(self) -> bytes:
        return json.dumps(self.d, cls=JsonEncoder).encode()

    @property
    def kind(self) -> int:
        return INFERENCE_JSON

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
    def kind(self) -> int:
        return INFERENCE_IMAGE

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


class DynamicOutput(Output):

    def __init__(self, kind: int, reader: BinaryIO) -> None:
        self.kind = kind
        self.reader = reader

    def encode(self) -> bytes:
        return self.reader.read()

    def text(self) -> str:
        if self.kind != INFERENCE_JSON:
            raise TypeError('output connot be decoded as a string')
        return self.reader.read().decode()

    def json(self) -> object:
        if self.kind != INFERENCE_JSON:
            raise TypeError('output connot be decoded as JSON')
        return json.load(self.reader)
    
    def image(self) -> Image.Image:
        if self.kind != INFERENCE_IMAGE:
            raise TypeError('output connot be decoded as a PIL image ')
        return Image.open(self.reader)