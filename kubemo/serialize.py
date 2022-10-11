from typing import BinaryIO, Union
from io import StringIO
from kubemo.protocol import IMAGE, JSON, TEXT
from PIL import Image as PIL_Image

import json
import numpy


class Serial:

    def __init__(self, kind: int, reader: BinaryIO) -> None:
        self.kind = kind
        self.reader = reader

    
    def text(self) -> str:
        if self.kind == TEXT or self.kind == JSON:
            return self.reader.read()
        
        raise TypeError('data connot be decoded as a string')
        
    def json(self) -> object:
        if self.kind == JSON:
            return json.load(self.reader)
        raise TypeError('data connot be decoded as a JSON object')
        

    def image(self) -> PIL_Image.Image:
        if self.kind == IMAGE:
            return PIL_Image.open(self.reader)
        raise TypeError('data connot be decoded as a PIL image')
        

    def encode(self) -> bytes:
        b = self.reader.read()
        if isinstance(b, str):
            return b.encode()
        return b


class Input(Serial): ...

class Output(Serial): ...


class Text(Serial):

    def __init__(self, text: str) -> None:
        super().__init__(TEXT, StringIO(text))

    def __repr__(self) -> str:
        return self.text()


class Json(Serial):

    def __init__(self, obj: object) -> None:
        buffer = StringIO()
        json.dump(obj, buffer, cls=JsonEncoder)
        buffer.seek(0)
        super().__init__(JSON, buffer)

    def __repr__(self) -> str:
        return self.json().__repr__()


class Image(Serial):

    def __init__(self, fp: Union[str, BinaryIO]) -> None:
        reader = fp
        if isinstance(fp, str):
            reader = open(fp, 'rb') # todo: how am I supposed to close fp
        super().__init__(IMAGE, reader)

    def __repr__(self) -> str:
        return self.image().__repr__()


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