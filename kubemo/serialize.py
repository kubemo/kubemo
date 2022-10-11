from abc import abstractmethod
from typing import IO, AnyStr, BinaryIO, Dict, Generic, TextIO, Type, TypeVar, Union
from pathlib import Path
from io import BytesIO, StringIO, BufferedReader
from kubemo.protocol import IMAGE, JSON, TEXT
from PIL import Image as PIL_Image

import json
import numpy


T = TypeVar('T')

class Serial(Generic[T]):
    '''A generic interface for serialization and deserialization.'''

    def __init__(self, kind: int, reader: IO[AnyStr]) -> None:
        self.kind = kind
        self.reader = reader

    def encode(self) -> bytes:
        '''Returns the underlying data as raw bytes.
        '''
        b = self.reader.read()
        if isinstance(b, str):
            return b.encode()
        return b

    @abstractmethod
    def decode(self) -> T:
        '''Decodes the underlying data as a specific type.'''



class Input(Serial): ...

class Output(Serial): ...


class Dynamic(Serial):

    def __init__(self, kind: int, reader: IO[AnyStr]) -> None:
        self.serial = _default_serializations[kind](reader)
    
    def __repr__(self) -> str:
        return self.decode().__repr__()

    def decode(self) -> T:
        return self.serial.decode()

    def encode(self) -> bytes:
        return self.serial.encode()


class Text(Serial[str]):

    def __init__(self, s: Union[str, BinaryIO, TextIO, BytesIO, StringIO]) -> None:
        '''Creates a text input/ouput.

        Args:
            s: A string or a BinaryIO object containing the text.

        Raises:
            TypeError

        Example:
        >>> from kubemo import Text
        >>>
        >>> x = Text('kubemo is awesome!')
        >>> x.decode()
        >>>
        '''
        if isinstance(s, str):
            reader = BytesIO(s.encode())
        elif isinstance(s, (TextIO, StringIO, BinaryIO, BytesIO)):
            reader = s
        else:
            raise TypeError(f'expected str or IO-like object, not {type(s)}')

        super().__init__(TEXT, reader)

    def __repr__(self) -> str:
        return self.decode()

    def decode(self) -> str:
        '''Decodes the underlying data as a string.

        Raises:
            TypeError: An error occurred when the underlying data can not be
                       deserialized as a string.
        '''
        if self.kind == TEXT or self.kind == JSON:
            b = self.reader.read()
            return b if isinstance(b, str) else b.decode()
        
        raise TypeError('data connot be decoded as a string')


class Json(Serial):

    def __init__(self, obj: Union[object, BinaryIO, TextIO, BytesIO, StringIO]) -> None:
        '''Creates a JSON input/ouput.

        Args:
            obj: An object that can be serialized to a JSON string.

        Example:
        >>> from kubemo import Json
        >>>
        >>> o = {'foo': 'bar'}
        >>> x = Json(o)
        >>> x.decode()
        >>>
        '''
        
        if isinstance(obj, (TextIO, StringIO, BinaryIO, BytesIO)):
            buffer = obj
        else:
            buffer = StringIO()
            json.dump(obj, buffer, cls=JsonEncoder)
            buffer.seek(0)

        super().__init__(JSON, buffer)

    def __repr__(self) -> str:
        return self.decode().__repr__()

    def decode(self) -> object:
        '''Decodes the underlying data as a JSON object.

        Raises:
            TypeError: An error occurred when the underlying data can not be
                       deserialized as a JSON object.
        '''
        if self.kind == JSON:
            return json.load(self.reader)
        raise TypeError('data connot be decoded as a JSON object')


class Image(Serial):

    def __init__(self, fp: Union[str, Path, BufferedReader, PIL_Image.Image]) -> None:
        '''Creates a image input/ouput.

        Args:
            fp: An union typed argument that can be either the path to an image
                or a BineryIO created by `open`.

        Raises:
            TypeError

        Example:
        >>> from pathlib import Path
        >>> from PIL import Image as PIL_Image
        >>> from kubemo import Image
        >>>
        >>> x = Image('foo.jpg')
        >>> x = Image(Path('foo.jpg'))
        >>> x = Image(open('foo.jpg', 'rb))
        >>> x = Image(PIL_Image.open('foo.jpg'))
        >>> 
        >>> x.decode().show()
        >>>
        '''

        if isinstance(fp, (str, Path)):
            buffer = open(fp, 'rb') # todo: how am I supposed to close fp?
        elif isinstance(fp, (BinaryIO, BytesIO, BufferedReader)):
            buffer = fp
        elif isinstance(fp, PIL_Image.Image):
            buffer = BytesIO()
            fp.save(buffer, format=fp.format)
            buffer.seek(0)
        else:
            raise TypeError(f'expected str, IO-like object or PIL image, not {type(fp)}')

        super().__init__(IMAGE, buffer)

    def __repr__(self) -> str:
        return self.decode().__repr__()

    def decode(self) -> PIL_Image.Image:
        '''Decodes the underlying data as a PIL image.

        Raises:
            TypeError: An error occurred when the underlying data can not be
                       deserialized as PIL image.
        '''
        if self.kind == IMAGE:
            return PIL_Image.open(self.reader)
        raise TypeError('data connot be decoded as a PIL image')


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



_default_serializations: Dict[int, Type[Serial]] = {
    TEXT: Text,
    JSON: Json,
    IMAGE: Image,
}

def register(s: Type[Serial]):
    _default_serializations[s.kind] = s


