from typing import Union

import cv2
from os import PathLike
from pathlib import Path
from numpy import ndarray, uint8, frombuffer

from file_handlers.binary_file import BinaryFile
from file_handlers.image_file import ImageFile
from file_handlers.text_file import TextFile
from utils import get_mime_from_buffer, load_data

FileType = Union[ImageFile, BinaryFile, TextFile]


class AsyncFileFactory:

    @staticmethod
    async def _parse_image(file_path: PathLike, data: bytes, mime: str) -> ImageFile:
        raw_array: ndarray = frombuffer(data, uint8)
        cvimg: ndarray = cv2.imdecode(raw_array, cv2.IMREAD_UNCHANGED)
        return ImageFile(file_path, cvimg, mime)

    @staticmethod
    async def _parse_text(file_path: PathLike, data: str,):
        pass

    @staticmethod
    async def _parse_binary(file_path: PathLike, data: bytes,):
        pass

    @classmethod
    async def _parse_file_type(cls, file_path: PathLike, data: bytes, ftype: str, mime: str) -> FileType:
        f, fp = None, Path(file_path).absolute()
        if ftype == 'image':
            f = await cls._parse_image(fp, data, mime)
        elif ftype == 'text':
            f = await cls._parse_text(fp, data.decode('utf-8'))
        elif ftype == 'application':
            f = await cls._parse_binary(fp, data)
        if f is not None:
            return f
        raise TypeError(f'{cls.__name__} was not able to parse file type of {file_path}')

    @classmethod
    async def create_from_bytes(cls, name: str, output_dir: PathLike, data: bytes) -> FileType:
        mime: str = get_mime_from_buffer(data)
        file_type, file_subtype = mime.split('/')
        file_path = Path(output_dir, name).absolute()
        return await cls._parse_file_type(file_path, data, file_type, mime)

    @classmethod
    async def create_from_path(cls, file_path: PathLike) -> FileType:
        data: bytes = await load_data(file_path)
        mime: str = get_mime_from_buffer(data)
        file_type, file_subtype = mime.split('/')
        return await cls._parse_file_type(file_path, data, file_type, mime)
