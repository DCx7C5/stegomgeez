import os
from abc import ABC, abstractmethod
from os import PathLike
from typing import Union, Dict
from aiofiles.os import remove

FileContent = Union[bytes, str, None]
MetaData = Dict[str, Union[str, int, float]]


class BaseFile(ABC):
    """Represents a simple Unix filetype."""

    __slots__ = ('name', 'directory', 'extension', 'content', 'mime_type', 'format',
                 'size', 'permissions', 'atime', 'mtime', 'ctime')

    def __init__(
            self, name: str, directory: PathLike, extension: str, content: FileContent = None,
            mime_type: str = '', size: int = 0, permissions: str = '', file_format: str = None,
            atime: str = '', mtime: str = '', ctime: str = ''):
        self.name = name
        self.directory = directory
        self.extension = extension
        self.content = content
        self.mime_type = mime_type
        self.format = file_format
        self.size = size
        self.permissions = permissions
        self.atime = atime
        self.mtime = mtime
        self.ctime = ctime

    @classmethod
    def __subclasshook__(cls, subclass):
        if cls is BaseFile:
            required_methods = {'read_from_file', 'save_to_file', 'copy_to', '__str__', '__repr__'}
            if all(any(method in B.__dict__ for B in subclass.__mro__) for method in required_methods):
                return True
        return NotImplemented

    @abstractmethod
    async def read_from_file(self, size: int = None) -> FileContent:
        """Reads content from a file."""
        raise NotImplementedError

    @abstractmethod
    async def save_to_file(self) -> None:
        """Saves content to file."""
        raise NotImplementedError

    @abstractmethod
    async def copy_to(self, new_directory: str) -> None:
        """Copies the file to a new directory asynchronously."""
        raise NotImplementedError

    async def delete(self) -> None:
        """Deletes the file asynchronously."""
        path = os.path.join(self.directory, f"{self.name}.{self.extension}")
        await remove(path)

    def get_metadata(self) -> MetaData:
        """Returns the metadata of the file."""
        return {
            "name": self.name,
            "directory": self.directory,
            "extension": self.extension,
            "mime_type": self.mime_type,
            "format": self.format,
            "size": self.size,
            "permissions": self.permissions,
            "atime": self.atime,
            "mtime": self.mtime,
            "ctime": self.ctime
        }

    def __len__(self):
        return len(self.content)

    def __str__(self):
        return f"{self.name}.{self.extension}"

    def __repr__(self):
        metadata = self.get_metadata()
        metadata_str = ", ".join(f"{key}={value!r}" for key, value in metadata.items())
        return f"{self.__class__.__name__}({metadata_str})"
