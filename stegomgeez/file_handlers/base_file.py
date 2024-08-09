import asyncio
import os
from abc import ABC, abstractmethod
from asyncio import AbstractEventLoop, Queue
from pathlib import Path
from typing import Union, Dict
from aiofiles.os import remove

FileContent = Union[bytes, str, None]
MetaData = Dict[str, Union[str, int, float]]


class _BaseFile(ABC):
    """Represents a simple Unix filetype."""

    __slots__ = ('name', 'directory', 'extension', 'content', 'mime_type', 'format',
                 'size', 'permissions', 'atime', 'mtime', 'ctime', '_out_dir', '_task_queue', '_loop')

    def __init__(
            self, name: str, directory: Path, extension: str, content: FileContent = None,
            mime_type: str = '', size: int = 0, permissions: str = '', file_format: str = None,
            atime: str = '', mtime: str = '', ctime: str = '', loop: AbstractEventLoop = None):
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
        self._out_dir: Path | None = None
        self._loop = loop if loop is not None else asyncio.get_event_loop()
        self._task_queue: Queue | None = None

    @abstractmethod
    async def _feed_tasks(self, tasks):
        raise NotImplementedError

    @abstractmethod
    async def worker(self, tasks) -> None:
        raise NotImplementedError

    @staticmethod
    def change_name(val1: str, val2: str) -> str:
        if val2 == "True":
            return f"_{val1}"
        return f"_{val1}-{val2}"

    @classmethod
    def __subclasshook__(cls, subclass):
        if cls is _BaseFile:
            required_methods = {'save_to_file', 'copy_to', '__str__', '__repr__'}
            if all(any(method in B.__dict__ for B in subclass.__mro__) for method in required_methods):
                return True
        return NotImplemented

    @abstractmethod
    async def save_to_file(self) -> None:
        """Saves content to file."""
        raise NotImplementedError

    @abstractmethod
    async def copy_to(self, new_directory: str) -> None:
        """Copies the file to a new directory asynchronously."""
        raise NotImplementedError

    @property
    def out_dir(self) -> Path:
        return self._out_dir

    @out_dir.setter
    def out_dir(self, new_dir: Path):
        self._out_dir = new_dir

    @property
    def task_queue(self) -> Queue:
        return self._task_queue

    @task_queue.setter
    def task_queue(self, queue: Queue):
        self._task_queue = queue


    @property
    def src_file_path(self):
        return self.directory / f"{self.name}.{self.extension}"

    @property
    def dst_file_path(self):
        if not self.out_dir:
            raise AttributeError('out_dir must be set')
        return self.out_dir / f"{self.name}.{self.extension}"

    async def delete(self) -> None:
        """Deletes the file asynchronously."""
        path = os.path.join(self.directory, f"{self.name}.{self.extension}")
        await remove(path)

    def _get_file_metadata(self) -> MetaData:
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

    def __hash__(self) -> int:
        return hash(self.content)

    def __len__(self):
        return len(self.content)

    def __eq__(self, file) -> bool:
        return hash(self.content) == hash(file.content)

    def __str__(self):
        return f"{self.name}.{self.extension}"

    def __repr__(self):
        raise NotImplementedError
