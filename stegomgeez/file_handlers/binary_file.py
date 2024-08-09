import abc
import os
from pathlib import Path

from file_handlers.base_file import _BaseFile
from aiofiles import open as aopen


class BinaryFile(_BaseFile):

    async def save_to_file(self, *args, **kwargs) -> None:
        path = os.path.join(self.directory, f"{self.name}.{self.extension}")
        async with aopen(path, 'wb') as file:
            await file.write(self.content)

    async def copy_to(self, new_path: Path = None) -> None:
        if new_path is not None:
            new_path = new_path / f"{self.name}.{self.extension}"
        else:
            new_path = self.out_dir / f"{self.name}_original.{self.extension}"
        async with aopen(self.src_file_path, 'rb') as src_file:
            async with aopen(new_path, 'wb') as dst_file:
                while True:
                    chunk = await src_file.read(1024)
                    if not chunk:
                        break
                    await dst_file.write(chunk)


"""
    def __and__(self, j: Job) -> ndarray:
        return self.data & j.data

    def __or__(self, j: Job) -> ndarray:
        return self.data | j.data

    def __xor__(self, j: Job) -> ndarray:
        return self.data ^ j.data
"""