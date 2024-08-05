import os
from os import PathLike
from pathlib import Path

from file_handlers.base_file import BaseFile, FileContent
from aiofiles import open as aopen


class BinaryFile(BaseFile):
    async def read_from_file(self, size: int = None) -> FileContent:
        path = os.path.join(self.directory, f"{self.name}.{self.extension}")
        async with aopen(path, 'rb') as file:
            content = await file.read(size)
        return content

    async def save_to_file(self) -> None:
        path = os.path.join(self.directory, f"{self.name}.{self.extension}")
        async with aopen(path, 'wb') as file:
            await file.write(self.content)

    async def copy_to(self, new_directory: PathLike) -> None:
        old_path = Path(self.directory) / f"{self.name}.{self.extension}"
        new_path = Path(new_directory) / f"{self.name}.{self.extension}"
        async with aopen(old_path, 'rb') as src_file:
            async with aopen(new_path, 'wb') as dst_file:
                while True:
                    chunk = await src_file.read(1024)
                    if not chunk:
                        break
                    await dst_file.write(chunk)
