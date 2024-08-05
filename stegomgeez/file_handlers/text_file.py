import os
from os import PathLike
from pathlib import Path
from aiofiles import open as aopen
from file_handlers.base_file import BaseFile, FileContent


class TextFile(BaseFile):
    async def read_from_file(self, size: int = None) -> FileContent:
        path = os.path.join(self.directory, f"{self.name}.{self.extension}")
        async with aopen(path, 'r') as src_file:
            content = await src_file.read(size)
        return content

    async def save_to_file(self) -> None:
        path = os.path.join(self.directory, f"{self.name}.{self.extension}")
        async with aopen(path, 'w') as dst_file:
            await dst_file.write(self.content)

    async def copy_to(self, new_directory: PathLike) -> None:
        old_path = Path(self.directory) / f"{self.name}.{self.extension}"
        new_path = Path(new_directory) / f"{self.name}.{self.extension}"
        async with aopen(old_path, 'r') as src_file:
            async with aopen(new_path, 'w') as dst_file:
                while True:
                    chunk = await src_file.read(1024)
                    if not chunk:
                        break
                    await dst_file.write(chunk)
