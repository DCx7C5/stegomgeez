from datetime import datetime
from os import stat
from pathlib import Path
from typing import Union, Literal, List

import cv2
from numpy import ndarray, zeros_like

from file_handlers.binary_file import BinaryFile

ImageContentType = Union[bytes, bytearray, ndarray]


class ImageFile(BinaryFile):
    """Represents a JPEG/PNG file."""
    def __init__(self, file_path: Path, content: ImageContentType, mime_type: str,):
        fp, f_fmt = Path(file_path).absolute(), mime_type.split('/')[-1].upper()
        f_size, f_perm = content.size, oct(stat(fp).st_mode & 0o777)
        f_dir, f_ext = fp.parent, fp.suffix[1:]
        f_name = str(fp)[:-len(fp.suffix)].split("/")[-1]
        astmp, mstmp, cstmp = fp.stat()[-3:]
        atime = datetime.fromtimestamp(astmp).strftime('%Y:%m:%d %H:%M:%S')
        mtime = datetime.fromtimestamp(mstmp).strftime('%Y:%m:%d %H:%M:%S')
        ctime = datetime.fromtimestamp(cstmp).strftime('%Y:%m:%d %H:%M:%S')
        super().__init__(f_name, f_dir, f_ext, content, mime_type,
                         f_size, f_perm, f_fmt, atime, mtime, ctime)
        self.bands_bgr = cv2.split(content)

    async def _feed_tasks(self, tasks):
        for task in tasks:
            self.task_queue.put_nowait(task)

    async def worker(self, tasks) -> None:
        await self._feed_tasks(tasks)
        while not self.task_queue.empty():
            func_name, params = await self.task_queue.get()
            if params is None:
                continue
            getattr(self, func_name)(params)
            self.name += self.change_name(func_name, params)
        self.save_to_file()

    @property
    def bands_rgb(self) -> List[ndarray]:
        return self.bands_bgr[:, :, ::-1]

    @property
    def has_transparency_channel(self) -> bool:
        return len(self.bands_bgr) == 4

    def reverse(self, arg: bool):
        if arg:
            self.content = self.content[:, :, ::-1]

    def flip(self, axis: Literal['x', 'y', 'xy']) -> None:
        """Flip the image horizontally, vertically or both."""
        if axis == 'x':
            fc = 0
        elif axis == 'y':
            fc = 1
        elif axis == 'xy':
            fc = -1
        else:
            raise ValueError(f'Axis {axis} not supported')
        self.content = cv2.flip(self.content, fc)

    def rotate(self, angle: int) -> None:
        """Rotate the image around the center by degrees."""
        if angle == 0 or angle == 360:
            return
        (height, width) = self.content.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        self.content = cv2.warpAffine(self.content, rotation_matrix, (width, height))

    def extract_channels(self, channels: str) -> None:
        channel_index = {'B': 0, 'G': 1, 'R': 2}
        if self.has_transparency_channel:
            channel_index.update({'A': 3})
        channel_img = zeros_like(self.content).copy()
        for channel in channels:
            channel_img[:, :, channel_index[channel]] = self.content[:, :, channel_index[channel]]
        self.content = channel_img

    def save_to_file(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def __repr__(self):
        metadata = self._get_file_metadata()
        metadata_str = ", ".join(f"{key}={value!r}" for key, value in metadata.items() if key != 'format')
        return f"<{metadata['format']} {self.__class__.__name__} ({metadata_str})>"


class JPGFile(ImageFile):

    def save_to_file(self, quality: int = 100) -> None:
        cv2.imwrite(self.dst_file_path, self.content, [1, quality])

    def save_as_png(self, comp_level: int = 0) -> None:
        cv2.imwrite(self.dst_file_path, self.content, [16, comp_level])


class PNGFile(ImageFile):
    def save_to_file(self, comp_level: int = 0) -> None:
        cv2.imwrite(self.dst_file_path, self.content, [16, comp_level])
