from datetime import datetime
from os import PathLike, stat
from pathlib import Path
from typing import Union
from numpy import ndarray
from file_handlers.binary_file import BinaryFile


ImageContentType = Union[bytes, bytearray, ndarray]


class ImageFile(BinaryFile):
    """Represents a JPEG/PNG file."""
    def __init__(self, file_path: PathLike, content: ImageContentType, mime_type: str,):
        fp, f_fmt = Path(file_path).absolute(), mime_type.split('/')[-1].upper()
        f_size, f_perm = content.size, oct(stat(fp).st_mode & 0o777)
        f_dir, f_name, f_ext = fp.parent, fp[:-len(fp.suffix)], fp.suffix[1:]
        astmp, mstmp, cstmp = fp.stat()[-3:]
        atime = datetime.fromtimestamp(astmp).strftime('%Y:%m:%d %H:%M:%S')
        mtime = datetime.fromtimestamp(mstmp).strftime('%Y:%m:%d %H:%M:%S')
        ctime = datetime.fromtimestamp(cstmp).strftime('%Y:%m:%d %H:%M:%S')
        super().__init__(f_name, f_dir, f_ext, content, mime_type,
                         f_size, f_perm, f_fmt, atime, mtime, ctime)


    def __repr__(self):
        return (

            super().__repr__()
        )
