from __future__ import annotations
import asyncio
from asyncio import AbstractEventLoop
import uvloop
from os import (
    get_terminal_size,
    getuid, isatty,
    getcwd,
)

from pathlib import Path
from pwd import getpwuid

from aiofiles import open
import magic

from argparse import Namespace, ArgumentParser
from PIL import Image
from numpy import array, ndarray
from rich.console import Console
from transforms.bands import get_band
from transforms.geometric import change_angle, flip_image
from ciphers import encrypt, decrypt
from cupy import ndarray, array
from utils import (
    get_mime_from_file,
    save_to_disk,
    pil2opencv,
    load_data,
    save_png,
    log_sem, RGBA_VALS,
)

from typing import (
    AsyncIterator,
    Sequence,
    Callable,
    Optional,
    Literal,
    Union,
    Tuple,
    List,
    Any,
)

PilImageType = Image.Image
Pathslist = Tuple[Image.Image, Path, str]
CombinationItem = Tuple[Callable, Union[str, int]]
Combination = List[CombinationItem]
CombinationList = List[Combination]
AnyDataContent = Union[str, bytes, Image.Image, bytes]
ChangeAngleFunc = Callable[[Image.Image, int], Image.Image]
FlipImageFunc = Callable[[Image.Image, Literal['ver', 'hor']], Image.Image]
GetBandFunc = Callable[[Image.Image, str], Image.Image]
TransformType = Union[ChangeAngleFunc, FlipImageFunc, GetBandFunc]


console = Console(
    color_system='truecolor',
    force_terminal=True,
    width=999 if not isatty(1) else get_terminal_size().columns,
)


HOME_DIR = Path(f'/home/{getpwuid(getuid())[0]}')
APP_DIR = HOME_DIR / 'STEGOSUITE'

if not APP_DIR.exists():
    APP_DIR.mkdir()


async def load_jobs(func: Callable, *args: Any):
    async with asyncio.TaskGroup() as tg:
        await tg.create_task(func(*args))


class Job:
    __slots__ = (
        'tasks', 'image', 'output_dir', 'filename', 'ciphers',
        'debug', 'rw', 'mime', 'ext', 'task_queue', '_cvimg'
    )

    def __init__(self, img: PilImageType, tasks: CombinationList,
                 output_dir: Path, filename: str, ext: str, mime: str, rw: bool, debug: bool):
        tasks = [t for t in tasks if t is not None]
        self.image: PilImageType = img
        self.tasks: CombinationList = tasks
        self.output_dir: Path = output_dir
        self.filename: str = filename
        self.ext: str = ext
        self.mime: str = mime
        self.rw: bool = rw
        self._cvimg: Optional[ndarray] = None
        self.debug: bool = debug

        self.task_queue = asyncio.Queue()
        for task in tasks:
            self.task_queue.put_nowait(task)

        asyncio.create_task(self.worker())

    async def worker(self) -> None:
        while not self.task_queue.empty():
            function, params = await self.task_queue.get()
            self.image = function(self.image, params)
            self.change_filename(function.__name__, params)
            if self.task_queue.empty() and (self.mime == 'image/jpeg') and (function.__name__ == 'get_band'):
                await self.save(self.image, True)
            elif self.task_queue.empty():
                await self.save(self.image, False)

    def change_filename(self, last_func_name: str, last_param: str | int):
        if last_func_name == 'change_angle':
            self.filename += f"_angle-{last_param}"
        elif last_func_name == 'flip_image':
            self.filename += f"_flip-{last_param}"
        elif last_func_name == 'get_band':
            self.filename += f"_chan-{last_param}"

    async def save(self, obj: AnyDataContent, convert: bool = False,
                   keep_rgb: bool = True, quality: int = 95):
        """
        Saves image to disc
        """
        file_name = self.filename + '.' + self.ext
        abs_path = self.output_dir / file_name
        await save_to_disk(obj, abs_path, convert, keep_rgb, quality)

    @property
    def data(self) -> Sequence | array:
        return self.image.getdata()

    @property
    def cvimg(self) -> ndarray:
        if self._cvimg is None:
            self._cvimg = pil2opencv(self.image)
        return self._cvimg

    @property
    def has_transp(self) -> bool:
        return self.image.has_transparency_data

    @property
    def file_size(self) -> int:
        return len(self.raw)

    @property
    def raw(self) -> bytes:
        return self.image.tobytes('raw')

    @property
    def size(self) -> Tuple[int, int]:
        return self.image.size

    def __and__(self, j: Job) -> ndarray:
        return self.data & j.data

    def __or__(self, j: Job) -> ndarray:
        return self.data | j.data

    def __xor__(self, j: Job) -> ndarray:
        return self.data ^ j.data

    def __eq__(self, j: Job) -> bool:
        return self.data == j.data

    def __hash__(self) -> int:
        return hash(self.data)


class Manager:

    def __init__(self, project_name: str, file_paths: List[Path], options: Namespace, _loop=None):
        self.loop: AbstractEventLoop = _loop if _loop is not None else asyncio.get_event_loop()
        self.project_name: str = project_name
        self.filepaths: List[Path] = file_paths
        self.options: Namespace = options
        self.output_dir: Path = options.outdir / f"{project_name}"
        self.scan_depth: int = options.depth
        self.debug: bool = options.debug

        if not self.output_dir.exists():
            self.output_dir.mkdir()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        # cleanup
        pass

    async def load_file_paths(self) -> AsyncIterator[Pathslist]:
        """Loads the file paths that the user put as params at program start"""
        for path in self.filepaths:
            mime_type = magic.from_file(path, mime=True)
            if mime_type in ['image/jpeg', 'image/png']:
                image = Image.open(path)
                image.load()
            else:
                image = await load_data(path)
            yield image, path, mime_type

    async def create_backup(self, filename: str, data: PilImageType | bytes, src_path: Path):
        """
        Creates a copy of the users image in project directory and
        a decompressed version of the image if it's a PNG.
        """
        file_name, ext = filename.split('.')
        file_name = file_name + '_original.' + ext
        backup_path = self.output_dir / file_name
        if ext in ['png', 'PNG']:
            png_name, ext = filename.split('.')
            png_name = png_name + '_decompressed.' + ext
            dec_path = self.output_dir / png_name
            save_png(data, dec_path)
        # save_image_to_disk(data, backup_path)
        async with open(src_path, 'rb') as src:
            async with open(backup_path, 'wb') as dst:
                await dst.write(await src.read())

    async def parse_transformation_ruleset(
        self, image: PilImageType, mime: str
    ) -> AsyncIterator[CombinationList]:
        """
        Creates combinations of all transformation rules applied to the image
        """
        rules, bands, mirrors, angles = [], [], [], []
        opt_angles = self.options.angles
        opt_mirror = self.options.mirror
        opt_bands = self.options.bands

        angles = [int(r) for r in opt_angles if int(r) < 360]

        if len(opt_mirror) > 0:
            mirrors = [m for m in opt_mirror] + [None]
        if len(opt_bands) > 0:
            if not image.has_transparency_data:
                band_extractions = [x for x in RGBA_VALS if 'A' not in x]
            else:
                band_extractions = RGBA_VALS

            bands = [b for b in opt_bands if b in band_extractions]

        combs = []
        if mime == "image/jpeg":
            combs = ([[(change_angle, a), (flip_image, m) if m is not None else None] for a in angles for m in mirrors]
                     + [[(get_band, b)] for b in bands])
        elif mime == "image/png":
            combs = [[(change_angle, a), (flip_image, m) if m is not None else None, (get_band, b)]
                     for a in angles for m in mirrors for b in bands]

        for x in combs:
            yield x

    async def do_report(self, name: str, vals: Tuple[str, int, float, int], force_overwrite: bool = False):
        await log_sem.acquire()
        rep_log = (f"{name} - Type: {name[-3:].upper()}, Method: {vals[0]}, Size: {vals[1]},"
                   f" Entropy: {vals[2]}, Filesize: {vals[3]}")
        async with open(f"{self.output_dir}" + "report.txt", 'r+') as file:
            lines = await file.readlines()
            for line in lines:
                if name in line:
                    if force_overwrite:
                        ln = lines.index(line)
                        lines[ln] = rep_log
                else:
                    lines.append(rep_log)
                    lines.sort()
            await file.writelines(lines)

    async def load(self, image: PilImageType | bytes,
                   path: Path, mime: str, combination: CombinationList) -> Job:
        """Instantiates a Job class"""
        filename, ext = path.name.split('.')
        return Job(
            tasks=combination,
            img=image,
            output_dir=self.output_dir,
            filename=filename,
            ext=ext,
            mime=mime,
            rw=self.options.force,
            debug=self.debug,
        )

    async def start_manager(self) -> None:
        """Starts the manager"""
        async for img, path, mime in self.load_file_paths():
            await self.create_backup(path.name, img, path)
            async for combi in self.parse_transformation_ruleset(img, mime):
                async with asyncio.TaskGroup() as tg:
                    await tg.create_task(self.load(img, path, mime, combi))


async def main(ev_loop: AbstractEventLoop):
    parser = ArgumentParser(prog="stegomgeez", description="Steganography toolset", )
    pos = parser.add_argument_group(title='Positional arguments:')
    pos.add_argument(
        "filepath",
        help="One or multiple image files",
        nargs="+",
        default=[],
        action="store",
    )
    pos.add_argument(
        "projectname",
        help="Name of the project.",
        action="store", type=str,
    )
    mut = parser.add_argument_group(
        title="Image mutations",
        description='Create different mutations of the image',
    )
    mut.add_argument(
        "--angles", "-a",
        help="List of angles to apply to the image. Everything between 0° and 360°",
        default=[], nargs="+",
        required=False,
    )
    mut.add_argument(
        "--mirror", "-m",
        help="Mirrors the picture.",
        default="both", type=str,
        choices=("hor", "ver", "all"),
        nargs='+', required=False,
    )
    mut.add_argument(
        "--bands",
        help="Bands to extract from the image.",
        default=[], required=False,
        choices=RGBA_VALS, nargs='+'
    )
    meth = parser.add_argument_group(title="Methods to apply")
    meth.add_argument(
        "--lsb",
        help="Extracts LSB for further analysis",
        required=False, default=[],
    )

    parser.add_argument(
        "--outdir", "-o",
        help="Path to the output directory",
        default=f"{APP_DIR}",
    )

    parser.add_argument(
        "--force", "-f",
        help="Overwrite existing files",
        action='store_true',
    )
    parser.add_argument(
        "--depth", "-d",
        help="Scan depth of files",
        default=5, type=int,
    )
    parser.add_argument(
        "--debug",
        help="Debug mode",
        default=False, action='store_true',
    )

    args = parser.parse_args()

    if args.mirror == ["all"]:
        args.mirror = ["hor", "ver"]
    else:
        args.mirror = args.mirror

    if args.bands == ["all"]:
        args.bands = RGBA_VALS

    if "0" not in args.angles:
        args.angles = ["0"] + args.angles

    args.outdir = Path(args.outdir).absolute()
    if args.debug:
        ev_loop.set_debug(True)
        args.outdir = Path(getcwd()).absolute() / '..' / "TESTOUT"
        if not args.outdir.exists():
            args.outdir.mkdir()

    async with Manager(
            project_name=args.projectname,
            file_paths=[Path(p).absolute() for p in args.filepath],
            options=args,
            _loop=ev_loop,
    ) as mngr:
        await mngr.start_manager()


if __name__ == '__main__':
    try:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main(loop))
    except KeyboardInterrupt:
        pass
