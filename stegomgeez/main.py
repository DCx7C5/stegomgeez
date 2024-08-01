from __future__ import annotations
import asyncio
import uvloop
from asyncio import AbstractEventLoop

from os import (getuid, isatty, get_terminal_size, getcwd)
from pathlib import Path
from types import FunctionType
from pwd import getpwuid

import aiofiles
import numpy as np
import magic
import logging

from argparse import Namespace, ArgumentParser
from typing import (List, Tuple, Sequence)
from PIL import Image
from rich.console import Console
from rich.logging import RichHandler
from transformers import pipeline
from ciphers import encrypt, decrypt
from transforms.bands import get_band
from transforms.geometric import change_angle, flip_image
from stegomgeez.helper import (
    save_image_to_disk,
    save_png,
    RGBA_VALS,
    load_data,
    log_sem,
    save_to_disk,
)

language_detection = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")

console = Console(
    color_system="truecolor",
    force_terminal=True,
    width=999 if not isatty(1) else get_terminal_size().columns,
)

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[
        RichHandler(
            console=console,
        )
    ]
)

HOME_DIR = Path(f"/home/{getpwuid(getuid())[0]}")
APP_DIR = HOME_DIR / "STEGOSUITE"

if not APP_DIR.exists():
    APP_DIR.mkdir()


def detect_language_transformers(text):
    try:
        result = language_detection(text)
        language = result[0]['label']
        score = result[0]['score']
        return language, score
    except Exception as e:
        return str(e)


async def extract_bits(image, bit_type, scan_direction):
    width, height = image.size
    bit_values = []
    for x in range(width if scan_direction == 'col' else height):
        for y in range(height if scan_direction == 'row' else width):
            if bit_type == 'lsb':
                bit_values += [(r & 1, g & 1, b & 1) for r, g, b in image.getpixel((x, y))]
            elif bit_type == 'msb':
                bit_values += [((r >> 7) & 1, (g >> 7) & 1, (b >> 7) & 1) for r, g, b in image.getpixel((x, y))]
    return bit_values


async def load_jobs(func, *args):
    async with asyncio.TaskGroup() as tg:
        await tg.create_task(func(*args))


class Job:
    __slots__ = (
        'tasks', 'image', 'output_dir', 'filename', 'ciphers',
        'loop', 'debug', 'rw', 'mime', 'ext', 'task_queue'
    )

    def __init__(self, img, tasks, output_dir, filename, ext, mime, rw, debug):
        tasks = [t for t in tasks if t is not None]
        self.image: Image = img
        self.tasks: List[Tuple[FunctionType, str | int],] | List[FunctionType, str | int] = tasks
        self.output_dir: Path = output_dir
        self.filename: str = filename
        self.ext: str = ext
        self.mime: str = mime
        self.rw: bool = rw
        self.loop: AbstractEventLoop = asyncio.get_event_loop()
        self.debug: bool = debug

        self.task_queue = asyncio.Queue()
        for task in tasks:
            self.task_queue.put_nowait(task)

        asyncio.create_task(self.worker())

    async def worker(self):
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

    async def save(self, obj: str | bytes | Image.Image, convert=False, quality=95):
        file_name = self.filename + '.' + self.ext
        abs_path = self.output_dir / file_name
        if self.debug:
            logging.debug(f"Saving to {self.output_dir} / {file_name}")
        await save_to_disk(obj, abs_path, convert, True, quality, self.loop)

    @property
    def data(self, numpy=False) -> Sequence | np.array:
        data = self.image.getdata()
        if numpy:
            return np.array(data, np.uint8)
        return data

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

    def __and__(self, j: Job):
        return self.data & j.data

    def __or__(self, j: Job):
        return self.data | j.data

    def __xor__(self, j: Job):
        return self.data ^ j.data

    def __eq__(self, j: Job):
        return self.data == j.data

    def __hash__(self):
        return hash(self.data)


class Manager:

    def __init__(self, project_name: str, file_paths: List[Path], options: Namespace, _loop=None):
        self.ext = None
        self.loop = _loop if _loop is not None else asyncio.get_event_loop()
        self.project_name = project_name
        self.filepaths = file_paths
        self.options: Namespace = options
        self.output_dir = options.outdir / f"{project_name}"
        self.scan_depth = options.depth
        self.debug = options.debug

        if not self.output_dir.exists():
            self.output_dir.mkdir()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        # cleanup
        pass

    async def load_file_paths(self):
        """Loads the file paths that the user put as params at program start"""
        for path in self.filepaths:
            mime_type = magic.from_file(path, mime=True)
            if mime_type in ['image/jpeg', 'image/png']:
                image = Image.open(path)
                image.load()
            else:
                image = await load_data(path)
            yield [image, path, mime_type]

    async def create_backup(self, filename: str, data: bytes | Image):
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
        save_image_to_disk(data, backup_path)

    async def parse_transformation_ruleset(self, image, mime: str):
        """Creates combinations of all transformation rules applied to the image"""
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
            combs = ([[(change_angle, a), (flip_image, m) if m is not None else None] for a in angles for m in mirrors] +
                     [[(get_band, b)] for b in bands])
        elif mime == "image/png":
            combs = [[(change_angle, a), (flip_image, m) if m is not None else None, (get_band, b)]
                     for a in angles for m in mirrors for b in bands]

        for x in combs:
            yield x

    async def do_report(self, name, vals, force_overwrite=False):
        await log_sem.acquire()
        rep_log = (f"{name} - Type: {name[-3:].upper()}, Method: {vals[0]}, Size: {vals[1]},"
                   f" Entropy: {vals[2]}, Filesize: {vals[3]}")
        async with aiofiles.open(f"{self.output_dir}" + "report.txt", 'r+') as file:
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
        logging.debug(rep_log)

    async def load(self, image: Image.Image | bytes, path: Path, mime: str, combination):
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

    async def start_manager(self):
        """Starts the manager"""
        async for img, path, mime in self.load_file_paths():
            await self.create_backup(path.name, img)
            async for combi in self.parse_transformation_ruleset(img, mime):
                async with asyncio.TaskGroup() as tg:
                    await tg.create_task(self.load(img, path, mime, combi))


async def main(ev_loop):
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
        logging.debug(args)

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
