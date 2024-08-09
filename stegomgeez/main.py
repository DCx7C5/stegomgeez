from __future__ import annotations
import asyncio
from asyncio import AbstractEventLoop
from uvloop import EventLoopPolicy
from os import (get_terminal_size, getuid, isatty, getcwd)
from argparse import Namespace, ArgumentParser
from pathlib import Path
from pwd import getpwuid
from PIL import Image
from rich.console import Console
from aiofiles import open as aopen
from file_handlers import AsyncFileFactory, ImageFile
from ciphers import encrypt, decrypt
from utils import (
    RGBA_VALS,
    generate_combinations,
)

from typing import (
    AsyncIterator,
    Callable,
    Literal,
    Union,
    Tuple,
    List,
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

    async def load_file_paths(self) -> AsyncIterator[Path]:
        """Loads the file path arguments"""
        for path in self.filepaths:
            yield path

    async def create_backup(self, src_path: Path):
        """
        Creates a copy of the users image in project directory and
        a decompressed version of the image if it's a PNG.
        """
        file_name, ext = src_path.name.split('.')
        file_name = file_name + '_original.' + ext
        backup_path = self.output_dir / file_name
        async with aopen(src_path, 'rb') as src:
            async with aopen(backup_path, 'wb') as dst:
                await dst.write(await src.read())

    async def parse_transformation_ruleset(self) -> AsyncIterator[CombinationList]:
        """
        Creates combinations of all transformation rules applied to the image
        """
        func_rot = ImageFile.rotate.__name__
        func_flip = ImageFile.flip.__name__
        func_bands = ImageFile.extract_channels.__name__
        func_reverse = ImageFile.reverse.__name__

        opt_angles = [(func_rot, a) for a in self.options.angles]
        opt_mirror = [(func_flip, f) for f in self.options.flip]
        opt_bands = [(func_bands, b) for b in self.options.bands]
        opt_rev = [(func_reverse, r) for r in self.options.reverse]

        combs = generate_combinations(opt_angles, opt_mirror, opt_bands, opt_rev)

        for x in combs:
            #  strips reverse bands from single channels
            if isinstance(x[2][1], str):
                if len(x[2][1]) < 2 and x[3][1] is not None:
                    x = x[:-1]
            yield x

    async def start_manager(self) -> None:
        """Starts the manager"""
        tasks, queue = [], asyncio.Queue()
        pdir = self.output_dir
        async for path in self.load_file_paths():
            await self.create_backup(path)
            async for combi in self.parse_transformation_ruleset():
                if self.options.multipath:
                    self.output_dir = self.output_dir / path.name.split('.')[0]
                    if not self.output_dir.exists():
                        self.output_dir.mkdir()
                image = await AsyncFileFactory.create_from_path(path)
                image.out_dir = self.output_dir
                image.task_queue = queue
                tasks.append(image.worker(combi))
                self.output_dir = pdir
        await asyncio.gather(*tasks)


async def main(ev_loop: AbstractEventLoop):
    parser = ArgumentParser(prog="stegomgeez", description="Steganography toolset", )
    pos = parser.add_argument_group(title='Positional arguments:')
    pos.add_argument(
        "filepath",
        help="One or multiple image files",
        nargs="+", default=[],
        action="store", type=Path,
    )
    pos.add_argument(
        "projectname",
        help="Name of the project.",
        action="store", type=str,
    )
    mut = parser.add_argument_group(
        title="Image transforms:",
        description='Create different transformations of an image',
    )
    mut.add_argument(
        "--angles", "-a",
        help="List of angles to apply to the image. Everything between 0° and 360°",
        default=[], nargs="+",
        required=False, type=int
    )
    mut.add_argument(
        "--flip", "-f",
        help="Mirrors the picture horizontally/vertically.",
        default=[], type=str,
        choices=("x", "y", "xy", "all"),
        nargs='+', required=False,
    )
    mut.add_argument(
        "--bands",
        help="Bands to extract from the image.",
        default=[], required=False,
        nargs='+'
    )
    mut.add_argument(
        "--reverse",
        help="Reverse the order of the bands.",
        action='store_true',
        required=False,
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
        required=False, type=Path
    )
    parser.add_argument(
        "--force",
        help="Overwrite existing files",
        action='store_true',
        required=False,
    )
    parser.add_argument(
        "--depth",
        help="Scan depth of files",
        default=5, type=int,
        required=False,
    )
    parser.add_argument(
        "--debug",
        help="Debug mode",
        default=False, action='store_true',
        required=False,
    )

    args = parser.parse_args()

    args.angles = [0] + args.angles

    args.filepath = [f.absolute() for f in args.filepath]

    if args.flip:
        args.flip = [None] + args.flip

    if 'all' in args.flip:
        args.flip = args.flip[:-1]
        args.flip += ["x", "y", "xy"]

    if args.bands:
        args.bands = [None] + args.bands

    if args.bands == ["all"]:
        args.bands = RGBA_VALS

    args.reverse = [None]

    if args.reverse:
        args.reverse += [True]

    if len(args.filepath) > 1:
        args.multipath = True
    else:
        args.multipath = False

    if args.debug:
        ev_loop.set_debug(True)
        args.outdir = Path(getcwd()).absolute() / '..' / "TESTOUT"

        if not args.outdir.exists():
            args.outdir.mkdir()

    console.print(args)

    async with Manager(
        project_name=args.projectname,
        file_paths=args.filepath,
        options=args, _loop=ev_loop,
    ) as mngr:
        await mngr.start_manager()


if __name__ == '__main__':
    try:
        asyncio.set_event_loop_policy(EventLoopPolicy())
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main(loop))
    except KeyboardInterrupt:
        pass
