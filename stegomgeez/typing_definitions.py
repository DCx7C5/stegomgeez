from pathlib import Path
from typing import (
    AsyncIterator,
    LiteralString,
    Sequence,
    Optional,
    Callable,
    Literal,
    Tuple,
    Union,
    Dict,
    List,
    Any,
)
from PIL import Image
from numpy import ndarray

RGBA_VALS = [
    'R', 'G', 'B', 'A',
    'RG', 'RB', 'RA', 'GB',
    'GA', 'BA', 'RGB', 'RGA',
    'RBA', 'GBA', 'RGBA'
]

PilImage = Image.Image
ImageOrBytes = Union[PilImage, bytes]
PilOrCvImage = Union[PilImage, ndarray]
AnyDataContent = Union[str, bytes, PilImage, ImageOrBytes]

ChangeAngleFunc = Callable[[PilImage, int], PilImage]
FlipImageFunc = Callable[[PilImage, Literal['ver', 'hor']], PilImage]
GetBandFunc = Callable[[PilImage, str], PilImage]
FunctionType = Union[ChangeAngleFunc, FlipImageFunc, GetBandFunc]

Pathslist = Tuple[PilImage, Path, str]

CombinationItem = Tuple[FunctionType, Union[str, int]]
Combination = List[CombinationItem]
CombinationList = List[Combination]

HistogramType = Literal['color', 'grayscale']
BitMatrix = List[Tuple[int, int, int]]

BandCombinations = Dict[LiteralString | str, Image]
