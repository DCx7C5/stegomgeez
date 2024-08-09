from __future__ import annotations
from typing import Tuple, List, Literal, Union, Dict, Any
import asyncio
from asyncio import Semaphore
from pathlib import Path

from aiofiles import open as aopen
import cv2
from PIL import ExifTags, Image
from magic import (
    MAGIC_MIME_TYPE,
    Magic,
)
from pathlib import Path
from re import compile
from itertools import product
from hashid import HashID
from sympy import primerange
import matplotlib.pyplot as plt
from numpy import array, ndarray


PilImageType = Image.Image
AnyDataContent = Union[str, bytes, Image.Image, bytes]
ImgInfoType = Dict[str, str | None | int | Dict[str, Any]]

hashid = HashID()


filename_log_pattern = compile(r'^(\w{,255}\.\w{3,4}) |:')
binary_pattern = compile(r'\b[01]{8,}\b')
whitespace_pattern = compile(r'[ \t]{2,}')
capital_pattern = compile(r'[A-Z]{2,}')
punctuation_pattern = compile(r'[!?,;:]{2,}')
line_break_pattern = compile(r'\n{2,}')


ZERO_WIDTH_CHARS = [
    '\u200B',  # Zero Width Space
    '\u200C',  # Zero Width Non-Joiner
    '\u200D',  # Zero Width Joiner
    '\u200E',  # Left-to-Right Mark
    '\u200F',  # Right-to-Left Mark
    '\u202A',  # Left-to-Right Embedding
    '\u202B',  # Right-to-Left Embedding
    '\u202C',  # Pop Directional Formatting
    '\u202D',  # Left-to-Right Override
    '\u202E',  # Right-to-Left Override
    '\u2060',  # Word Joiner
    '\u2061',  # Function Application
    '\u2062',  # Invisible Times
    '\u2063',  # Invisible Separator
    '\u2064',  # Invisible Plus
    '\u2066',  # Left-to-Right Isolate
    '\u2067',  # Right-to-Left Isolate
    '\u2068',  # First Strong Isolate
    '\u2069',  # Pop Directional Isolate
    '\uFEFF'  # Zero Width No-Break Space
]

RGBA_VALS = [
    'R', 'G', 'B', 'A',
    'RG', 'RB', 'RA', 'GB',
    'GA', 'BA', 'RGB', 'RGA',
    'RBA', 'GBA', 'RGBA'
]

common_english_letters_freq = "ETAOINSHRDLCUMWFGYPBVKJXQZ"


def get_mime_from_file(fpath: Path) -> str:
    with Magic(flags=MAGIC_MIME_TYPE) as m:
        return m.id_filename(str(fpath))


def get_mime_from_buffer(data: bytes) -> str:
    with Magic(flags=MAGIC_MIME_TYPE) as m:
        return m.id_buffer(data)


def generate_combinations(*lists):
    """Cross combination of list values"""
    return list(product(*lists))


def get_prime_numbers_up_to(n):
    """Generate a list of prime numbers up to n."""
    return list(primerange(2, n))


def str2bin_str(text):
    """Converts string to binary string"""
    return ''.join(format(ord(x), '08b') for x in text)


def bin_str2str(binary_message) -> str:
    """Converts binary string to string"""
    message = ''
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i + 8]
        if len(byte) < 8:
            break
        message += chr(int(byte, 2))
    return message


def int2bin(n):
    """Simple integer to binary converter"""
    return "{0:08b}".format(n)


def bit2binstr(bit_values):
    return ''.join(str(bit) for bit in bit_values)


def convert_lsb_mask(mask):
    """Convert a mask to an 8-bit integer."""
    if isinstance(mask, str):
        if mask.startswith('0x'):
            return int(mask, 16)
        elif mask.startswith('0b'):
            return int(mask, 2)
        else:
            return int(mask, 10)
    return int(mask)


def text_fits_in_image(img_size: Tuple[int, int], text: bytes) -> bool:
    """Checks if text fits in image using LSB steganography."""
    w, h = img_size
    return len(text) <= w * h * 3


def identify_hash(text: str) -> List[str]:
    """Checks if string is a hash value"""
    possible_hashes = []
    hashes = hashid.identifyHash(text)
    if hashes:
        possible_hashes.extend(hashes)
    return possible_hashes


def pil2opencv(image: PilImageType) -> ndarray:
    open_cv_image = array(image)
    return open_cv_image[:, :, ::-1].copy()


def create_histogram(
    image: ndarray | PilImageType,
    htype: Literal['color', 'grayscale'],
    fpath: Path,
) -> None:
    if isinstance(image, PilImageType):
        image = pil2opencv(image)
    if htype == 'color':
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        plt.title('Color Histogram')
    elif htype == 'grayscale':
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist, color='black')
        plt.xlim([0, 256])
        plt.title('Grayscale Histogram')

    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(fpath)


def save_png(image: PilImageType, path: Path, compression_level=0, iformat='PNG') -> None:
    """Save a PNG file"""
    image.save(path, format=iformat, compress_level=compression_level)


def save_jpg(image: PilImageType, path: Path, quality=100, iformat='JPEG', keep_rgb=True) -> None:
    """Save a JPG file"""
    image.save(path, format=iformat, quality=quality, keep_rgb=keep_rgb)


def save_image(image: PilImageType, path: Path, convert=False, keep_rgb=True, quality=95) -> None:
    """Save an image file, wrapper function fpr save_jpg, save_png"""
    if convert and path.suffix in ['.png', '.PNG']:
        path = path.with_suffix('.jpg')
        save_jpg(image, path, keep_rgb=keep_rgb, quality=quality)
    elif not convert and path.suffix in ['.png', '.PNG']:
        save_png(image, path)
    elif convert and path.suffix in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
        path = path.with_suffix('.png')
        save_png(image, path)
    elif not convert and path.suffix in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
        save_jpg(image, path, keep_rgb=keep_rgb, quality=quality)


async def save_data(data: bytes, path: Path) -> None:
    """Write bytes asynchronously to a file"""
    async with aopen(path, 'wb') as f:
        await f.write(data)


async def load_data(path: Path) -> bytes:
    """Load bytes asynchronously from a file"""
    async with aopen(path, 'rb') as f:
        return await f.read()


async def save_text(text: str, path: Path) -> None:
    """Write text asynchronously to a file"""
    async with aopen(path, 'w') as f:
        await f.write(text)


async def load_text(path: Path) -> str:
    """Load text asynchronously from a file"""
    async with open(path, 'r') as f:
        return await f.read()


def save_image_to_disk(image: PilImageType, path: Path, convert=False, keep_rgb=True, quality=95):
    save_image(image, path, convert=convert, keep_rgb=keep_rgb, quality=quality)


async def save_to_disk(
        obj: AnyDataContent,
        path: Path, convert=False,
        keep_rgb=True,
        quality=95,
) -> None:
    if isinstance(obj, PilImageType):
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, save_image, obj, path, convert, keep_rgb, quality)
        # save_image(obj, path, convert=convert, keep_rgb=keep_rgb, quality=quality)
    elif isinstance(obj, bytes):
        await save_data(obj, path)
    elif isinstance(obj, str):
        await save_text(obj, path)


def get_image_info(image: PilImageType, fpath: Path) -> ImgInfoType:

    with image:
        # Image information
        img_info = {
            'File Type': image.format,
            'MIME Type': get_mime_from_file(fpath),
            'Image Width': image.width,
            'Image Height': image.height,
            'Bit Depth': image.info.get('bits', 'N/A'),
            'Color Type': image.mode,
            'Compression': image.info.get('compression', 'N/A'),
            'Filter': image.info.get('filter', 'N/A'),
            'Interlace': 'Interlaced' if image.info.get('interlace') else 'Noninterlaced',
            'Significant Bits': ' '.join([str(image.info.get('bitdepth', 'N/A'))] * len(image.mode)) if image.info.get(
                'bitdepth') else 'N/A',
            'Background Color': image.info.get('background', 'N/A'),
            'Exif Byte Order': image.info.get('exif_byte_order', 'N/A'),
            'Software': image.info.get('software', 'N/A'),
            'Image Size': f"{image.width}x{image.height}",
            'Megapixels': f"{(image.width * image.height) / 1_000_000:.3f}"
        }

        # Extract EXIF data
        exif_data = image.getexif()
        if exif_data:
            img_info['EXIF'] = {ExifTags.TAGS.get(tag, tag): value for tag, value in exif_data.items()}

    return img_info


log_sem = Semaphore(1)


async def write_bytes(path: Path, content: bytes) -> None:
    async with aopen(path, "wb") as f:
        await f.write(content)


async def write_text(path: Path, content: str) -> None:
    async with aopen(path, "w") as f:
        await f.write(content)
