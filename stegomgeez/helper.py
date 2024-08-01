from asyncio import Semaphore

from aiofiles import open
from pathlib import Path
from re import compile
from itertools import product
from typing import Tuple, List

from PIL import Image
from hashid import HashID
from numpy import array, ndarray
from sympy import primerange


filename_log_pattern = compile(r'^(\w{,255}\.\w{3,4}) |:')
binary_pattern = compile(r'\b[01]{8,}\b')
whitespace_pattern = compile(r'[ \t]{2,}')
capital_pattern = compile(r'[A-Z]{2,}')
punctuation_pattern = compile(r'[!?,;:]{2,}')
line_break_pattern = compile(r'\n{2,}')

hashid = HashID()

RGBA_VALS = [
    'R', 'G', 'B', 'A',
    'RG', 'RB', 'RA', 'GB',
    'GA', 'BA', 'RGB', 'RGA',
    'RBA', 'GBA', 'RGBA'
]


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


def pil2opencv(image: Image.Image) -> ndarray:
    open_cv_image = array(image)
    return open_cv_image[:, :, ::-1].copy()


def save_png(image: Image.Image, path: Path, compression_level=0, iformat='PNG') -> None:
    """Save a PNG file"""
    image.save(path, format=iformat, compress_level=compression_level)


def save_jpg(image: Image.Image, path: Path, quality=100, iformat='JPEG', keep_rgb=True) -> None:
    """Save a JPG file"""
    image.save(path, format=iformat, quality=quality, keep_rgb=keep_rgb)


def save_image(image: Image.Image, path: Path, convert=False, keep_rgb=True, quality=95) -> None:
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
    async with open(path, 'wb') as f:
        await f.write(data)


async def load_data(path: Path) -> bytes:
    """Load bytes asynchronously from a file"""
    async with open(path, 'rb') as f:
        return await f.read()


async def save_text(text: str, path: Path) -> None:
    """Write text asynchronously to a file"""
    async with open(path, 'w') as f:
        await f.write(text)


async def load_text(path: Path) -> str:
    """Load text asynchronously from a file"""
    async with open(path, 'r') as f:
        return await f.read()


def save_image_to_disk(image: Image.Image, path: Path, convert=False, keep_rgb=True, quality=95):
    save_image(image, path, convert=convert, keep_rgb=keep_rgb, quality=quality)


async def save_to_disk(
        obj: str | bytes | Image.Image,
        path: Path, convert=False,
        keep_rgb=True,
        quality=95,
        loop=None
) -> None:
    if isinstance(obj, Image.Image):
        loop.run_in_executor(None, save_image, obj, path, convert, keep_rgb, quality)
        # save_image(obj, path, convert=convert, keep_rgb=keep_rgb, quality=quality)
    elif isinstance(obj, bytes):
        await save_data(obj, path)
    elif isinstance(obj, str):
        await save_text(obj, path)


log_sem = Semaphore(1)  # semaphore for log


async def write_to_logfile(line: str, path: Path) -> None:
    await log_sem.acquire()
    async with open(path, "r+") as f:
        pass

    log_sem.release()
