from typing import Literal, Union, List, Tuple

import cv2
from PIL import Image
from cupy import ndarray
from cupy.array_api import uint8

from utils import (
    str2bin_str,
    bin_str2str,
    bit2binstr,
    pil2opencv,
)

PilImageType = Image.Image
ImageOrBytes = Union[Image.Image, bytes]
PilOrCvImage = Union[Image.Image, ndarray]
LsbOrMsbType = Literal['lsb', 'msb']
BitMatrix = List[Tuple[int, int, int]]
ScanDirectionType = Literal['row', 'col']

@njit
def lsb_encode(
        image: PilOrCvImage,
        message: str,
        bitmatrix: BitMatrix,
        bit_type: LsbOrMsbType = 'lsb',
        scan_direction: ScanDirectionType = 'row',
        delimiter: str = '#####',
) -> PilImageType:
    """
    Encode a message into an image using a bit matrix and a bit type.
    """

    # Load image
    if isinstance(image, PilImageType):
        image = pil2opencv(image)

    # Convert image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.astype(uint8)

    # Convert message to binary with delimiter
    message += delimiter
    binary_message = str2bin_str(message)
    bit_index = 0

    @njit
    def set_bits(value, bitmask):
        nonlocal bit_index
        for i in range(8):
            if bitmask & (1 << i):
                if bit_index < len(binary_message):
                    if bit_type == 'lsb':
                        value = (value & ~(1 << i)) | (int(binary_message[bit_index]) << i)
                    elif bit_type == 'msb':
                        value = (value & ~(1 << (7 - i))) | (int(binary_message[bit_index]) << (7 - i))
                    bit_index += 1
        return value

    # Process each pixel and modify bits
    rows, cols, channels = pixels.shape
    for x in range(rows if scan_direction == 'col' else cols):
        for y in range(cols if scan_direction == 'row' else rows):
            if bit_index >= len(binary_message):
                break

            # Extract the current color components
            r, g, b = pixels[x, y][:3]
            a = pixels[x, y][3] if channels == 4 else 0

            # Get the bit matrix for the current pixel
            bitmatrix_r, bitmatrix_g, bitmatrix_b, *bitmatrix_a = bitmatrix
            bitmatrix_a = bitmatrix_a[0] if bitmatrix_a else 0b00000000

            # Modify bits according to bit matrix
            r = set_bits(r, bitmatrix_r)
            g = set_bits(g, bitmatrix_g)
            b = set_bits(b, bitmatrix_b)
            if channels == 4:
                a = set_bits(a, bitmatrix_a)
                pixels[x, y] = (r, g, b, a)
            else:
                pixels[x, y] = (r, g, b)

    # Convert image back to BGR for saving
    return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)


@njit
def lsb_decode(
        image: PilOrCvImage,
        bitmatrix: BitMatrix,
        bit_type: LsbOrMsbType = 'lsb',
        scan_direction: ScanDirectionType = 'row',
        delimiter: str = 'DCXXX',
) -> str:
    """Decode a message from an image using a bit matrix and a bit type."""

    # Load image
    if isinstance(image, PilImageType):
        image = pil2opencv(image)

    # Convert image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.astype(uint8)

    bit_values = []

    @njit
    def extract_bits(value, bitmask, bt=bit_type):
        bits = []
        for i in range(8):
            if bitmask & (1 << i):
                if bt == 'lsb':
                    bits.append((value >> i) & 1)
                elif bt == 'msb':
                    bits.append((value >> (7 - i)) & 1)
        return bits

    # Extract bits from pixels
    rows, cols, channels = pixels.shape
    for x in range(rows if scan_direction == 'col' else cols):
        for y in range(cols if scan_direction == 'row' else rows):
            if channels == 4:
                r, g, b, a = pixels[x, y]
            else:
                r, g, b = pixels[x, y]
                a = 0

            # Get the bit matrix for the current pixel
            bitmatrix_r, bitmatrix_g, bitmatrix_b, *bitmatrix_a = bitmatrix
            bitmatrix_a = bitmatrix_a[0] if bitmatrix_a else 0b00000000

            bit_values.extend(extract_bits(r, bitmatrix_r))
            bit_values.extend(extract_bits(g, bitmatrix_g))
            bit_values.extend(extract_bits(b, bitmatrix_b))
            if channels == 4:
                bit_values.extend(extract_bits(a, bitmatrix_a))

    # Convert bits to binary string
    binary_message = bit2binstr(bit_values)

    # Convert binary message to string
    message = bin_str2str(binary_message)

    # Find the delimiter
    delimiter_idx = message.find(delimiter)
    if delimiter_idx != -1:
        message = message[:delimiter_idx]

    return message
