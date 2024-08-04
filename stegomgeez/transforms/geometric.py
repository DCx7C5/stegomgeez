from typing import Literal

from PIL import Image
from PIL.Image import Transpose


FLIP_HOR = Transpose.FLIP_TOP_BOTTOM
FLIP_VER = Transpose.FLIP_LEFT_RIGHT


PilImageType = Image.Image


def change_angle(image: PilImageType, angle: int) -> PilImageType:
    if not (-359 < angle < 359):
        raise ValueError("angle must be between -359 and 359")
    if angle == 0:
        return image
    return image.rotate(angle)


def flip_image(image: PilImageType, direction: Literal['ver', 'hor']) -> PilImageType:
    return image.transpose(FLIP_HOR if direction == 'hor' else FLIP_VER)
