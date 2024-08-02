from PIL.Image import Transpose

from typing_definitions import PilImage, Literal

FLIP_HOR = Transpose.FLIP_TOP_BOTTOM
FLIP_VER = Transpose.FLIP_LEFT_RIGHT


def change_angle(image: PilImage, angle: int) -> PilImage:
    if not (-359 < angle < 359):
        raise ValueError("angle must be between -359 and 359")
    if angle == 0:
        return image
    return image.rotate(angle)


def flip_image(image: PilImage, direction: Literal['ver', 'hor']) -> PilImage:
    return image.transpose(FLIP_HOR if direction == 'hor' else FLIP_VER)
