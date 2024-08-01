from itertools import combinations
from PIL import Image


def new_channel(size):
    return Image.new('L', size)


def merge_channels(mode, *channels):
    return Image.merge(mode, channels)


def create_combinations(channels, size):
    combinations_dict = {}
    keys = channels.keys()
    for length in range(1, len(keys) + 1):
        for comb in combinations(keys, length):
            comb_str = ''.join(comb)
            comb_list = [channels[ch] if ch in comb else new_channel(size) for ch in 'RGBA']
            mode = 'RGB' if 'A' not in comb else 'RGBA'
            comb_image = merge_channels(mode, *comb_list[:len(mode)])
            combinations_dict[comb_str] = comb_image
    return combinations_dict


def get_band(img: Image.Image, bands: str):
    band_to_int = {'R': 0, 'G': 1, 'B': 2, 'A': 3}
    channels = {band: img.getchannel(band_to_int[band]) for band in bands}
    combinations_dict = create_combinations(channels, img.size)
    return combinations_dict[bands]
