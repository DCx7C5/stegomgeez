from typing import List


qwerty_rows = [
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm"
]


def keyboard_shift(text: str, shift: int) -> str:
    char_map = {}
    for row in qwerty_rows:
        for i, char in enumerate(row):
            shifted_char = row[(i + shift) % len(row)]
            char_map[char] = shifted_char
            char_map[char.upper()] = shifted_char.upper()

    return ''.join(char_map.get(char, char) for char in text)


def bruteforce(text: str) -> List[str]:
    texts = []
    max_shift = max(len(row) for row in qwerty_rows)

    for shift in range(-max_shift + 1, max_shift):
        texts.append(keyboard_shift(text, shift))
    return texts


decrypt = keyboard_shift
encrypt = keyboard_shift
