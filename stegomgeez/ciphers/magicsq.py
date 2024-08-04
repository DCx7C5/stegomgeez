from typing import List

MagicSqType = List[List[int]]


def encrypt(text: str, square: MagicSqType) -> str:
    size = len(square)
    if len(text) != size * size:
        raise ValueError("Text length must be equal to the size of the magic square.")
    text = text.upper()
    cipher = [''] * (size * size)
    for i in range(size):
        for j in range(size):
            index = square[i][j] - 1
            cipher[index] = text[i * size + j]
    return ''.join(cipher)


def decrypt(cipher: str, square: MagicSqType) -> str:
    size = len(square)
    if len(cipher) != size * size:
        raise ValueError("Cipher length must be equal to the size of the magic square.")
    cipher = cipher.upper()
    text = [''] * (size * size)
    for i in range(size):
        for j in range(size):
            index = square[i][j] - 1
            text[i * size + j] = cipher[index]
    return ''.join(text)
