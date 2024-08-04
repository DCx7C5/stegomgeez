from typing import List, Set

SatorSqType = List[List[str]]


SATOR_SQUARE = [
    ["S", "A", "T", "O", "R"],
    ["A", "R", "E", "P", "O"],
    ["T", "E", "N", "E", "T"],
    ["O", "P", "E", "R", "A"],
    ["R", "O", "T", "A", "S"]
]


def generate_flat_list(square: SatorSqType) -> List[str]:
    _flat_list = []
    for row in square:
        _flat_list.extend(row)
    return list(set(_flat_list))


def generate_maps(flat_list: List):
    _encrypt_map = {}
    _decrypt_map = {}

    for i, char in enumerate(flat_list):
        _encrypt_map[char] = flat_list[(i + 2) % len(flat_list)]
        _decrypt_map[flat_list[(i + 2) % len(flat_list)]] = char

    return _encrypt_map, _decrypt_map


flat_list = generate_flat_list(SATOR_SQUARE)
encrypt_map, decrypt_map = generate_maps(flat_list)


def encrypt(text: str) -> str:
    text = text.upper()
    encrypted_text = ""
    for char in text:
        if char in encrypt_map:
            encrypted_text += encrypt_map[char]
        else:
            encrypted_text += char  # Non-square characters remain unchanged
    return encrypted_text


def decrypt(cipher: str) -> str:
    cipher = cipher.upper()
    decrypted_text = ""
    for char in cipher:
        if char in decrypt_map:
            decrypted_text += decrypt_map[char]
        else:
            decrypted_text += char  # Non-square characters remain unchanged
    return decrypted_text
