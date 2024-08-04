def encrypt(text: str, key: str) -> str:
    key = key.upper()
    key_len = len(key)
    result = ""
    for i, char in enumerate(text):
        if char.isalpha():
            shift_base = ord('A') if char.isupper() else ord('a')
            result += chr((ord(char) - shift_base + ord(key[i % key_len]) - ord('A')) % 26 + shift_base)
        else:
            result += char
    return result


def decrypt(text: str, key: str) -> str:
    key = key.upper()
    key_len = len(key)
    result = ""
    for i, char in enumerate(text):
        if char.isalpha():
            shift_base = ord('A') if char.isupper() else ord('a')
            result += chr((ord(char) - shift_base - (ord(key[i % key_len]) - ord('A'))) % 26 + shift_base)
        else:
            result += char
    return result
