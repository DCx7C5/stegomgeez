import numpy as np

# key_matrix = np.array([[3, 3], [2, 5]])


def encrypt(text, key_matrix):
    n = len(key_matrix)
    text = text.upper().replace(' ', '')
    if len(text) % n != 0:
        text += 'X' * (n - len(text) % n)

    result = ''
    for i in range(0, len(text), n):
        block = np.array([ord(char) - ord('A') for char in text[i:i+n]])
        encrypted_block = np.dot(key_matrix, block) % 26
        result += ''.join(chr(num + ord('A')) for num in encrypted_block)
    return result


def decrypt(text, key_matrix):
    n = len(key_matrix)
    inverse_matrix = np.linalg.inv(key_matrix).astype(int) % 26
    result = ''
    for i in range(0, len(text), n):
        block = np.array([ord(char) - ord('A') for char in text[i:i+n]])
        decrypted_block = np.dot(inverse_matrix, block) % 26
        result += ''.join(chr(int(num) + ord('A')) for num in decrypted_block)
    return result
