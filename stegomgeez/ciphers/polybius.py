

def generate_square(key):
    key = key.upper().replace('J', 'I')
    matrix = []
    for char in key:
        if char not in matrix and char.isalpha():
            matrix.append(char)
    alphabet = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    for char in alphabet:
        if char not in matrix:
            matrix.append(char)
    return [matrix[i:i+5] for i in range(0, 25, 5)]


def encrypt(text: str, key):
    matrix = generate_square(key)
    position = {char: (r, c) for r, row in enumerate(matrix) for c, char in enumerate(row)}
    result = ''
    for char in text.upper():
        if char in position:
            r, c = position[char]
            result += str(r + 1) + str(c + 1)
        else:
            result += char
    return result


def decrypt(text: str, key):
    matrix = generate_square(key)
    result = ''
    i = 0
    while i < len(text):
        if text[i].isdigit() and text[i + 1].isdigit():
            r = int(text[i]) - 1
            c = int(text[i + 1]) - 1
            result += matrix[r][c]
            i += 2
        else:
            result += text[i]
            i += 1
    return result
