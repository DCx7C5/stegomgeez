def get_position(char, matrix):
    for row in range(5):
        for col in range(5):
            if matrix[row][col] == char:
                return row, col
    return None


def generate_matrix(key):
    matrix = []
    for char in key.upper():
        if char not in matrix and char != 'J':
            matrix.append(char)
    alphabet = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    for char in alphabet:
        if char not in matrix:
            matrix.append(char)
    return [matrix[i:i+5] for i in range(0, 25, 5)]


def encrypt(text, key):
    text = text.upper().replace('J', 'I').replace(' ', '')
    matrix = generate_matrix(key)
    pairs = [(text[i], text[i+1]) for i in range(0, len(text), 2)]
    if len(pairs[-1]) == 1:
        pairs[-1] = (pairs[-1][0], 'X')

    result = ""
    for a, b in pairs:
        row_a, col_a = get_position(a, matrix)
        row_b, col_b = get_position(b, matrix)
        if row_a == row_b:
            result += matrix[row_a][(col_a + 1) % 5]
            result += matrix[row_b][(col_b + 1) % 5]
        elif col_a == col_b:
            result += matrix[(row_a + 1) % 5][col_a]
            result += matrix[(row_b + 1) % 5][col_b]
        else:
            result += matrix[row_a][col_b]
            result += matrix[row_b][col_a]
    return result


def decrypt(text, key):
    matrix = generate_matrix(key)

    result = ""
    for i in range(0, len(text), 2):
        a, b = text[i], text[i+1]
        row_a, col_a = get_position(a, matrix)
        row_b, col_b = get_position(b, matrix)
        if row_a == row_b:
            result += matrix[row_a][(col_a - 1) % 5]
            result += matrix[row_b][(col_b - 1) % 5]
        elif col_a == col_b:
            result += matrix[(row_a - 1) % 5][col_a]
            result += matrix[(row_b - 1) % 5][col_b]
        else:
            result += matrix[row_a][col_b]
            result += matrix[row_b][col_a]
    return result
