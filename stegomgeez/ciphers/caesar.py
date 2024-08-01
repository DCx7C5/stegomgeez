def encrypt(text, shift):
    result = ""
    for char in text:
        if char.isalpha():
            shift_base = ord('A') if char.isupper() else ord('a')
            result += chr((ord(char) - shift_base + shift) % 26 + shift_base)
        else:
            result += char
    return result


def decrypt(text, shift):
    return encrypt(text, -shift)


def bruteforce(text):
    lines = []
    for n in range(26):
        ld = decrypt(text, n)
        lines.append(ld)
    return lines
