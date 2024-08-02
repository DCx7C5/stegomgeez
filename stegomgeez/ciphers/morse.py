

MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..',
    '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..',
    '9': '----.', '0': '-----', ', ': '--..--', '.': '.-.-.-', '?': '..--..',
    '/': '-..-.', '-': '-....-', '(': '-.--.', ')': '-.--.-', ' ': '/'
}


def encrypt(text: str, short='.', long='-', sep=' '):
    morse_dict = MORSE_CODE_DICT
    if short != '.':
        morse_dict = {k: v.replace('.', short) for k, v in morse_dict.items()}
    if long != '-':
        morse_dict = {k: v.replace('-', long) for k, v in morse_dict.items()}
    return sep.join(morse_dict[char] for char in text.upper())


def decrypt(text: str, short=".", long="-", char_sep=' '):
    morse_dict = {v: k for k, v in MORSE_CODE_DICT.items()}
    if short != '.':
        morse_dict = {k.replace('.', short): v for k, v in morse_dict.items()}
    if long != '-':
        morse_dict = {k.replace('-', long): v for k, v in morse_dict.items()}
    if char_sep == '':
        raise ValueError("Too many possibilities without character sep")
    else:
        return ''.join(morse_dict[char] for char in text.split(char_sep))
