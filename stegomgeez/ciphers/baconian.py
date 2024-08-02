baconian_alphabet = {
    'A': 'AAAAA', 'B': 'AAAAB', 'C': 'AAABA', 'D': 'AAABB',
    'E': 'AABAA', 'F': 'AABAB', 'G': 'AABBA', 'H': 'AABBB',
    'I': 'ABAAA', 'J': 'ABAAB', 'K': 'ABABA', 'L': 'ABABB',
    'M': 'ABBAA', 'N': 'ABBAB', 'O': 'ABBBA', 'P': 'ABBBB',
    'Q': 'BAAAA', 'R': 'BAAAB', 'S': 'BAABA', 'T': 'BAABB',
    'U': 'BABAA', 'V': 'BABAB', 'W': 'BABBA', 'X': 'BABBB',
    'Y': 'BBAAA', 'Z': 'BBAAB'
}


def encrypt(text: str) -> str:

    text = text.upper().replace('J', 'I').replace('U', 'V')
    result = ''
    for char in text:
        if char in baconian_alphabet:
            result += baconian_alphabet[char] + ' '
        else:
            result += char
    return result.strip()


def decrypt(text: str) -> str:
    text = text.replace(' ', '')
    result = ''
    for i in range(0, len(text), 5):
        segment = text[i:i+5]
        if segment in baconian_alphabet:
            result += baconian_alphabet[segment]
        else:
            result += '?'
    return result
