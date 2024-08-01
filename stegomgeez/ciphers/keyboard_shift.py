from typing import Literal


def keyboard_shift(text, shift, qwerty_rows):
    # Create a mapping for each character to its shifted counterpart
    char_map = {}
    for row in qwerty_rows:
        for i, char in enumerate(row):
            shifted_char = row[(i + shift) % len(row)]
            char_map[char] = shifted_char
            char_map[char.upper()] = shifted_char.upper()

    # Apply the shift to the input text
    shifted_text = ''.join(char_map.get(char, char) for char in text)

    return shifted_text


def bruteforce(text, layout=Literal["us", "de"]):
    qwerty_rows = [
        f"qwert{'z' if layout == 'de' else 'y'}uiop",
        "asdfghjkl",
        f"{'y' if layout == 'de' else 'z'}xcvbnm"
    ]

    max_shift = max(len(row) for row in qwerty_rows)

    for shift in range(-max_shift + 1, max_shift):
        shifted_text = keyboard_shift(text, shift, qwerty_rows)
        direction = "right" if shift > 0 else "left" if shift < 0 else "no shift"
        print(f"Shift {shift} ({direction}): {shifted_text}")


decrypt = keyboard_shift
encrypt = keyboard_shift
