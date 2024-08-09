from collections import Counter
from utils import (
    binary_pattern,
    whitespace_pattern,
    capital_pattern,
    punctuation_pattern,
    line_break_pattern,
    ZERO_WIDTH_CHARS,
)


def find_zero_width_chars(text):
    """Detects zero-width Unicode characters in the provided text."""
    findings = []
    for i, char in enumerate(text):
        if char in ZERO_WIDTH_CHARS:
            findings.append((i, char))
    return findings


def find_unusual_unicode(text):
    """Detects unusual Unicode characters in the provided text."""
    findings = []
    for i, char in enumerate(text):
        if ord(char) > 127 and char not in ZERO_WIDTH_CHARS:
            findings.append((i, char))
    return findings


def find_repeated_phrases(text):
    """Finds repeated words or phrases in the provided text."""
    findings = []
    words = text.split()
    seen = {}
    for i, word in enumerate(words):
        if word in seen:
            findings.append((seen[word], i, word))
        seen[word] = i
    return findings


def find_binary_data(text):
    """Detects binary data in the provided text."""
    findings = []
    matches = binary_pattern.findall(text)
    if matches:
        findings.extend(matches)
    return findings


def find_whitespace_patterns(text):
    """Finds whitespace patterns in the provided text that could be used for encoding."""
    findings = []
    matches = whitespace_pattern.findall(text)
    if matches:
        findings.extend(matches)
    return matches


def find_capitalization_patterns(text):
    """Finds unusual capitalization patterns in the provided text."""
    findings = []
    matches = capital_pattern.findall(text)
    if matches:
        findings.extend(matches)
    return findings


def find_homoglyphs(text):
    """Detects homoglyphs in the provided text."""
    findings = []
    homoglyphs = {'а': 'a', 'е': 'e', 'о': 'o', 'і': 'i', 'с': 'c'}  # Cyrillic homoglyphs
    for i, char in enumerate(text):
        if char in homoglyphs:
            findings.append((i, char))
    return findings


def find_out_of_place_punctuation(text):
    """Finds out-of-place punctuation in the provided text."""
    findings = []
    matches = punctuation_pattern.findall(text)
    if matches:
        findings.extend(matches)
    return findings


def find_unexpected_line_breaks(text):
    """Finds unexpected line breaks in the provided text."""
    findings = []
    matches = line_break_pattern.findall(text)
    if matches:
        findings.extend(matches)
    return findings


def character_frequency_analysis(text):
    """Performs character frequency analysis on the provided text."""
    counter = Counter(text)
    return counter.most_common()
