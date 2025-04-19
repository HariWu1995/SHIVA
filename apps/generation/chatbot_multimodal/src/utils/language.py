
import re


# Convert string to integer
def atoi(text: str):
    return int(text) if text.isdigit() else text.lower()


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


# Replace multiple string pairs in a string
def replace_all(text, replacements):
    for i, j in replacements.items():
        text = text.replace(i, j)
    return text

