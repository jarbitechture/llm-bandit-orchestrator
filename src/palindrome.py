import re

def is_palindrome(text: str) -> bool:
    # Normalize the text: lower case and remove non-alphanumeric characters
    normalized_text = re.sub(r'[^a-z0-9]', '', text.lower())
    # Check if the normalized text is equal to its reverse
    return normalized_text == normalized_text[::-1]