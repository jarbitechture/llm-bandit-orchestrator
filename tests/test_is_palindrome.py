"""Tests for src/palindrome.py — is_palindrome(text) -> bool."""

import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from palindrome import is_palindrome


class TestSimplePalindromes:
    def test_single_char(self):
        assert is_palindrome("a") is True

    def test_two_same(self):
        assert is_palindrome("aa") is True

    def test_racecar(self):
        assert is_palindrome("racecar") is True

    def test_madam(self):
        assert is_palindrome("madam") is True

    def test_level(self):
        assert is_palindrome("level") is True


class TestNonPalindromes:
    def test_hello(self):
        assert is_palindrome("hello") is False

    def test_python(self):
        assert is_palindrome("python") is False

    def test_ab(self):
        assert is_palindrome("ab") is False


class TestCaseInsensitive:
    def test_mixed_case(self):
        assert is_palindrome("Racecar") is True

    def test_all_caps(self):
        assert is_palindrome("MADAM") is True

    def test_alternating(self):
        assert is_palindrome("RaCeCaR") is True


class TestIgnoresNonAlphanumeric:
    def test_with_spaces(self):
        assert is_palindrome("nurses run") is True

    def test_with_punctuation(self):
        assert is_palindrome("A man, a plan, a canal: Panama") is True

    def test_with_mixed_symbols(self):
        assert is_palindrome("Was it a car or a cat I saw?") is True

    def test_numeric_palindrome(self):
        assert is_palindrome("12321") is True

    def test_numeric_non_palindrome(self):
        assert is_palindrome("12345") is False


class TestEdgeCases:
    def test_empty_string(self):
        assert is_palindrome("") is True

    def test_only_symbols(self):
        assert is_palindrome("!@#$%") is True

    def test_single_space(self):
        assert is_palindrome(" ") is True
