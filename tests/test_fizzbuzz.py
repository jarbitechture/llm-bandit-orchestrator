"""Tests for src/fizzbuzz.py — fizzbuzz(n) -> list[str]."""

import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fizzbuzz import fizzbuzz


class TestFizzbuzzReturnType:
    def test_returns_list(self):
        assert isinstance(fizzbuzz(1), list)

    def test_returns_strings(self):
        result = fizzbuzz(3)
        assert all(isinstance(item, str) for item in result)


class TestFizzbuzzLength:
    def test_length_1(self):
        assert len(fizzbuzz(1)) == 1

    def test_length_15(self):
        assert len(fizzbuzz(15)) == 15

    def test_length_0(self):
        assert len(fizzbuzz(0)) == 0


class TestFizzbuzzValues:
    def test_plain_number(self):
        result = fizzbuzz(2)
        assert result[0] == "1"
        assert result[1] == "2"

    def test_fizz_at_3(self):
        assert fizzbuzz(3)[2] == "Fizz"

    def test_buzz_at_5(self):
        assert fizzbuzz(5)[4] == "Buzz"

    def test_fizzbuzz_at_15(self):
        assert fizzbuzz(15)[14] == "FizzBuzz"

    def test_fizz_at_6(self):
        assert fizzbuzz(6)[5] == "Fizz"

    def test_buzz_at_10(self):
        assert fizzbuzz(10)[9] == "Buzz"

    def test_fizzbuzz_at_30(self):
        assert fizzbuzz(30)[29] == "FizzBuzz"


class TestFizzbuzzFullSequence:
    def test_first_15(self):
        expected = [
            "1", "2", "Fizz", "4", "Buzz",
            "Fizz", "7", "8", "Fizz", "Buzz",
            "11", "Fizz", "13", "14", "FizzBuzz",
        ]
        assert fizzbuzz(15) == expected
