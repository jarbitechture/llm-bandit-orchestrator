"""Tests for src/eval_rpn.py — eval_rpn(tokens) -> int."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval_rpn import eval_rpn


class TestBasicArithmetic:
    def test_addition(self):
        assert eval_rpn(["2", "3", "+"]) == 5

    def test_subtraction(self):
        assert eval_rpn(["5", "3", "-"]) == 2

    def test_multiplication(self):
        assert eval_rpn(["4", "3", "*"]) == 12

    def test_division(self):
        assert eval_rpn(["10", "3", "/"]) == 3

    def test_single_number(self):
        assert eval_rpn(["42"]) == 42


class TestTruncateTowardZero:
    """Division must truncate toward zero, not floor."""

    def test_negative_division_truncates(self):
        # -7 / 2 = -3 (truncate), not -4 (floor)
        assert eval_rpn(["7", "-2", "*", "2", "/"]) == -7

    def test_negative_numerator(self):
        assert eval_rpn(["-7", "2", "/"]) == -3

    def test_negative_denominator(self):
        assert eval_rpn(["7", "-2", "/"]) == -3

    def test_both_negative(self):
        assert eval_rpn(["-7", "-2", "/"]) == 3

    def test_truncation_6_minus4(self):
        assert eval_rpn(["6", "-4", "/"]) == -1


class TestComplexExpressions:
    def test_classic_example(self):
        # (2 + 1) * 3 = 9
        assert eval_rpn(["2", "1", "+", "3", "*"]) == 9

    def test_nested(self):
        # ((4 + 13) / 5) = 3
        assert eval_rpn(["4", "13", "5", "/", "+"]) == 6

    def test_leetcode_example(self):
        # ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
        tokens = ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
        assert eval_rpn(tokens) == 22

    def test_chained_operations(self):
        # 5 1 2 + 4 * + 3 - = 5 + (1+2)*4 - 3 = 14
        assert eval_rpn(["5", "1", "2", "+", "4", "*", "+", "3", "-"]) == 14


class TestNegativeNumbers:
    def test_negative_operand(self):
        assert eval_rpn(["-3", "4", "+"]) == 1

    def test_both_negative_add(self):
        assert eval_rpn(["-3", "-4", "+"]) == -7

    def test_zero(self):
        assert eval_rpn(["0"]) == 0


class TestErrors:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            eval_rpn([])

    def test_insufficient_operands(self):
        with pytest.raises(ValueError):
            eval_rpn(["+"])

    def test_unknown_operator(self):
        with pytest.raises(ValueError):
            eval_rpn(["1", "2", "%"])

    def test_leftover_operands(self):
        with pytest.raises(ValueError):
            eval_rpn(["1", "2", "3", "+"])
