"""Tests for src/matrix_spiral.py — spiral_order(matrix) -> list[int]."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from matrix_spiral import spiral_order


class TestBasicSpiral:
    def test_3x3(self):
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert spiral_order(matrix) == [1, 2, 3, 6, 9, 8, 7, 4, 5]

    def test_1x1(self):
        assert spiral_order([[42]]) == [42]

    def test_2x2(self):
        matrix = [[1, 2], [3, 4]]
        assert spiral_order(matrix) == [1, 2, 4, 3]


class TestNonSquare:
    def test_3x4(self):
        matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        assert spiral_order(matrix) == [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]

    def test_4x3(self):
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        assert spiral_order(matrix) == [1, 2, 3, 6, 9, 12, 11, 10, 7, 4, 5, 8]

    def test_1x4_single_row(self):
        assert spiral_order([[1, 2, 3, 4]]) == [1, 2, 3, 4]

    def test_4x1_single_column(self):
        assert spiral_order([[1], [2], [3], [4]]) == [1, 2, 3, 4]

    def test_2x4(self):
        matrix = [[1, 2, 3, 4], [5, 6, 7, 8]]
        assert spiral_order(matrix) == [1, 2, 3, 4, 8, 7, 6, 5]


class TestEdgeCases:
    def test_empty(self):
        assert spiral_order([]) == []

    def test_empty_rows(self):
        assert spiral_order([[]]) == []

    def test_5x1(self):
        assert spiral_order([[i] for i in range(1, 6)]) == [1, 2, 3, 4, 5]

    def test_1x5(self):
        assert spiral_order([[1, 2, 3, 4, 5]]) == [1, 2, 3, 4, 5]

    def test_4x4(self):
        matrix = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]
        assert spiral_order(matrix) == [
            1, 2, 3, 4, 8, 12, 16, 15, 14, 13, 9, 5, 6, 7, 11, 10
        ]
