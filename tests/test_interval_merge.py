"""Tests for src/interval_merge.py — merge_intervals(intervals) -> list[list[int]]."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from interval_merge import merge_intervals


class TestBasicMerge:
    def test_overlapping(self):
        assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [
            [1, 6], [8, 10], [15, 18]
        ]

    def test_touching(self):
        assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]

    def test_no_overlap(self):
        assert merge_intervals([[1, 2], [5, 6]]) == [[1, 2], [5, 6]]

    def test_complete_overlap(self):
        assert merge_intervals([[1, 10], [2, 5], [3, 7]]) == [[1, 10]]


class TestUnsortedInput:
    def test_reverse_order(self):
        assert merge_intervals([[8, 10], [1, 3], [2, 6]]) == [[1, 6], [8, 10]]

    def test_random_order(self):
        assert merge_intervals([[5, 8], [1, 3], [2, 4], [9, 10]]) == [
            [1, 4], [5, 8], [9, 10]
        ]


class TestEdgeCases:
    def test_empty(self):
        assert merge_intervals([]) == []

    def test_single(self):
        assert merge_intervals([[1, 5]]) == [[1, 5]]

    def test_point_interval(self):
        assert merge_intervals([[2, 2]]) == [[2, 2]]

    def test_point_intervals_touching(self):
        assert merge_intervals([[2, 2], [2, 2]]) == [[2, 2]]

    def test_point_interval_adjacent(self):
        assert merge_intervals([[1, 2], [2, 2], [2, 3]]) == [[1, 3]]

    def test_all_same(self):
        assert merge_intervals([[1, 4], [1, 4], [1, 4]]) == [[1, 4]]

    def test_nested(self):
        assert merge_intervals([[1, 10], [2, 3], [4, 5], [6, 7]]) == [[1, 10]]


class TestNegativeValues:
    def test_negative_intervals(self):
        assert merge_intervals([[-5, -1], [-3, 2], [4, 6]]) == [[-5, 2], [4, 6]]

    def test_crossing_zero(self):
        assert merge_intervals([[-2, 0], [0, 3]]) == [[-2, 3]]
