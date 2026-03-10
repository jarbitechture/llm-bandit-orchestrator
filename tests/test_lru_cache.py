"""Tests for src/lru_cache.py — LRUCache class."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lru_cache import LRUCache


class TestBasicOperations:
    def test_put_and_get(self):
        cache = LRUCache(2)
        cache.put(1, 10)
        assert cache.get(1) == 10

    def test_miss_returns_neg1(self):
        cache = LRUCache(2)
        assert cache.get(99) == -1

    def test_update_existing(self):
        cache = LRUCache(2)
        cache.put(1, 10)
        cache.put(1, 20)
        assert cache.get(1) == 20


class TestEviction:
    def test_evicts_lru_on_overflow(self):
        cache = LRUCache(2)
        cache.put(1, 1)
        cache.put(2, 2)
        cache.put(3, 3)  # evicts key 1
        assert cache.get(1) == -1
        assert cache.get(2) == 2
        assert cache.get(3) == 3

    def test_get_refreshes_usage(self):
        cache = LRUCache(2)
        cache.put(1, 1)
        cache.put(2, 2)
        cache.get(1)      # refreshes key 1
        cache.put(3, 3)   # evicts key 2 (not 1)
        assert cache.get(1) == 1
        assert cache.get(2) == -1
        assert cache.get(3) == 3

    def test_put_refreshes_usage(self):
        cache = LRUCache(2)
        cache.put(1, 1)
        cache.put(2, 2)
        cache.put(1, 10)  # update refreshes key 1
        cache.put(3, 3)   # evicts key 2
        assert cache.get(1) == 10
        assert cache.get(2) == -1

    def test_capacity_1(self):
        cache = LRUCache(1)
        cache.put(1, 1)
        cache.put(2, 2)  # evicts key 1
        assert cache.get(1) == -1
        assert cache.get(2) == 2


class TestSequence:
    def test_leetcode_example(self):
        cache = LRUCache(2)
        cache.put(1, 1)
        cache.put(2, 2)
        assert cache.get(1) == 1
        cache.put(3, 3)
        assert cache.get(2) == -1
        cache.put(4, 4)
        assert cache.get(1) == -1
        assert cache.get(3) == 3
        assert cache.get(4) == 4

    def test_many_operations(self):
        cache = LRUCache(3)
        for i in range(10):
            cache.put(i, i * 10)
        # Only last 3 should survive
        for i in range(7):
            assert cache.get(i) == -1
        for i in range(7, 10):
            assert cache.get(i) == i * 10


class TestEdgeCases:
    def test_get_miss_does_not_affect_order(self):
        cache = LRUCache(2)
        cache.put(1, 1)
        cache.put(2, 2)
        cache.get(99)     # miss — should not change LRU order
        cache.put(3, 3)   # should evict key 1 (oldest)
        assert cache.get(1) == -1
        assert cache.get(2) == 2

    def test_overwrite_does_not_grow_size(self):
        cache = LRUCache(2)
        cache.put(1, 1)
        cache.put(1, 2)
        cache.put(1, 3)
        cache.put(2, 2)
        # Should still have room — only 2 keys
        assert cache.get(1) == 3
        assert cache.get(2) == 2
