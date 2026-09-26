"""Step 15 - Score fusion maths (src/hybrid.py), checked on numbers you can verify by hand."""
import numpy as np
import pytest

from hybrid import minmax, rrf, weighted


def test_minmax_scales_to_0_1():
    assert minmax(np.array([2.0, 4.0, 6.0])) == pytest.approx([0.0, 0.5, 1.0])


def test_minmax_constant_vector_gives_zeros():
    assert minmax(np.array([3.0, 3.0])) == pytest.approx([0.0, 0.0])


def test_weighted_alpha_extremes():
    bm25, dense = np.array([10.0, 0.0, 5.0]), np.array([0.80, 0.90, 0.85])
    assert weighted(bm25, dense, alpha=1.0) == pytest.approx(minmax(dense))   # pure dense
    assert weighted(bm25, dense, alpha=0.0) == pytest.approx(minmax(bm25))    # pure BM25


def test_rrf_by_hand():
    # chunk 0 is rank 1 in both lists -> 1/61 + 1/61
    bm25, dense = np.array([3.0, 2.0, 1.0]), np.array([0.9, 0.1, 0.5])
    fused = rrf(bm25, dense, k=60)
    assert fused[0] == pytest.approx(2 / 61)
    assert fused[1] == pytest.approx(1 / 62 + 1 / 63)
