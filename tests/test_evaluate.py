from __future__ import annotations

import numpy as np
import pytest

from src.models.evaluate import select_operating_threshold


def test_select_operating_threshold_picks_argmax_1d():
    px = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1.0]
    f1 = np.array([0.1, 0.4, 0.9, 0.3, 0.0])
    threshold, best_f1 = select_operating_threshold(px, f1)
    assert threshold == pytest.approx(0.5)
    assert best_f1 == pytest.approx(0.9)


def test_select_operating_threshold_averages_across_classes_2d():
    px = np.linspace(0, 1, 4)
    # 2 classes x 4 thresholds; mean is highest at index 2.
    f1 = np.array(
        [
            [0.2, 0.5, 0.9, 0.1],
            [0.4, 0.5, 0.7, 0.2],
        ]
    )
    threshold, best_f1 = select_operating_threshold(px, f1)
    assert threshold == pytest.approx(px[2])
    assert best_f1 == pytest.approx((0.9 + 0.7) / 2)


def test_select_operating_threshold_returns_plain_floats():
    px = np.linspace(0, 1, 3)
    f1 = np.array([0.1, 0.2, 0.3])
    threshold, best_f1 = select_operating_threshold(px, f1)
    assert isinstance(threshold, float)
    assert isinstance(best_f1, float)
