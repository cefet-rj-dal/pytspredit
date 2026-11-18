"""Time series augmentation.

Only the identity augmenter (``ts_aug_none``) ships in v1; the full augmentation
family (jitter, flip, stretch, shrink, wormhole, awareness, ...) is deferred to a
later release. ``TsAugNone`` exists so :class:`~pytspredit.TsIntegtune` has a
default, no-op augmentation step.
"""
from __future__ import annotations

from pydaltoolbox import DalTransform

__all__ = ["TsAugNone"]


class TsAugNone(DalTransform):
    """Identity augmentation (``ts_aug_none``): returns the windows unchanged."""

    def fit(self, data=None, **kwargs) -> TsAugNone:
        return self

    def transform(self, data, **kwargs):
        return data
