"""Intentional package-private ranking policy for application services."""

from ..behavior_config import BEHAVIORAL_RANKING_METRICS, ranking_relation

RANKING_METRICS = BEHAVIORAL_RANKING_METRICS

__all__ = ["RANKING_METRICS", "ranking_relation"]
