"""Application services that compose lower-level CommCanary boundaries."""

from .behavior_search import synthesize_behavioral_canary
from .compile import compile_trace
from .reduction import ddmin_ranking_reduction

__all__ = ["compile_trace", "ddmin_ranking_reduction", "synthesize_behavioral_canary"]
