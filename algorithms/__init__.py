"""
Algorithms module for LA-MAS experiments.
"""

from .ucb import UCB1, SlidingWindowUCB
from .ucb_bo import UCB1_with_BO_Optimization
from .thompson_sampling import SW_TS_gaussian, DS_TS_gaussian, ThompsonSampling_fDSW
from .lamas import LikelihoodAdaptiveMultiAgentSystem
from .glr_klucb import GLRklUCB
from .monitored_ucb import MonitoredUCB, ChangeDetectorCD
from .damas import UCB1_multiple_C, MultiAgentSystem_BO

__all__ = [
    'UCB1', 'SlidingWindowUCB',
    'UCB1_with_BO_Optimization',
    'SW_TS_gaussian', 'DS_TS_gaussian', 'ThompsonSampling_fDSW',
    'LikelihoodAdaptiveMultiAgentSystem',
    'GLRklUCB',
    'MonitoredUCB',
    'ChangeDetectorCD',
    'UCB1_multiple_C', 'MultiAgentSystem_BO'
]
