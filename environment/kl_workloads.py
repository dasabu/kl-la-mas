from typing import Tuple

import numpy as np


def workloads2(n_env: int, num_actions: int) -> Tuple[np.ndarray, np.ndarray]:
    mus = np.zeros((n_env, num_actions))
    sigmas = np.zeros((n_env, num_actions))
    for env in range(n_env):
        if env % 3 == 0:
            mus[env] = np.random.uniform(low=1.5, high=2.5, size=num_actions)
            sigmas[env] = np.random.uniform(low=0.1, high=0.3, size=num_actions)
            dominant_action = np.random.randint(num_actions)
            mus[env, dominant_action] = 1.0
            sigmas[env, dominant_action] = 0.4
        elif env % 3 == 1:
            mus[env] = np.random.uniform(low=0.02, high=0.1, size=num_actions)
            sigmas[env] = np.random.uniform(low=0.005, high=0.015, size=num_actions)
            dominant_action = np.random.randint(num_actions)
            mus[env, dominant_action] = 0.015
            sigmas[env, dominant_action] = 0.02
        else:
            mus[env] = np.random.uniform(low=0.3, high=0.6, size=num_actions)
            sigmas[env] = np.random.uniform(low=0.01, high=0.02, size=num_actions)
            dominant_action = np.random.randint(num_actions)
            mus[env, dominant_action] = 0.2
            sigmas[env, dominant_action] = 0.03
    return mus, sigmas


def controlled_variance_workloads():
    mus = np.full((3, 1), 2.0)
    sigmas = np.array([[0.40], [0.04], [0.40]])
    return mus, sigmas
