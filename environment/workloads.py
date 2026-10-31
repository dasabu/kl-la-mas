"""
Synthetic + Real-world workload generation for bandit experiments.
"""

import os
import numpy as np
import pandas as pd


def synthetic_workloads(n_env=4, num_actions=10):
    """
    Generate synthetic workloads with different characteristics.
    
    Args:
        n_env: Number of different environments
        num_actions: Number of actions/arms per environment
        
    Returns:
        mus: Array of means for each environment and action
        sigmas: Array of standard deviations for each environment and action
    """
    mus = np.zeros((n_env, num_actions))
    sigmas = np.zeros((n_env, num_actions))
    
    np.random.seed(42)  # For reproducibility
    
    for env in range(n_env):
        if env % 4 == 0:
            mus[env] = np.random.uniform(low=0.3, high=0.6, size=num_actions)
            sigmas[env] = np.random.uniform(low=0.01, high=0.02, size=num_actions)
            # One action to be dominant (lower mean and slightly higher std)
            dominant_action = np.random.randint(num_actions)
            mus[env, dominant_action] = 0.2
            sigmas[env, dominant_action] = 0.03
        
        elif env % 4 == 1:
            mus[env] = np.random.uniform(low=0.02, high=0.1, size=num_actions)
            sigmas[env] = np.random.uniform(low=0.005, high=0.015, size=num_actions)
            # One action to be dominant
            dominant_action = np.random.randint(num_actions)
            mus[env, dominant_action] = 0.015
            sigmas[env, dominant_action] = 0.02
        
        elif env % 4 == 2:
            mus[env] = np.random.uniform(low=0.6, high=0.9, size=num_actions)
            sigmas[env] = np.random.uniform(low=0.02, high=0.05, size=num_actions)
            # One action to be dominant
            dominant_action = np.random.randint(num_actions)
            mus[env, dominant_action] = 0.5
            sigmas[env, dominant_action] = 0.15
        
        else:  # env % 4 == 3
            # New fourth environment type
            mus[env] = np.random.uniform(low=1.5, high=2.5, size=num_actions)
            sigmas[env] = np.random.uniform(low=0.4, high=0.6, size=num_actions)
            # One action to be dominant
            dominant_action = np.random.randint(num_actions)
            mus[env, dominant_action] = 1.0
            sigmas[env, dominant_action] = 0.5
    
    return mus, sigmas

def load_dataset(data_dir, file_names):
    """
    Load all sigmas and mus from a CSV file, each is a different env
    Each env contains 38 configurations (38 actions)
    Each configuration describes a reward distribution, characterized by mean (mu) and std (sigma)
    
    Return:
        mus_all: mean of reward distribution of each action in every env
        sigmas_all: std of reward distribution of each action in every env
        
        mus_all[e, a]: mean of action a in env e
        sigmas_all[e, a]: std of action a in env e
    """
    mus_all, sigmas_all = [], []
    for fname in file_names:
        df = pd.read_csv(os.path.join(data_dir, fname))
        mus_all.append(df['Mean Average Response Time'].to_numpy(dtype=float))
        sigmas_all.append(df['STD'].to_numpy(dtype=float))
    mus_all = np.vstack(mus_all)
    sigmas_all = np.vstack(sigmas_all)
    return mus_all, sigmas_all

def real_world_workloads():
    """
    Generate real-world workloads with different characteristics.
    """
    mus_all, sigmas_all = load_dataset(
        os.path.join(os.path.dirname(__file__), "real_world_scenario"),
        ["HTML_text.csv", "MP4.csv", "MP4_high_entropy.csv", "JPG.csv"]
    ) # (4, 38)

    return mus_all, sigmas_all
