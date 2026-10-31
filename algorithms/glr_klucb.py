"""
GLR-klUCB: Generalized Likelihood Ratio test with KL-UCB.

Uses GLR test for changepoint detection and KL-UCB for action selection.
"""

import numpy as np
from math import log, sqrt


class GLRklUCB:
    """
    GLR-klUCB algorithm for Gaussian rewards with known variance.
    
    Combines:
    - Generalized Likelihood Ratio (GLR) test for changepoint detection
    - KL-UCB for action selection with exploration bonus
    
    Supports both local (per-arm) and global (all arms) restart strategies.
    """
    
    def __init__(
        self,
        n_actions: int,
        sigma: float = 1.0,
        c_index: float = 1.0,
        glr_window: int = 50,
        glr_threshold: float = 10.0,
        restart_mode: str = "global"
    ):
        """
        GLR-klUCB for Gaussian rewards with known variance sigma^2.
        
        Args:
            n_actions: Number of arms
            sigma: Known standard deviation of noise
            c_index: Extra log-log term weight in index
            glr_window: Window size for GLR test
            glr_threshold: Threshold to declare changepoint
            restart_mode: "local" (reset only detected arm) or "global" (reset all arms)
        """
        self.K = n_actions
        self.sigma2 = sigma**2
        self.c_index = c_index
        self.w = glr_window
        self.h = glr_threshold
        self.restart_mode = restart_mode.lower()
        assert self.restart_mode in {"local", "global"}
        
        # UCB state
        self.N = np.zeros(self.K, dtype=int)      # pulls per arm
        self.S = np.zeros(self.K, dtype=float)    # sum of rewards
        self.mu = np.zeros(self.K, dtype=float)   # sample means
        
        # Per-arm reward histories (for GLR)
        self.hist = [[] for _ in range(self.K)]
        
        # Global time
        self.t = 0
        
        # Response time tracking
        self.total_response_time = 0
        self.total_responses = 0
        self.average_response_time = 0
        self.reset_times = []
        self.reset_events = []
        
    def track_response_time(self, delay):
        """Track response time metrics."""
        self.total_response_time += delay 
        self.total_responses += 1 
        self.average_response_time = (
            self.total_response_time / self.total_responses 
            if self.total_responses else 0
        )
        return self.average_response_time
        
    def select_action(self) -> int:
        """
        Select action using KL-UCB index.
        
        Returns:
            Selected action index
        """
        self.t += 1
        
        # Initial round-robin
        for k in range(self.K):
            if self.N[k] == 0:
                return k
                
        idx = np.zeros(self.K)
        bonus_term = log(self.t) + self.c_index * log(max(2, log(self.t)))
        
        for k in range(self.K):
            idx[k] = self.mu[k] + sqrt(2 * self.sigma2 * bonus_term / self.N[k])
            
        return int(np.argmax(idx))
    
    def _glr_stat(self, k: int) -> float:
        """
        Compute GLR statistic over the last w observations of arm k.
        
        GLR_k(n) = sup_{s in [n-w, n)} [ L_{s:n}(mu1) + L_{1:s}(mu0) - L_{1:n}(mu_all) ]
        
        All likelihoods under N(mean, sigma2).
        
        Args:
            k: Arm index
            
        Returns:
            GLR statistic value
        """
        x = np.array(self.hist[k], dtype=float)
        n = len(x)
        if n < 2:
            return 0.0
            
        # Only consider splits within last w observations
        start = max(0, n - self.w)
        glr = 0.0
        
        # Precompute cumulative sums for efficiency
        csum = np.cumsum(x)
        
        for s in range(start + 1, n):
            # Segment means
            mu0 = csum[s-1] / s if s > 0 else 0.0  
            mu1 = (csum[n-1] - csum[s-1]) / (n - s) if (n-s) > 0 else 0.0
            mu_all = csum[n-1] / n
            
            
            L0 = -np.sum((x[:s] - mu0)**2) / (2*self.sigma2) if s > 0 else 0.0
            L1 = -np.sum((x[s:] - mu1)**2) / (2*self.sigma2) if (n-s) > 0 else 0.0
            Lall = -np.sum((x - mu_all)**2) / (2*self.sigma2)
            
            glr = max(glr, (L0 + L1 - Lall))
            
        return glr
    
    def _reset_arm(self, k: int):
        """Reset a single arm."""
        self.N[k] = 0
        self.S[k] = 0.0
        self.mu[k] = 0.0
        self.hist[k].clear()
    
    def _reset_all_arms(self):
        """Reset all arms (global restart)."""
        for k in range(self.K):
            self._reset_arm(k)
    
    def update(self, action: int, reward: float, delay=0):
        """
        Update after observing reward and check for changepoints.
        
        Args:
            action: Arm pulled
            reward: Observed reward
            delay: Response time
            
        """
        # Record observation
        k = action
        self.hist[k].append(reward)
        self.N[k] += 1
        self.S[k] += reward
        self.mu[k] = self.S[k] / self.N[k]
        
        # Track response time
        self.track_response_time(delay)
        
        # Changepoint test on arm k
        if len(self.hist[k]) >= self.w:
            if self._glr_stat(k) > self.h:
                # Record the reset event
                self.reset_times.append(self.t)
                
                if self.restart_mode == "local":
                    # Local restart: reset only the detected arm
                    self._reset_arm(k)
                    self.reset_events.append((self.t, "local", k))
                else:  # global restart
                    # Global restart: reset all arms
                    self._reset_all_arms()
                    self.reset_events.append((self.t, "global", "all"))
    
    def get_reset_info(self):
        """
        Get information about reset events.
        
        Returns:
            Dictionary with reset times, events, and total count
        """
        return {
            'reset_times': self.reset_times.copy(),
            'reset_events': self.reset_events.copy(),
            'total_resets': len(self.reset_times)
        }
