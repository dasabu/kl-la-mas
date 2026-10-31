"""
M-UCB: Monitored UCB with change detection.

Alternates between forced exploration and UCB exploitation while monitoring
for environment changes.
"""

import numpy as np


class ChangeDetectorCD:
    """
    Change detector based on comparing window halves.
    
    Detects change if the absolute difference between first and second halves
    of the window exceeds a threshold.
    """
    
    def __init__(self, w, b):
        """
        Initialize change detector.
        
        Args:
            w: Window size (must be even)
            b: Detection threshold
        """
        assert w % 2 == 0, "Window size must be even"
        self.w = w
        self.b = b

    def test(self, window):
        """
        Test for change in the window.
        
        Args:
            window: List of recent observations
            
        Returns:
            True if change detected, False otherwise
        """
        tail = window[-self.w:]
        half = self.w // 2
        return abs(sum(tail[half:]) - sum(tail[:half])) > self.b


class MonitoredUCB:
    """
    Monitored UCB (M-UCB) algorithm.
    
    Alternates between:
    - Forced exploration phase: Systematically tries each arm
    - UCB exploitation phase: Uses UCB to select arms
    
    Continuously monitors for changes and resets when detected.
    """
    def __init__(self,
        n_actions: int,
        w: int,
        b: float,
        gamma: float,
        c: float = 1.0
    ):
        """
        Initialize M-UCB.
        
        Args:
            n_actions: Number of actions
            w: Window size for change detection (must be even)
            b: Threshold for change detection
            gamma: Fraction of time reserved for forced exploration
            c: UCB bonus constant
        """
        self.K = n_actions
        self.w = w
        self.b = b
        self.gamma = gamma
        self.c = c

        # Period length = ⌊K/γ⌋
        self.period = int(np.floor(self.K / self.gamma))
        assert self.period > 0, "Period must be positive"

        # Last reset time
        self.tau = 0
        # Global time
        self.t = 0

        # Per-arm UCB statistics
        self.counts = np.zeros(self.K, dtype=int)
        self.sums = np.zeros(self.K, dtype=float)
        self.q = np.zeros(self.K, dtype=float)

        # Per-arm reward history (for change detection)
        self.history = [[] for _ in range(self.K)]
        
        # Change detector
        self.detector = ChangeDetectorCD(self.w, self.b)
        
        # Response time tracking for compatibility
        self.total_response_time = 0
        self.total_responses = 0
        self.average_response_time = 0
        
        # Track when changepoints are detected (reset times_
        self.reset_times = []
        
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
        Select action using forced exploration or UCB.
        
        For the first K steps of each period, forces exploration by playing
        each arm once. Otherwise, uses UCB to select the best arm.
        
        Returns:
            Selected action index
        """
        self.t += 1
        A = (self.t - self.tau) % self.period
        
        if A < self.K:
            # Forced exploration: play arm A
            return int(A)
        else:
            # UCB phase: select arm with highest UCB value
            total_counts = np.sum(self.counts)
            bonus = self.c * np.sqrt(
                2 * np.log((self.t - self.tau) + 1) / (self.counts + 1e-9)
            )
            return int(np.argmax(self.q + bonus))

    def update(self, action: int, reward: float, delay=0):
        """
        Update statistics and check for changepoint.
        
        If change is detected, resets all arms and restarts the algorithm.
        
        Args:
            action: Action taken
            reward: Observed reward
            delay: Response time
        """
        # Record observation
        self.history[action].append(reward)
        self.counts[action] += 1
        self.sums[action] += reward
        self.q[action] = self.sums[action] / self.counts[action]
        
        # Track response time
        self.track_response_time(delay)

        # Change detection and possible reset
        if self.counts[action] >= self.w:
            if self.detector.test(self.history[action]):
                # Record changepoint detection time
                self.reset_times.append(self.t)
                
                # Reset all arms
                self.tau = self.t
                self.counts.fill(0)
                self.sums.fill(0.0)
                self.q.fill(0.0)
                self.history = [[] for _ in range(self.K)]
