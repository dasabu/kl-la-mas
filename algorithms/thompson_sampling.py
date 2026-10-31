"""
Thompson Sampling algorithms for non-stationary bandits.
"""

import numpy as np
from collections import deque


class SW_TS_gaussian:
    """
    Sliding-Window Thompson Sampling for Gaussian rewards.
    
    Maintains a fixed-size window of recent observations per arm.
    """
    
    def __init__(
        self,
        n_actions: int, 
        W: int,
        kappa: float, 
        tau_max: float, 
        prior_mean: float = 0.0
    ):
        """
        Initialize SW-TS.
        
        Args:
            n_actions: Number of arms
            W: Window size (number of recent observations to keep)
            kappa: Scale for sampling std; for Gaussian rewards set kappa = 2*sigma
            tau_max: Cap on std for early exploration
            prior_mean: Prior mean when no observations exist
        """
        self.K = int(n_actions)
        self.W = int(W)
        self.kappa = float(kappa)
        self.tau_max = float(tau_max)
        self.prior_mean = float(prior_mean)

        # Per-arm state
        self.t = 0  # global time
        self.buffers = [deque() for _ in range(self.K)]  # (t, reward)
        self.sum_win = np.zeros(self.K, dtype=float)
        self.n_win = np.zeros(self.K, dtype=float)
        self.mu_hat = np.full(self.K, prior_mean, dtype=float)
        self.tau = np.full(self.K, tau_max, dtype=float)
        self.q_values = np.zeros(self.K, dtype=float)

        # Runtime metrics
        self.total_response_time = 0.0
        self.total_responses = 0
        self.average_response_time = 0.0

    def _evict_old(self, arm: int):
        """Evict observations older than window W for given arm."""
        cutoff = self.t - self.W
        buf = self.buffers[arm]
        while buf and buf[0][0] <= cutoff:
            _, r_old = buf.popleft()
            self.sum_win[arm] -= r_old
            self.n_win[arm] -= 1.0

    def _refresh_arm_stats(self, arm: int):
        """Update posterior parameters for given arm."""
        if self.n_win[arm] > 1e-12:
            self.mu_hat[arm] = self.sum_win[arm] / self.n_win[arm]
            self.tau[arm] = min(self.kappa / np.sqrt(self.n_win[arm]), self.tau_max)
        else:
            self.mu_hat[arm] = self.prior_mean
            self.tau[arm] = self.tau_max

    def track_response_time(self, delay: float) -> float:
        """Track response time metrics."""
        self.total_response_time += float(delay)
        self.total_responses += 1
        self.average_response_time = (
            self.total_response_time / self.total_responses 
            if self.total_responses else 0.0
        )
        return self.average_response_time

    def select_action(self) -> int:
        """
        Select action by Thompson Sampling.
        
        Returns:
            Selected action index
        """
        # Refresh all arms
        for a in range(self.K):
            self._evict_old(a)
            self._refresh_arm_stats(a)
        
        # Sample from posterior
        samples = np.random.normal(self.mu_hat, np.clip(self.tau, 1e-12, None))
        self.q_values = samples.copy()
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float, delay: float = 0.0):
        """
        Update after observing reward.
        
        Args:
            arm: Chosen arm
            reward: Observed reward
            delay: Response time/delay
        """
        self.t += 1
        self._evict_old(arm)
        self.buffers[arm].append((self.t, float(reward)))
        self.sum_win[arm] += float(reward)
        self.n_win[arm] += 1.0
        self._refresh_arm_stats(arm)
        self.track_response_time(delay)

    def effective_samples(self, arm: int) -> float:
        """Get effective number of samples for arm."""
        return float(self.n_win[arm])
    
    def variance(self, arm: int) -> float:
        """Get variance for arm."""
        return float(self.tau[arm] ** 2)


class DS_TS_gaussian:
    """
    Discounted Thompson Sampling for Gaussian rewards.
    
    Uses exponential discounting of historical observations.
    """
    
    def __init__(self, n_actions=10, gamma=0.95, kexi=1, 
                 tao_max=0.18, prior_mean=0, prior_var=1):
        """
        Initialize DS-TS.
        
        Args:
            n_actions: Number of arms
            gamma: Discount factor (0 < gamma <= 1)
            kexi: Scale parameter for posterior variance
            tao_max: Maximum standard deviation
            prior_mean: Prior mean
            prior_var: Prior variance
        """
        self.n_actions = n_actions
        self.gamma = gamma
        self.kexi = kexi
        self.tao_max = tao_max
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        
        # Discounted sufficient statistics
        self.S = np.zeros(self.n_actions)  # Discounted sum of rewards
        self.nb_draws = np.zeros(self.n_actions)  # Discounted number of draws
        
        # Posterior estimates
        self.mu_hat = np.zeros(self.n_actions)
        self.tao = np.ones(self.n_actions) * tao_max
        
        # Runtime metrics
        self.total_response_time = 0
        self.total_responses = 0
        self.average_response_time = 0
        
        self.q_values = np.zeros(self.n_actions)
        
    def track_response_time(self, delay):
        """Track response time metrics."""
        self.total_response_time += delay
        self.total_responses += 1
        self.average_response_time = (
            self.total_response_time / self.total_responses 
            if self.total_responses else 0
        )
        return self.average_response_time
    
    def max_(self, values):
        """Find index of maximum value."""
        max_index = 0
        maxv = values[max_index]
        for i in range(len(values)):
            if values[i] > maxv:
                maxv = values[i]
                max_index = i
        return max_index
    
    def select_action(self):
        """
        Select action by Thompson Sampling with discounting.
        
        Returns:
            Selected action index
        """
        # Update posterior parameters for all arms
        for i in range(self.n_actions):
            if self.nb_draws[i] < 1e-8:
                self.tao[i] = self.tao_max
                self.mu_hat[i] = self.prior_mean
            else:
                self.tao[i] = min(self.kexi / (self.nb_draws[i]**0.5), self.tao_max)
                self.mu_hat[i] = self.S[i] / self.nb_draws[i]
        
        # Sample from posterior
        samples = np.random.normal(self.mu_hat, self.tao)
        self.q_values = samples.copy()
        
        return self.max_(samples)
    
    def update(self, chosen_act, reward, delay=0, env_prob=1.0):
        """
        Update with exponential discounting.
        
        Args:
            chosen_act: Chosen action
            reward: Observed reward
            delay: Response time/delay
            env_prob: Environment probability weight
        """
        # Apply discounting to all statistics
        self.S *= self.gamma
        self.nb_draws *= self.gamma
        
        # Update chosen action
        weighted_reward = reward * env_prob
        self.S[chosen_act] += weighted_reward
        self.nb_draws[chosen_act] += env_prob
        
        # Update q_values
        if self.nb_draws[chosen_act] > 1e-8:
            self.q_values[chosen_act] = self.S[chosen_act] / self.nb_draws[chosen_act]
        else:
            self.q_values[chosen_act] = self.prior_mean
        
        self.track_response_time(delay)
    
    def calculate_variance(self, action):
        """Calculate posterior variance for action."""
        if self.nb_draws[action] < 1e-8:
            return self.tao_max
        else:
            return min(self.kexi / (self.nb_draws[action]**0.5), self.tao_max)


class ThompsonSampling_fDSW:
    """
    Function Discounted and Sliding Window Thompson Sampling.
    
    Combines exponential discounting with sliding window approach.
    """
    
    def __init__(self, n_actions=10, mu_0=0, lambda_0=1, alpha_0=1, 
                 beta_0=1, window_size=100, discount_factor=0.9):
        """
        Initialize fDSW-TS.
        
        Args:
            n_actions: Number of arms
            mu_0: Prior mean
            lambda_0: Prior precision scaling
            alpha_0: Prior alpha for Gamma distribution
            beta_0: Prior beta for Gamma distribution
            window_size: Size of sliding window
            discount_factor: Exponential discount factor
        """
        self.n_actions = n_actions

        # Hyperparameters for Normal-Gamma prior
        self.mu_0 = mu_0
        self.lambda_0 = lambda_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        self.counts = np.zeros(self.n_actions)
        self.sum_x = np.zeros(self.n_actions)
        self.sum_x_square = np.zeros(self.n_actions)

        self.window_size = window_size
        self.reward_windows = [deque(maxlen=window_size) for _ in range(n_actions)]
        self.discount_factor = discount_factor
        
        self.n_actions_taken = np.zeros(self.n_actions)
        self.total_response_time = 0
        self.total_responses = 0
        self.average_response_time = 0

        self.q_values = np.zeros(self.n_actions)

    def track_response_time(self, delay):
        """Track response time metrics."""
        self.total_response_time += delay
        self.total_responses += 1
        self.average_response_time = (
            self.total_response_time / self.total_responses 
            if self.total_responses else 0
        )
        return self.average_response_time

    def max_(self, values):
        """Find index of maximum value."""
        max_index = 0
        maxv = values[max_index]
        for i in range(len(values)):
            if values[i] > maxv:
                maxv = values[i]
                max_index = i
        return max_index

    def select_action(self):
        """
        Select action by fused Thompson Sampling.
        
        Returns:
            Selected action index
        """
        samples = np.zeros(self.n_actions)
        for k in range(self.n_actions):
            # Discounted historical data
            historical_alpha = self.alpha_0 + self.counts[k] / 2
            historical_beta = (
                self.beta_0 + 
                0.5 * (self.sum_x_square[k] - 
                       (self.sum_x[k] ** 2) / (self.counts[k] + 1))
            )
            
            # Sliding window data
            window_rewards = np.array(self.reward_windows[k])
            window_alpha = self.alpha_0 + len(window_rewards) / 2
            window_beta = (
                self.beta_0 + 
                0.5 * (np.sum(window_rewards**2) - 
                       (np.sum(window_rewards) ** 2) / (len(window_rewards) + 1))
            )
            
            # Aggregate using mean
            combined_alpha = (historical_alpha + window_alpha) / 2
            combined_beta = (historical_beta + window_beta) / 2

            # Ensure combined_beta is positive
            if combined_beta <= 0:
                combined_beta = 1e-10

            # Sample from posterior
            tau_k = np.random.gamma(combined_alpha, 1 / combined_beta)
            
            historical_mean = (
                self.sum_x[k] / (self.lambda_0 + self.counts[k])
                if self.counts[k] > 0 else 0
            )
            window_mean = (
                np.sum(window_rewards) / (self.lambda_0 + len(window_rewards))
                if len(window_rewards) > 0 else 0
            )
            
            mu_k = np.random.normal(
                np.mean([historical_mean, window_mean]), 
                1 / np.sqrt(tau_k * (
                    self.lambda_0 + self.counts[k] + len(window_rewards)
                ))
            )
            
            samples[k] = mu_k

        self.q_values = samples
        return self.max_(samples)
    
    def update(self, chosen_act, reward, delay=0, env_prob=1.0):
        """
        Update with discounting and sliding window.
        
        Args:
            chosen_act: Chosen action
            reward: Observed reward
            delay: Response time/delay
            env_prob: Environment probability weight
        """
        # Apply discount factor to historical data
        self.counts[chosen_act] *= self.discount_factor
        self.sum_x[chosen_act] *= self.discount_factor
        self.sum_x_square[chosen_act] *= self.discount_factor

        # Update historical data with new reward
        self.counts[chosen_act] += 1 * env_prob
        self.sum_x[chosen_act] += reward * env_prob
        self.sum_x_square[chosen_act] += (reward * env_prob) ** 2

        # Update sliding window
        self.reward_windows[chosen_act].append(reward * env_prob)
        
        # Update q_values
        self.q_values[chosen_act] = (
            self.sum_x[chosen_act] / self.counts[chosen_act] 
            if self.counts[chosen_act] > 0 else 0
        )

        self.track_response_time(delay)

    def calculate_variance(self, action):
        """Calculate variance for action."""
        if len(self.reward_windows[action]) > 1:
            return np.var(self.reward_windows[action])
        else:
            return 1
