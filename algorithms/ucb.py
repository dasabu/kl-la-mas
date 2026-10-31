from collections import deque

import numpy as np


class UCB1:
    def __init__(self, c=1.0, n_actions=10, var_window=50):
        self.n_actions = n_actions
        self.c = c
        self.q_values = np.zeros(n_actions) # Estimate of q value, which is reward average, for each action
        self.n_actions_taken = np.zeros(n_actions) # Number of times each action has been taken
        self.total_rewards = np.zeros(n_actions) # Total sampled reward for each action
        self.reward_history = [[] for _ in range(n_actions)]  # Reward history list of each action
        
        # Track total number of responses and total response time for this agent
        self.total_response_time = 0 
        self.total_responses = 0
        
        self.var_window = var_window
    
    def track_response_time(self, response_time):
        """
        Increase total response time and total number of responses
        then caluculate average response time for this agent
        """
        self.total_response_time += response_time
        self.total_responses += 1
        self.average_response_time = self.total_response_time / self.total_responses if self.total_responses else 0
        return self.average_response_time
    
    def select_action(self):
        """
        Calculate UCB value for each action and choose the action with the highest UCB value
        """
        total_counts = np.sum(self.n_actions_taken)
        ucb_values = self.q_values + self.c * np.sqrt(
            2 * np.log(total_counts + 1) 
            / (self.n_actions_taken + 1)
        )
        return np.argmax(ucb_values)

    def update(self, action, reward, response_time=0, env_prob=1.0):
        """
        After choosing an action and receive a reward, update estimates
        Each estimate is updated with its corresponding weight, which is probability of current env
        """
        self.n_actions_taken[action] += 1 * env_prob    # p_i
        self.total_rewards[action] += reward * env_prob # r_i * p_i
        self.reward_history[action].append(reward * env_prob)
        self.q_values[action] = self.total_rewards[action] / self.n_actions_taken[action] # sum(r_i * p_i) / sum(p_i)
        
        # Track response time for the action
        self.track_response_time(response_time)
    
    def calculate_variance(self, action):
        """
        Calculate the variance of the reward history for the given action
        Get the last self.var_window reward in the history list
        """
        history = self.reward_history[action]
        window = history[-self.var_window:]
        return np.var(window) if len(window) > 1 else 1.0


class SlidingWindowUCB:
    """
    UCB algorithm with sliding window for non-stationary environments.
    """
    
    def __init__(
        self,
        c=1.0, 
        n_actions=10, 
        window_size=100,
        var_window=200
    ):
        """
        Initialize Sliding Window UCB.
        
        Args:
            c: Exploration constant
            n_actions: Number of actions/arms
            window_size: Size of sliding window for reward history
            var_window: Window size for variance calculation
        """
        self.n_actions = n_actions
        self.c = c
        self.window_size = window_size
        self.var_window = var_window
        
        self.q_values = np.zeros(n_actions)
        self.n_actions_taken = np.zeros(n_actions)
        self.reward_history = [deque(maxlen=window_size) for _ in range(n_actions)]
        
        self.total_response_time = 0
        self.total_responses = 0
        self.average_response_time = 0
        
    def track_response_time(self, delay):
        """Track response time metrics."""
        self.total_response_time += delay
        self.total_responses += 1
        self.average_response_time = (
            self.total_response_time / self.total_responses 
            if self.total_responses else 0
        )
        return self.average_response_time
    
    def select_action(self):
        """
        Select action using UCB formula with sliding window estimates.
        
        Returns:
            Selected action index
        """
        total_counts = np.sum(self.n_actions_taken)
        ucb_values = (
            self.q_values + 
            self.c * np.sqrt(2 * np.log(total_counts + 1) / (self.n_actions_taken + 1))
        )
        return np.argmax(ucb_values)
    
    def update(self, action, reward, delay=0, env_prob=1.0):
        """
        Update estimates with sliding window.
        
        Args:
            action: Action that was taken
            reward: Observed reward
            delay: Response time/delay
            env_prob: Environment probability weight
        """
        if len(self.reward_history[action]) == self.window_size:
            oldest_reward = self.reward_history[action].popleft()
            self.n_actions_taken[action] -= 1 * env_prob
            self.q_values[action] -= oldest_reward / self.window_size
        
        self.reward_history[action].append(reward * env_prob)
        self.n_actions_taken[action] += 1 * env_prob
        self.q_values[action] += reward * env_prob / self.window_size
        
        self.track_response_time(delay)
    
    def calculate_variance(self, action):
        """Calculate variance for given action using variance window."""
        history = self.reward_history[action]
        window = list(history)[-self.var_window:]
        return np.var(window) if len(window) > 1 else 1.0
