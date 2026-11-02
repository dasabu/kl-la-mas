import math
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np


class UCB1:
    

    def __init__(self, c: float = 1.0, n_actions: int = 10):
        self.n_actions = n_actions
        self.c = c
        self.q_values = np.zeros(n_actions)
        self.n_actions_taken = np.zeros(n_actions)
        self.total_rewards = np.zeros(n_actions)
        # history of rewards for variance estimation
        self.reward_history: List[List[float]] = [[] for _ in range(n_actions)]
        # response time tracking (optional)
        self.total_response_time = 0.0
        self.total_responses = 0

    def track_response_time(self, delay: float) -> float:
        self.total_response_time += delay
        self.total_responses += 1
        return self.total_response_time / max(1, self.total_responses)

    def select_action(self) -> int:
        total_counts = np.sum(self.n_actions_taken)
        # ensure denominator is at least one to avoid division by zero
        ucb_values = self.q_values + self.c * np.sqrt(
            2 * np.log(total_counts + 1) / (self.n_actions_taken + 1)
        )
        return int(np.argmax(ucb_values))

    def update(self, action: int, reward: float, delay: float = 0.0, env_prob: float = 1.0) -> None:
        # weight increments by the environment probability
        self.n_actions_taken[action] += env_prob
        self.total_rewards[action] += reward * env_prob
        self.reward_history[action].append(reward * env_prob)
        # update mean estimate
        self.q_values[action] = self.total_rewards[action] / max(1e-8, self.n_actions_taken[action])
        # optional response time tracking
        self.track_response_time(delay)

    def calculate_variance(self, action: int) -> float:
        # return variance of the reward history; if fewer than two samples return 1 to avoid zero variance
        if len(self.reward_history[action]) > 1:
            return float(np.var(self.reward_history[action]))
        return 1.0

class LikelihoodAdaptiveMultiAgentSystem_KL:
    
    def __init__(
        self,
        agent_type,
        agent_params: dict,
        detection_threshold: float = 0.05,
        delta: float = 0.07,
        cooldown_steps: int = 40,
        min_data_points: int = 40,
        min_val: float = 1e-6,
        window_size: int = 50,
        window_overlap: int = 0,  #  overlap between windows
        kl_threshold: float = 0.1,
        kl_multiplier: float = 1.0,
    ) -> None:
        self.agent_type = agent_type
        self.agent_params = agent_params
        self.detection_threshold = detection_threshold
        self.delta = delta
        self.cooldown_steps = cooldown_steps
        self.min_val = min_val
        # sliding window parameters
        self.window_size = window_size
        self.window_overlap = min(window_overlap, window_size - 1)  # overlap must be < window_size
        self.window_step = window_size - self.window_overlap  # how many new samples before next comparison
        self.kl_threshold = kl_threshold
        self.kl_multiplier = kl_multiplier
        # create first agent
        self.agents: List[UCB1] = [agent_type(**agent_params)]
        self.p_env = np.array([1.0])
        # history and counters (for changepoint detection)
        self.global_step = 0
        self.steps_since_last_add = cooldown_steps # this allow detector to start detecting from starting point (in case enough sliding window)
        self.reward_history: List[float] = []
        self.changepoints: List[int] = []
        self.min_data_points = min_data_points
        # response time tracking
        self.total_response_time = 0.0
        self.total_responses = 0
        # likelihood history per agent
        self.likelihood_history: List[List[float]] = []
        # learning phase flags
        self.learning_agent_idx = -1
        self.learning_steps_taken = 0
        self.learning_phase_active = False
        
        # CONFIGURABLE OVERLAP WINDOWS
        n_actions = agent_params.get("n_actions", 1)
        # Store all rewards per action (we'll take slices for windows)
        self.all_rewards: List[Deque[float]] = [deque(maxlen=window_size * 2) for _ in range(n_actions)]
        # Track how many rewards we've collected per action
        self.reward_count_per_action: List[int] = [0] * n_actions
        # Track when we last computed KL for each action
        self.last_kl_step_per_action: List[int] = [0] * n_actions
        
        # store distributions computed from windows
        self.prev_distributions: List[Optional[Tuple[float, float]]] = [None] * n_actions
        self.current_distributions: List[Optional[Tuple[float, float]]] = [None] * n_actions
        
        # store KL divergence values
        self.kl_history: List[float] = []
        
        
        self.low_likelihood_count = 0
        self.min_m = 4

    def track_response_time(self, delay: float) -> float:
        self.total_response_time += delay
        self.total_responses += 1
        return self.total_response_time / max(1, self.total_responses)

    @staticmethod
    def local_likelihood(reward: float, mu: float, sigma: float, delta_frac: float = 3.0, min_delta: float = 0.07) -> float:
        delta = max(min_delta, delta_frac * sigma)
        from scipy.stats import norm
        return float(norm.cdf(reward + delta, mu, sigma) - norm.cdf(reward - delta, mu, sigma))

    @staticmethod
    def kl_divergence_gaussian(mu1: float, var1: float, mu2: float, var2: float) -> float:
        """Return the KL divergence KL(N(mu1, var1) || N(mu2, var2))."""
        
        
        return float(
            0.5 * (
                ((mu2 - mu1) ** 2) / var2 + 
                var1 / var2 - 
                math.log(var1 / var2) - 
                1
            )
        )
    
    

    def detect_changepoint_likelihood(self, action: int, reward: float) -> bool:
        if len(self.reward_history) < self.min_data_points:
            return False
        likelihoods: List[float] = []
        for ag in self.agents:
            mu = float(ag.q_values[action])
            std = float(np.sqrt(ag.calculate_variance(action)))
            p = self.local_likelihood(reward, mu, std, delta_frac=3.0, min_delta=self.delta)
            likelihoods.append(p)
        
    
        # if max(likelihoods) < self.detection_threshold:
        #     print(f"Step {self.global_step}: Likelihood change detected! max_p={max(likelihoods):.3f}")
        #     return True
        # return False
        
        self.low_likelihood_count += 1 if max(likelihoods) < self.detection_threshold else 0
        if self.low_likelihood_count >= self.min_m:
            # trigger change‑point
            self.low_likelihood_count = 0
            return True

    def detect_changepoint_kl(self, action: int) -> bool:
        """Detect change using KL divergence between overlapping windows.
        
        Only triggers when both previous and current distributions are available.
        """
        if self.prev_distributions[action] is None or self.current_distributions[action] is None:
            return False
    
        mu_prev, var_prev = self.prev_distributions[action]
        mu_curr, var_curr = self.current_distributions[action]
        
        kl = self.kl_divergence_gaussian(mu_prev, var_prev, mu_curr, var_curr)
        
        # store KL value
        self.kl_history.append(kl)
        
        # compute mean and std of previous KL values (excluding current)
        past_values = self.kl_history[:-1] if len(self.kl_history) > 1 else []
        mean_kl = np.mean(past_values) if past_values else 0.0
        std_kl = np.std(past_values) if past_values else 0.0
        
        # detection criteria: either above absolute threshold or above mean + k*std
        condition1 = kl > self.kl_threshold
        condition2 = kl > mean_kl + self.kl_multiplier * std_kl if past_values else False
        
        return  condition1

    def add_new_agent(self) -> None:
        idx = len(self.agents)
        self.agents.append(self.agent_type(**self.agent_params))
        self.p_env = np.ones(len(self.agents)) / len(self.agents)
        self.changepoints.append(self.global_step)
        self.steps_since_last_add = 0
        # begin learning phase for the new agent
        self.learning_agent_idx = idx
        self.learning_steps_taken = 0
        self.learning_phase_active = True

    def update_sliding_window(self, action: int, reward: float) -> None:
        
        # Add reward to the history
        self.all_rewards[action].append(reward)
        self.reward_count_per_action[action] += 1
        
        count = self.reward_count_per_action[action]
        last_kl_step = self.last_kl_step_per_action[action]
        
        # Check if we need to compute KL divergence
        # We need at least 2*window_size - overlap samples to have two full windows
        min_samples_needed = 2 * self.window_size - self.window_overlap
        
        if count < min_samples_needed:
            return  # Not enough data yet
        
        # Check if enough new samples have arrived since last KL computation
        samples_since_last_kl = count - last_kl_step
        if samples_since_last_kl < self.window_step:
            return  # Not time to compute KL yet
        
        # Extract two windows from the reward history
        all_rewards_list = list(self.all_rewards[action])
        
        # Previous window: oldest window_size samples
        prev_window = all_rewards_list[-(2 * self.window_size - self.window_overlap):-(self.window_size - self.window_overlap)]
        
        # Current window: most recent window_size samples
        curr_window = all_rewards_list[-self.window_size:]
        
        # Compute distributions
        if len(prev_window) == self.window_size and len(curr_window) == self.window_size:
            mu_prev = float(np.mean(prev_window))
            var_prev = float(np.var(prev_window)) + 1e-8
            self.prev_distributions[action] = (mu_prev, var_prev)
            
            mu_curr = float(np.mean(curr_window))
            var_curr = float(np.var(curr_window)) + 1e-8
            self.current_distributions[action] = (mu_curr, var_curr)
            
            # Update the last KL computation step
            self.last_kl_step_per_action[action] = count

    def select_action(self) -> int:
        """Select an action based on the current phase and agent probabilities."""
        if self.learning_phase_active and self.learning_agent_idx != -1 and self.learning_steps_taken < self.cooldown_steps:
            action = int(self.agents[self.learning_agent_idx].select_action())
            return action
        # outside learning phase: get proposals from all agents and sample according to p_env
        actions = [int(ag.select_action()) for ag in self.agents]
        return int(np.random.choice(actions, p=self.p_env))

    def update(self, action: int, reward: float, delay: float) -> None:
        self.global_step += 1
        self.steps_since_last_add += 1
        self.reward_history.append(reward)
        # track response time
        self.track_response_time(delay)
        
        # update sliding window for KL detection
        self.update_sliding_window(action, reward)
        
        # learning phase update
        if self.learning_phase_active and self.learning_agent_idx != -1:
            self.learning_steps_taken += 1
            # update only the learning agent
            self.agents[self.learning_agent_idx].update(action, reward, delay, env_prob=1.0)
            # accumulate likelihoods for analysis
            p_obs = []
            for i, ag in enumerate(self.agents):
                mu = float(ag.q_values[action])
                std = float(np.sqrt(ag.calculate_variance(action)))
                p = self.local_likelihood(reward, mu, std, delta_frac=3.0, min_delta=self.delta)
                p_obs.append(max(p * self.p_env[i], self.min_val))
            self.likelihood_history.append(p_obs.copy())
            # check if learning phase completed
            if self.learning_steps_taken >= self.cooldown_steps:
                self.learning_phase_active = False
                self.learning_agent_idx = -1
                self.learning_steps_taken = 0
                self.p_env = np.ones(len(self.agents)) / len(self.agents)
            return
        
        # outside learning phase: detect change by KL or likelihood
        change_detected = False
        if self.steps_since_last_add >= self.cooldown_steps:
            if self.detect_changepoint_kl(action):
                change_detected = True
        
        # fallback to likelihood-based detection if KL did not trigger
        # if not change_detected:
        #     if self.detect_changepoint_likelihood(action, reward):
        #         change_detected = True
        
        if change_detected:
            self.add_new_agent()
        
        # update all agents with weighted rewards
        p_obs = []
        for i, ag in enumerate(self.agents):
            ag.update(action, reward, delay, env_prob=self.p_env[i])
        
        # compute likelihoods and update p_env
        for i, ag in enumerate(self.agents):
            mu = float(ag.q_values[action])
            std = float(np.sqrt(ag.calculate_variance(action)))
            p = self.local_likelihood(reward, mu, std, delta_frac=3.0, min_delta=self.delta)
            p_obs.append(max(p * self.p_env[i], self.min_val))
        self.likelihood_history.append(p_obs.copy())
        
        # normalise p_env if more than one agent
        if len(self.agents) > 1:
            total = float(sum(p_obs))
            if total < self.min_val:
                self.p_env = np.ones(len(self.agents)) / len(self.agents)
            else:
                self.p_env = np.array(p_obs) / total
