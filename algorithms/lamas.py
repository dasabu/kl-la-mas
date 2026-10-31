"""
LA-MAS: Likelihood Adaptive Multi-Agent System.

Multi-agent system that detects environment changes using likelihood-based methods
and spawns new agents to adapt to changing environments.
"""

import numpy as np
from scipy.stats import norm


class LikelihoodAdaptiveMultiAgentSystem:
    """
    Likelihood Adaptive Multi-Agent System (LA-MAS).
    
    Maintains multiple bandit agents and uses likelihood-based changepoint detection
    to identify environment changes. When a changepoint is detected, spawns a new
    agent to adapt to the new environment while retaining knowledge from previous agents.
    """
    
    def __init__(
        self, 
        agent_type,
        agent_params,
        detection_threshold=0.00005, 
        delta=0.07,
        cooldown_steps=40, 
        min_data_points=40,
        min_val=1e-6
    ):
        """
        Initialize LA-MAS.
        
        Args:
            agent_type: Class of bandit agent to use (e.g., UCB1)
            agent_params: Dictionary of parameters for agent initialization
            detection_threshold: Threshold for low likelihood detection
            delta: Delta parameter for likelihood calculation
            cooldown_steps: Cooldown period after adding new agent
            min_data_points: Minimum data before changepoint detection
            min_val: Minimum value to avoid numerical issues
        """
        self.agent_type = agent_type
        self.agent_params = agent_params
        self.detection_threshold = detection_threshold
        self.delta = delta
        self.cooldown_steps = cooldown_steps
        self.min_val = min_val
        
        # Initialize with single agent
        self.agents = [agent_type(**agent_params)]
        self.p_env = np.array([1.0])
        
        # State tracking
        self.global_step = 0
        self.steps_since_last_add = cooldown_steps
        self.reward_history = []
        self.changepoints = []
        self.likelihood_history = []
        
        # Learning phase tracking
        self.min_data_points = min_data_points
        self.learning_agent_idx = -1
        self.learning_steps_taken = 0
        self.learning_phase_active = False
        self.low_likelihood_count = 0
        self.min_m = 10  # Minimum consecutive low likelihoods for changepoint

    def local_likelihood(self, reward, mu, sigma, delta_frac=2.0, min_delta=0.07):
        """
        Calculate local likelihood of observing reward given mean and std.
        
        Uses a tolerance window around the reward to account for noise.
        
        Args:
            reward: Observed reward
            mu: Expected mean
            sigma: Expected standard deviation
            delta_frac: Fraction of sigma for tolerance
            min_delta: Minimum tolerance
            
        Returns:
            Likelihood probability
        """
        delta = max(min_delta, delta_frac * sigma)
        return norm.cdf(reward + delta, mu, sigma) - norm.cdf(reward - delta, mu, sigma)

    def detect_changepoint(self, action, reward):
        """
        Detect if a changepoint has occurred.
        
        Checks if the observed reward has low likelihood under all current agents.
        Requires consecutive low likelihoods to avoid false positives.
        
        Args:
            action: Action taken
            reward: Observed reward
            
        Returns:
            True if changepoint detected, False otherwise
        """
        if len(self.reward_history) < self.min_data_points:
            return False
            
        likelihoods = []
        for ag in self.agents:
            mu = ag.q_values[action]
            std = np.sqrt(ag.calculate_variance(action))
            p = self.local_likelihood(reward, mu, std)
            likelihoods.append(p)
            
        # Check if maximum likelihood is below threshold
        self.low_likelihood_count += 1 if max(likelihoods) < self.detection_threshold else 0
        
        if self.low_likelihood_count >= self.min_m:
            self.low_likelihood_count = 0
            return True
            
        return False

    def add_new_agent(self):
        """
        Add a new agent to adapt to detected environment change.
        
        Initializes a new agent and enters learning phase where only the new
        agent is active for a cooldown period.
        """
        idx = len(self.agents)
        self.agents.append(self.agent_type(**self.agent_params))
        self.p_env = np.ones(len(self.agents)) / len(self.agents)
        self.changepoints.append(self.global_step)
        self.steps_since_last_add = 0
        self.learning_agent_idx = idx
        self.learning_steps_taken = 0
        self.learning_phase_active = True

    def select_action(self):
        """
        Select action using mixture of agents.
        
        During learning phase, only the new agent is active.
        Otherwise, agents vote weighted by their environment probabilities.
        
        Returns:
            Selected action index
        """
        if (self.learning_phase_active and self.learning_agent_idx != -1 and 
            self.learning_steps_taken < self.cooldown_steps):
            return self.agents[self.learning_agent_idx].select_action()
            
        actions = [ag.select_action() for ag in self.agents]
        return int(np.random.choice(actions, p=self.p_env))

    def update(self, action, reward, delay):
        """
        Update all agents and environment probabilities.
        
        Args:
            action: Action taken
            reward: Observed reward
            delay: Response time
        """
        self.global_step += 1
        self.steps_since_last_add += 1
        self.reward_history.append(reward)

        if self.learning_phase_active and self.learning_agent_idx != -1:
            # Learning phase: only update new agent
            self.learning_steps_taken += 1
            self.agents[self.learning_agent_idx].update(action, reward, delay, env_prob=1.0)
            
            # Calculate likelihoods for all agents
            p_obs = []
            for i, ag in enumerate(self.agents):
                mu = ag.q_values[action]
                std = np.sqrt(ag.calculate_variance(action))
                p = self.local_likelihood(reward, mu, std)
                p_obs.append(max(p * self.p_env[i], self.min_val))
            self.likelihood_history.append(p_obs.copy())
            
            # Exit learning phase after cooldown
            if self.learning_steps_taken >= self.cooldown_steps:
                self.learning_phase_active = False
                self.learning_agent_idx = -1
                self.learning_steps_taken = 0
                self.p_env = np.ones(len(self.agents)) / len(self.agents)
        else:
            # Normal operation: check for changepoint
            if self.detect_changepoint(action, reward):
                self.add_new_agent()
                
            # Update all agents weighted by environment probabilities
            for i, ag in enumerate(self.agents):
                ag.update(action, reward, delay, env_prob=self.p_env[i])
                
            # Update environment probabilities using Bayes rule
            p_obs = []
            for i, ag in enumerate(self.agents):
                mu = ag.q_values[action]
                std = np.sqrt(ag.calculate_variance(action))
                p = self.local_likelihood(reward, mu, std)
                p_obs.append(max(p * self.p_env[i], self.min_val))
            self.likelihood_history.append(p_obs.copy())
            
            # Normalize probabilities
            if len(self.agents) > 1:
                total = sum(p_obs)
                if total < self.min_val:
                    self.p_env = np.ones(len(self.agents)) / len(self.agents)
                else:
                    self.p_env = np.array(p_obs) / total
