# --------------------- 2. Environment ---------------------
import numpy as np


class DynamicEnvOrdered:
    def __init__(self, mus_all, sigmas_all, change_every):
        self.mus_all = mus_all
        self.sigmas_all = sigmas_all
        self.change_every = change_every
        self.timestep = 0
        self.env_key = -1
        self.reselect_env()
    
    def reselect_env(self):
        """
        Change to the next env at every self.change_every step
        """
        # Change to next env in order (0 -> 1 -> 2 -> 3 -> 0...)
        self.env_key = (self.env_key + 1) % len(self.mus_all)
        self.mus = self.mus_all[self.env_key]       # 38 means of 38 actions in the next env
        self.sigmas = self.sigmas_all[self.env_key] # 38 std of 38 actions in the next env
        
    def sample(self, action):
        """
        Sample a response time for the given action
        """
        mu, sigma = self.mus[action], self.sigmas[action]
        return np.random.normal(mu, sigma)
    
    def best_action(self):
        """
        Select the best action in current env, which has the smallest value of mean
        """
        return int(np.argmin(self.mus))
    
    def step(self):
        """
        Increase a time step
        """
        self.timestep += 1
        if self.timestep % self.change_every == 0:
            self.reselect_env()
    
    def current_env(self):
        return self.env_key


class Env:
    """For pretrained phase of each agent in DAMAS"""
    def __init__(self, mus, sigmas):
        self.mus = mus
        self.sigmas = sigmas
        self.env_key = 0
    def sample(self, action):
        mu, sigma = self.mus[action], self.sigmas[action]
        return float(np.random.normal(mu, sigma))
    def step(self, **kwargs):
        pass
    def best_action(self):
        return int(np.argmin(self.mus))
