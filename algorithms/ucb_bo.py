"""
UCB with Bayesian Optimization for adaptive hyperparameter tuning.
"""

import numpy as np
from .ucb import UCB1


class UCB1_with_BO_Optimization(UCB1):
    """
    UCB1 enhanced with Bayesian Optimization for automatic tuning of exploration parameter c.
    
    The algorithm starts by cycling through a predefined set of c values, then uses
    Gaussian Process-based Bayesian Optimization to find optimal c values over time.
    """
    
    def __init__(self, add_points_functions, remove_points_function, c_list, 
                 n_actions=10, var_window=50, c_timestep_change=100, 
                 verbose=False, c_min=None, c_max=None):
        """
        Initialize UCB1 with BO.
        
        Args:
            add_points_functions: List of functions to add new points
            remove_points_function: Function to remove old points
            c_list: Initial list of c values to try
            n_actions: Number of actions
            var_window: Window for variance calculation
            c_timestep_change: How often to update c
            verbose: Print debug information
            c_min: Minimum c value
            c_max: Maximum c value
        """
        super().__init__(c=c_list[0], n_actions=n_actions, var_window=var_window)
        
        self.c_timestep_change = c_timestep_change
        self.timestep = 0
        self.c_list = c_list
        self.c_ind = 0
        
        # BO functions
        self.add_points_functions = (
            add_points_functions if isinstance(add_points_functions, list) 
            else [add_points_functions]
        )
        self.remove_points_function = remove_points_function
        
        self.c_min = c_min or 0.1
        self.c_max = c_max or 10.0
        self.c = self.c_list[self.c_ind]
        
        # Logging
        self.c_history = []
        self.unique_c_history = []
        self.previous_c_values = []
        self.previous_c_scores = []
        self.bo_selection_log = []
        
        # BO data storage
        self.points_next_round = None
        self.scores_next_round = None
        self.verbose = verbose
        
        # Environment tracking for BO
        self.total_prob = 0.0
        self.total_obs = 0

    def update(self, action, reward, delay=0, env_prob=1.0, **kwargs):
        """
        Update after observing reward, including BO-based c tuning.
        
        Args:
            action: Action taken
            reward: Observed reward
            delay: Response time
            env_prob: Environment probability weight
            **kwargs: Additional arguments
        """
        # Record c value
        self.c_history.append(self.c)
        
        # Standard UCB1 update
        super().update(action, reward, delay, env_prob, **kwargs)
        
        # Accumulate environment probabilities for BO
        self.total_prob += env_prob
        self.total_obs += 1
        
        # Update c parameter using BO
        if self.timestep % self.c_timestep_change == 0:
            c_score = self.track_response_time(delay)
            
            if self.timestep < (self.c_timestep_change * len(self.c_list) - 1):
                # Initial cycling through c_list
                self.previous_c_values.append(self.c)
                self.previous_c_scores.append(c_score)
                
                self.c_ind += 1
                self.c = self.c_list[self.c_ind % len(self.c_list)]
                self.unique_c_history.append(self.c)
                
                self.points_next_round = np.atleast_2d(self.previous_c_values).reshape(-1, 1)
                self.scores_next_round = np.atleast_2d(self.previous_c_scores).reshape(-1, 1)
            else:
                # Bayesian optimization phase
                if self.points_next_round is None or self.scores_next_round is None:
                    self.points_next_round = np.atleast_2d([self.c])
                    self.scores_next_round = np.atleast_2d([c_score])
                else:
                    self.points_next_round = np.concatenate(
                        (self.points_next_round, np.atleast_2d([self.c]).reshape(-1, 1))
                    )
                    self.scores_next_round = np.concatenate(
                        (self.scores_next_round, np.atleast_2d([c_score]).reshape(-1, 1))
                    )
                    
                # Remove oldest points
                points_next_round, scores_next_round = self.remove_points_function(
                    self.points_next_round, self.scores_next_round,
                    actions=self.n_actions, c_value=self.c,
                    current_cycle=self.timestep, current_iteration=self.timestep
                )
                self.points_next_round = points_next_round
                self.scores_next_round = scores_next_round

                # Use BO to find new c
                found_points = []
                for add_func in self.add_points_functions:
                    new_c_array = add_func(
                        self.points_next_round, self.scores_next_round,
                        actions=self.n_actions, c_min=self.c_min, c_max=self.c_max,
                        total_prob=self.total_prob, total_obs=self.total_obs
                    )
                    found_points.append(new_c_array[0])

                # Select best c (minimum for optimization)
                self.c = min(found_points) if found_points else self.c
                self.unique_c_history.append(self.c)

        self.timestep += 1
