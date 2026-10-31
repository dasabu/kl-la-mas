import numpy as np
from scipy.stats import norm
from .ucb import UCB1


class UCB1_multiple_C(UCB1):
    def __init__(self, add_points_functions, remove_points_function, c_list,
                 n_envs=4, n_actions=10, c_timestep_change=100,
                 verbose=False, c_min=0.01, c_max=10.0, var_window=50):
        super().__init__(c=c_list[0], n_actions=n_actions, var_window=var_window)
        self.c_timestep_change = c_timestep_change
        self.num_actions = n_actions
        self.n_envs = n_envs
        self.timestep = 0
        self.c_list = c_list
        self.c_ind = 0
        self.add_points_functions = add_points_functions
        self.remove_points_function = remove_points_function
        self.c_min = c_min
        self.c_max = c_max
        self.c = self.c_list[self.c_ind]
        self.c_history = []
        self.unique_c_history = []
        self.previous_c_values = []
        self.previous_c_scores = []
        self.bo_selection_log = []
        self.points_next_round = None
        self.scores_next_round = None
        self.verbose = verbose
        self.total_response_time = 0.0
        self.total_responses = 0
        self.average_response_time = 0.0
        self.total_prob = np.zeros(n_envs)
        self.total_obs = np.zeros(n_envs)
        
    def track_response_time(self, delay):
        self.total_response_time += delay
        self.total_responses += 1
        self.average_response_time = self.total_response_time / self.total_responses if self.total_responses else 0.0
        return self.average_response_time
    
    def update(self, chosen_act, reward, delay, p_env=None, env_prob=1.0, **kwargs):
        self.c_history.append(self.c)
        super().update(chosen_act, reward, delay, env_prob=env_prob)
        if p_env is None:
            p_env = np.ones(self.n_envs) / self.n_envs
        for i in range(self.n_envs):
            self.total_prob[i] += p_env[i]
            self.total_obs[i] += 1
        if self.timestep % self.c_timestep_change == 0:
            c_score = self.track_response_time(delay)
            if self.timestep < (self.c_timestep_change * len(self.c_list) - 1):
                self.previous_c_values.append(self.c)
                self.previous_c_scores.append(c_score)
                self.c_ind += 1
                self.c = self.c_list[self.c_ind % len(self.c_list)]
                self.unique_c_history.append(self.c)
                self.points_next_round = np.atleast_2d(self.previous_c_values).reshape(-1, 1)
                self.scores_next_round = np.atleast_2d(self.previous_c_scores).reshape(-1, 1)
            else:
                if self.points_next_round is None or self.scores_next_round is None:
                    self.points_next_round = np.atleast_2d([self.c])
                    self.scores_next_round = np.atleast_2d([c_score])
                else:
                    self.points_next_round = np.concatenate(
                        (self.points_next_round, np.atleast_2d([self.c]).reshape(-1, 1)))
                    self.scores_next_round = np.concatenate(
                        (self.scores_next_round, np.atleast_2d([c_score]).reshape(-1, 1)))
                points_next_round, scores_next_round = self.remove_points_function(
                    self.points_next_round, self.scores_next_round,
                    actions=self.num_actions, c_value=self.c,
                    current_cycle=self.timestep, current_iteration=self.timestep
                )
                self.points_next_round = points_next_round
                self.scores_next_round = scores_next_round
                found_points = []
                for i, add_func in enumerate(self.add_points_functions):
                    new_c_array = add_func(
                        self.points_next_round, self.scores_next_round,
                        actions=self.num_actions, c_min=self.c_min, c_max=self.c_max,
                        total_prob=self.total_prob[i], total_obs=max(self.total_obs[i], 1.0)
                    )
                    found_points.append(float(new_c_array[0]))
                    self.bo_selection_log.append(f"Env_{i}")
                if len(found_points) > 0:
                    self.c = min(found_points)
                    self.unique_c_history.append(self.c)
        self.timestep += 1

class MultiAgentSystem_BO:
    def __init__(self, agent_type, agent_params, n_envs=4, **kwargs):
        self.n_envs = n_envs
        self.agents = [agent_type(**agent_params) for _ in range(n_envs)]
        self.p_env = np.ones(n_envs) / n_envs
        self.min_val = 1e-6
    
    def update_environment_probabilities(self, action, reward, delta=0.07):
        p_observed = []
        for i, agent in enumerate(self.agents):
            mean_i = agent.q_values[action]
            var_i = agent.calculate_variance(action)
            std_i = max(np.sqrt(var_i), 1e-6)
            low, high = reward - delta, reward + delta
            p_o_given_ei = float(norm.cdf(high, loc=mean_i, scale=std_i) - norm.cdf(low, loc=mean_i, scale=std_i))
            val = max(p_o_given_ei * self.p_env[i], self.min_val)
            p_observed.append(val)
        total = float(np.sum(p_observed))
        if total < self.min_val:
            self.p_env[:] = 1.0 / self.n_envs
        else:
            self.p_env = np.array(p_observed) / total
            
    def select_action(self):
        return [agent.select_action() for agent in self.agents]
    
    def track_response_time(self, delay):
        return [agent.track_response_time(delay) for agent in self.agents]
    
    def update_q_values(self, action, reward, delay):
        for i, agent in enumerate(self.agents):
            agent.update(action, reward, delay, env_prob=self.p_env[i], p_env=self.p_env)
