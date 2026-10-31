"""
Gaussian Process-based Bayesian Optimization for hyperparameter tuning.
"""

import numpy as np
from scipy.spatial.distance import cdist


def rbf_kernel(xa, xb, sigma=1.0):
    """
    Radial Basis Function (RBF) kernel.
    
    Args:
        xa: First set of points
        xb: Second set of points
        sigma: Length scale parameter
        
    Returns:
        Kernel matrix
    """
    sq_norm = -0.5 * cdist(xa, xb, 'sqeuclidean') / sigma**2
    return np.exp(sq_norm)


class PillarHolder:
    """
    Manages pillar points for stable Bayesian Optimization.
    
    Pillar points are strategically sampled points that are always retained
    to ensure stable GP predictions.
    """
    
    def __init__(self, num_pillar_points):
        """
        Initialize pillar holder.
        
        Args:
            num_pillar_points: Number of pillar points to maintain
        """
        self.num_pillar_points = num_pillar_points
        self.indices_ready = False
        self.indices = None
        self.non_indices = None

    def resample_pillar_indices(self, point_set):
        """
        Select pillar point indices from the current point set.
        
        Pillar points are evenly distributed across the sorted point set.
        
        Args:
            point_set: Current set of points
            
        Returns:
            Tuple of (pillar_indices, non_pillar_indices)
        """
        sorted_indices = np.argsort(point_set)
        sorted_point_set = point_set[sorted_indices]
        interval_size = len(sorted_point_set) // self.num_pillar_points
        indices = []

        for i in range(self.num_pillar_points):
            index_in_interval = i * interval_size
            if index_in_interval < len(sorted_point_set):
                indices.append(sorted_indices[index_in_interval])

        non_indices = [i for i in range(len(point_set)) if i not in indices]
        self.indices = np.array(indices)
        self.non_indices = np.array(non_indices)
        self.indices_ready = True
        
        return self.indices, self.non_indices


class GPWithPillarReRun2:
    """
    Gaussian Process for Bayesian Optimization with pillar point strategy.
    
    Alternates between sampling pillar points (for stability) and 
    optimizing acquisition function (for exploration/exploitation).
    """
    
    def __init__(self, pillar_holder, x_iteration=5, num_pillar_points=3, 
                 noise_level=0.05, sigma=1.0):
        """
        Initialize GP optimizer.
        
        Args:
            pillar_holder: PillarHolder instance
            x_iteration: Frequency of pillar point resampling
            num_pillar_points: Number of pillar points
            noise_level: Observation noise
            sigma: RBF kernel length scale
        """
        self.noise_level = noise_level
        self.sigma = sigma
        self.K_train_inv = None
        self.point_set = np.empty((0, 1))
        self.point_scores = np.empty((0,))
        self.next_sample_point = None
        self.acquisition_function_values = None
        self.c_min = 0.01
        self.c_max = 10.0
        
        self.x_iteration = x_iteration
        self.num_pillar_points = num_pillar_points
        self.current_iteration = 0
        self.pillar_point_index = 0
        self.pillar_points = None
        self.pillar_points_values = None
        self.pillar_holder = pillar_holder

    def fit_model(self, point_set, point_scores):
        """
        Fit GP model to observed data.
        
        Args:
            point_set: Observed points (hyperparameter values)
            point_scores: Observed scores (performance metrics)
        """
        point_set = np.atleast_2d(point_set).reshape(-1, 1)
        point_scores = np.atleast_1d(point_scores)
        K_train = (
            rbf_kernel(point_set, point_set, self.sigma) + 
            self.noise_level**2 * np.eye(len(point_set))
        )
        K_train_inv = np.linalg.inv(K_train)
        self.K_train_inv = K_train_inv
        self.point_set = point_set
        self.point_scores = point_scores

    def predict(self, X_test):
        """
        Make GP predictions at test points.
        
        Args:
            X_test: Test points
            
        Returns:
            Tuple of (posterior_mean, posterior_std)
        """
        X_test = np.atleast_2d(X_test).reshape(-1, 1)
        K_train_test = rbf_kernel(self.point_set, X_test, self.sigma)
        K_test = rbf_kernel(X_test, X_test, self.sigma)
        posterior_mean = K_train_test.T @ self.K_train_inv @ self.point_scores
        posterior_covariance = K_test - K_train_test.T @ self.K_train_inv @ K_train_test
        return posterior_mean.ravel(), np.sqrt(np.diag(posterior_covariance))
    
    def dynamic_range_adjustment(self, point_set, point_scores, 
                                 initial_expansion_factor=1.0, decay_rate=0.95):
        """
        Dynamically adjust search range based on observed data.
        
        Uses interquartile range (IQR) to focus search on promising regions.
        
        Args:
            point_set: Observed points
            point_scores: Observed scores
            initial_expansion_factor: Initial range expansion
            decay_rate: Decay rate for expansion over iterations
            
        Returns:
            Tuple of (new_c_min, new_c_max)
        """
        q1 = np.percentile(point_scores, 25)
        q3 = np.percentile(point_scores, 75)
        iqr_points = point_set[(point_scores >= q1) & (point_scores <= q3)]

        if len(iqr_points) > 0:
            iqm = np.mean(iqr_points)
            iqr_variance = np.var(iqr_points)
            dynamic_expansion_factor = min(1.5, max(0.1, 1.0 / (1.0 + iqr_variance)))
            iteration_expansion_factor = initial_expansion_factor * (decay_rate ** self.current_iteration)
            final_expansion_factor = dynamic_expansion_factor * iteration_expansion_factor
            new_c_min = max(0, iqm - final_expansion_factor * np.ptp(iqr_points))
            new_c_max = iqm + final_expansion_factor * np.ptp(iqr_points)
        else:
            new_c_min, new_c_max = self.c_min, self.c_max
            
        return new_c_min, new_c_max

    def select_points_GP(self, point_set, point_scores, total_prob, total_obs, **kwargs):
        """
        Select next point to evaluate using GP-based acquisition function.
        
        Alternates between pillar point sampling and Lower Confidence Bound (LCB) optimization.
        
        Args:
            point_set: Observed points
            point_scores: Observed scores
            total_prob: Total environment probability (for normalization)
            total_obs: Total observations (for normalization)
            **kwargs: Additional arguments
            
        Returns:
            Next point to evaluate
        """
        self.fit_model(point_set, point_scores)
        self.c_min, self.c_max = self.dynamic_range_adjustment(point_set, point_scores)
        
        # Every x_iteration steps, sample a pillar point
        if self.current_iteration % self.x_iteration == 0:
            if self.pillar_points is None:
                self.pillar_indices, self.non_pillar_indices = (
                    self.pillar_holder.resample_pillar_indices(point_set)
                )
                self.pillar_points = point_set[self.pillar_indices]
                self.pillar_points_values = point_scores[self.pillar_indices]

            new_point = self.pillar_points[self.pillar_point_index]
            self.pillar_point_index = (self.pillar_point_index + 1) % self.num_pillar_points
        else:    
            # Optimize acquisition function (LCB)
            new_points = np.random.uniform(self.c_min, self.c_max, 500).reshape(-1, 1)
            new_points_predictions, new_std = self.predict(new_points)
            adjusted_predictions = (
                new_points_predictions / max(total_prob, 1e-6) * max(total_obs, 1)
            )
            kappa = 3.5
            lcb = adjusted_predictions - kappa * new_std
            self.acquisition_function_values = lcb
            ind_next_sample = np.argmin(lcb)
            self.next_sample_point = new_points[ind_next_sample]
            new_point = new_points[ind_next_sample]
            
        new_point_score, _ = self.predict(new_point)
        
        # Update pillar points if needed
        if self.pillar_points is None:
            self.pillar_points = point_set[self.pillar_indices]
            self.pillar_points_values = point_scores[self.pillar_indices]
        elif new_point in self.pillar_points:
            index = np.where(self.pillar_points == new_point)[0][0]
            self.pillar_points_values[index] = new_point_score
        
        self.current_iteration += 1
        return np.array([new_point])


class RemoveOldestNonPillarClass:
    """
    Removes oldest non-pillar points to maintain a manageable dataset size.
    
    Pillar points are always retained for stability.
    """
    
    def __init__(self, pillar_holder):
        """
        Initialize remover.
        
        Args:
            pillar_holder: PillarHolder instance
        """
        self.pillar_holder = pillar_holder
        
    def remove_oldest_non_pillar(self, point_set, all_points_scores, 
                                 num_pillar_points=3, removal_interval=4, 
                                 current_iteration=0, **kwargs):
        """
        Remove oldest non-pillar point at specified intervals.
        
        Args:
            point_set: Current point set
            all_points_scores: Current scores
            num_pillar_points: Number of pillar points
            removal_interval: Frequency of removal
            current_iteration: Current iteration
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (new_point_set, new_scores)
        """
        pillar_holder = self.pillar_holder
        
        if not pillar_holder.indices_ready:
            indices = np.arange(len(point_set))[:3]
            non_indices = np.arange(len(point_set))[3:]
        else:
            indices = pillar_holder.indices
            non_indices = pillar_holder.non_indices
            
        pillar_points = point_set[indices]
        non_pillar_points = point_set[non_indices]
        pillar_scores = all_points_scores[indices]
        non_pillar_scores = all_points_scores[non_indices]

        # Remove oldest non-pillar point at intervals
        if current_iteration % removal_interval == 0 and len(non_pillar_points) > 0:
            non_pillar_points = non_pillar_points[1:]
            non_pillar_scores = non_pillar_scores[1:]

        new_point_set = np.concatenate([
            np.atleast_2d(pillar_points).reshape(-1, 1), 
            np.atleast_2d(non_pillar_points).reshape(-1, 1)
        ])
        new_scores = np.concatenate([
            np.atleast_2d(pillar_scores).reshape(-1, 1), 
            np.atleast_2d(non_pillar_scores).reshape(-1, 1)
        ])
        
        return new_point_set, new_scores
