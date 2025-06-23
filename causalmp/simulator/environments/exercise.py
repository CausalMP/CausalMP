import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from pathlib import Path
from .base_environment import BaseEnvironment
from .data_path_manager import DataPathManager

class ExerciseEnvironment(BaseEnvironment):
    """Environment for simulating an exercise encouragement program with social network effects.
    
    This environment models how digital encouragement messages influence exercise decisions
    within a social network context. The model combines individual characteristics from
    Census Bureau data with social network effects from Twitter social circles data.
    
    Attributes:
        ENVIRONMENT_TYPE (str): Identifier for the environment type
        tau_unit_coeff (float): Coefficient for individual treatment effects
        theta (float): Coefficient for peer influence effect
        lambda_param (float): Coefficient for interaction between treatment and peer effects
        warm_up_T (int): Number of warm-up periods before simulation starts
        adj_matrix (np.ndarray): Adjacency matrix representing social connections
        baseline_probs (np.ndarray): Base exercise probabilities over weekly cycle
        treatment_effects (np.ndarray): Individual treatment effects over weekly cycle
        eta (float): Sensitivity to neighbor outcome variance (higher values mean more sensitive)
    """
    
    ENVIRONMENT_TYPE = "Exercise Encouragement Program"
    
    def __init__(self, params: Dict[str, Any], seed: Optional[int] = None):
        """Initialize the exercise encouragement environment."""
        super().__init__(params, seed)
        
        # Extract parameters specific to exercise environment
        self.tau_unit_coeff = params.get("tau", 1)
        self.theta = params.get("theta", 0.04)
        self.lambda_param = params.get("lambda", 0.01)
        self.warm_up_T = params.get("warm_up_period", 7)
        self.eta = params.get("eta", 0.02)  
        
        # Initialize matrices to None
        self.adj_matrix = None
        self.baseline_probs = None
        self.treatment_effects = None
        
    def initialize_environment(self) -> None:
        """Initialize environment with network structure and probability matrices."""
        # Load and process network data
        try:
            edges_df = pd.read_csv(DataPathManager.get_environment_data_path('exercise_encouragement_program', '') / 'Twitter_edges.csv')
            
            # Filter edges to match environment size
            mask = (edges_df['source_id'] < self.N) & (edges_df['target_id'] < self.N)
            filtered_edges = edges_df[mask]
            
            # Create adjacency matrix
            self.adj_matrix = np.zeros((self.N, self.N))
            sources = filtered_edges['source_id'].values
            targets = filtered_edges['target_id'].values
            self.adj_matrix[sources, targets] = 1
            
            # Load exercise probabilities
            probs_df = pd.read_csv(DataPathManager.get_environment_data_path('exercise_encouragement_program', '') / 'exercise_probability.csv')
            
            # Extract baseline and treatment effect probabilities
            self.baseline_probs = probs_df.iloc[:self.N, -14:-7].to_numpy()
            raw_effects = probs_df.iloc[:self.N, -7:].to_numpy()
            self.treatment_effects = self.tau_unit_coeff * raw_effects
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Required data file not found: {e}")
            
    def _compute_neighbor_variance(self, Y_prev: np.ndarray) -> np.ndarray:
        """Compute the variance of neighbors' outcomes.
        
        Args:
            Y_prev: Previous outcomes for all individuals
            
        Returns:
            Array of variance values for each individual's neighbors
        """
        # For each individual, compute the mean of their neighbors' outcomes
        neighbor_means = np.zeros(self.N)
        neighbor_variances = np.zeros(self.N)
        
        for i in range(self.N):
            neighbors = np.where(self.adj_matrix[i] > 0)[0]
            if len(neighbors) > 0:
                neighbor_means[i] = np.mean(Y_prev[neighbors])
                # Calculate variance (sum of squared differences divided by count)
                neighbor_variances[i] = np.sum((Y_prev[neighbors] - neighbor_means[i])**2) / len(neighbors)
            # For individuals with no neighbors, variance is 0
        
        return neighbor_variances

    def generate_outcomes(self, W: np.ndarray) -> np.ndarray:
        """Generate outcomes for the exercise encouragement program."""
        # Create local RNG with fixed seed for reproducibility
        local_rng = np.random.default_rng(self.seed)
        
        # Create extended matrices including warm-up period
        W_extended = np.concatenate([
            np.zeros((self.N, self.warm_up_T)),  # No treatment during warm-up
            W
        ], axis=1)
        
        Y_extended = np.zeros((self.N, self.T + self.warm_up_T))
        Y_extended[:, 0] = self.baseline_probs[:, 0]  # Initialize with day 0 baseline
                
        # Generate outcomes including warm-up period
        for t in range(1, self.warm_up_T + self.T):
            # Get daily baseline and treatment effects (cycling through week)
            day_idx = t % 7
            baseline_t = self.baseline_probs[:, day_idx]
            tau_unit_t = self.treatment_effects[:, day_idx]
            
            # Calculate peer influence (number of exercising neighbors)
            Z = self.adj_matrix @ Y_extended[:, t-1]
            
            # Calculate neighbor variance
            neighbor_variance = self._compute_neighbor_variance(Y_extended[:, t-1])
            
            # Calculate exercise probabilities
            logits = (baseline_t + 
                     tau_unit_t * W_extended[:, t] + 
                     self.theta * Y_extended[:, t-1] * Z + 
                     self.lambda_param * W_extended[:, t] * Y_extended[:, t-1] * Z -
                     self.eta * neighbor_variance)  # Subtraction because higher variance should reduce likelihood
            
            # Apply sigmoid function
            probs = 1 / (1 + np.exp(-logits))
            
            # Generate random decisions
            Y_extended[:, t] = local_rng.random(self.N) < probs
        
        # Return only post-warm-up outcomes
        return Y_extended[:, self.warm_up_T:]