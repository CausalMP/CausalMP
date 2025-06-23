import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from pathlib import Path
from .base_environment import BaseEnvironment
from .data_path_manager import DataPathManager

class NYCTaxiRoutesEnvironment(BaseEnvironment):
    """Environment for simulating NYC taxi route dynamics and interventions.
    
    This environment models ride-sharing dynamics across New York City taxi zones using 
    real-world TLC Trip Record Data. The framework adapts the established linear-in-mean 
    outcome model to capture the dynamics of ride-sharing services.
    
    Attributes:
        ENVIRONMENT_TYPE (str): Identifier for the environment type
        theta (float): Network autocorrelation parameter
        tau_population (float): Population-level treatment effect
        tau_unit_coeff (float): Coefficient for route-specific treatment effects
        warm_up_T (int): Number of warm-up periods before simulation
        adj_matrix (np.ndarray): Adjacency matrix representing route relationships
        baseline_outcomes (np.ndarray): Historical baseline trip counts
        treatment_effects (np.ndarray): Route-specific treatment effects
        adj_ones (np.ndarray): Precomputed adjacency matrix @ ones vector
        tau_unit (np.ndarray): Individual route treatment effects
    """
    
    ENVIRONMENT_TYPE = "NYC Taxi Routes"
    
    def __init__(self, params: Dict[str, Any], seed: Optional[int] = None):
        """Initialize the NYC taxi routes environment."""
        super().__init__(params, seed)
        
        # Extract environment-specific parameters
        self.theta = params.get("theta", 0.4)
        self.tau_population = params.get("tau_population", 0.2)
        self.tau_unit_coeff = params.get("tau_unit_coeff", 1)
        self.warm_up_T = params.get("warm_up_period", 24)
        
        # Validate required parameters
        if any(param is None for param in 
               [self.theta, self.tau_population, self.tau_unit_coeff, self.warm_up_T]):
            raise ValueError("Missing required parameters for NYC Taxi Routes environment")
            
        # Initialize attributes to None
        self.adj_matrix = None
        self.baseline_outcomes = None
        self.treatment_effects = None
        self.adj_ones = None
        self.tau_unit = None
        
    def initialize_environment(self) -> None:
        """Initialize environment with network structure and baseline data."""
        try:
            # Load and process adjacency matrix
            self._load_adjacency_matrix()
            
            # Load baseline outcomes
            self._load_baseline_outcomes()
            
            # Load treatment effects
            self._load_treatment_effects()
            
            # Precompute frequently used values
            self.adj_ones = self.adj_matrix @ np.ones(self.N)
            self.tau_unit = self.tau_unit_coeff * self.treatment_effects[:self.N]
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Failed to load NYC Taxi Routes data: {e}")
            
    def generate_outcomes(self, W: np.ndarray) -> np.ndarray:
        """Generate outcomes for NYC taxi routes model."""
        # Initialize outcome matrices
        Y = np.zeros((self.N, self.T))
        Y_WU = np.zeros(self.N)
        
        # Run warm-up period
        for t in range(1, self.warm_up_T):
            baseline_t = self.baseline_outcomes[:, t]
            Y_WU[:] = baseline_t
            
        # Set initial state
        Y[:, 0] = Y_WU[:]
        
        # Main simulation
        for t in range(1, self.T):
            W_t = W[:, t]
            A_Y_prev = self.adj_matrix @ Y[:, t-1]
            A_baseline_prev = self.adj_matrix @ self.baseline_outcomes[:, t + self.warm_up_T - 1]
            baseline_t = self.baseline_outcomes[:, t + self.warm_up_T]
            
            # Update outcomes using safe division
            Y[:, t] = (
                baseline_t + 
                self.theta * (
                    self._safe_division(A_Y_prev, self.adj_ones)
                    - self._safe_division(A_baseline_prev, self.adj_ones)
                ) +
                self.tau_population * self._safe_division(
                    self.adj_matrix @ W_t, self.adj_ones
                ) + 
                self.tau_unit * W_t
            )
            
        return Y
    
    def _load_adjacency_matrix(self) -> None:
        """Load and process the route relationship network."""
        try:
            edges_df = pd.read_csv(
                DataPathManager.get_environment_data_path('NYC_taxi_routes', '') / 'zones_relationship_edges.csv'
            )
            
            # Initialize and fill adjacency matrix
            self.adj_matrix = np.zeros((self.N, self.N))
            mask = (edges_df['source'] < self.N) & (edges_df['target'] < self.N)
            filtered_edges = edges_df[mask]
            
            self.adj_matrix[
                filtered_edges['source'].values,
                filtered_edges['target'].values
            ] = 1
            
        except FileNotFoundError:
            raise FileNotFoundError("NYC taxi routes network data not found")
            
    def _load_baseline_outcomes(self) -> None:
        """Load historical baseline trip counts."""
        try:
            df = pd.read_csv(
                DataPathManager.get_environment_data_path('NYC_taxi_routes', '') / 'baseline_outcomes.csv'
            )
            self.baseline_outcomes = df.to_numpy()
            
            if self.baseline_outcomes.shape[0] < self.N:
                raise ValueError(
                    f"Insufficient baseline data: got {self.baseline_outcomes.shape[0]} "
                    f"routes, need {self.N}"
                )
                
        except FileNotFoundError:
            raise FileNotFoundError("NYC taxi routes baseline data not found")
            
    def _load_treatment_effects(self) -> None:
        """Load route-specific treatment effects."""
        try:
            df = pd.read_csv(
                DataPathManager.get_environment_data_path('NYC_taxi_routes', '') / 'direct_treatment_effect.csv'
            )
            self.treatment_effects = df.to_numpy().reshape(-1)
            
            if self.treatment_effects.shape[0] < self.N:
                raise ValueError(
                    f"Insufficient treatment effects data: got "
                    f"{self.treatment_effects.shape[0]} routes, need {self.N}"
                )
                
        except FileNotFoundError:
            raise FileNotFoundError("NYC taxi routes treatment effects data not found")
            
    @staticmethod
    def _safe_division(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Safely divide arrays, handling zero denominators."""
        return np.divide(
            a, b, 
            out=np.zeros_like(a, dtype=float), 
            where=b!=0
        )