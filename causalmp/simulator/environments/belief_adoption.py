import numpy as np
import pandas as pd
from scipy import sparse
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
from .base_environment import BaseEnvironment
from .data_path_manager import DataPathManager

class BeliefAdoptionEnvironment(BaseEnvironment):
    """Environment for simulating belief adoption in social networks.
    
    This environment models the diffusion of competing opinions within interconnected 
    communities, implementing the cascade model. The system examines how opinions 
    spread through social networks when individuals make decisions through 
    coordination games with their neighbors.
    
    Attributes:
        ENVIRONMENT_TYPE (str): Identifier for the environment type
        tau (float): Base treatment effect coefficient
        q (float): Initial proportion of nodes with Opinion B
        beta (float): Intensity of social influence 
        warm_up_T (int): Number of warm-up periods before simulation
        adj_matrix (sparse.csr_matrix): Adjacency matrix representing network connections
        profiles (pd.DataFrame): User demographic profiles
        base_values (pd.DataFrame): Base payoff values and treatment multipliers
        neighbors (List[np.ndarray]): Precomputed neighbor indices for each node
        neighbor_counts (np.ndarray): Number of neighbors for each node
        initial_behaviors (np.ndarray): Initial opinion states
        payoffs_a (Dict[int, float]): Payoff values for Opinion A
        payoffs_b (Dict[int, float]): Payoff values for Opinion B
        tau_individual (np.ndarray): Individual-specific treatment effects
    """
    
    ENVIRONMENT_TYPE = "Belief Adoption Model"
    
    def __init__(self, params: Dict[str, Any], seed: Optional[int] = None):
        """Initialize the belief adoption environment."""
        super().__init__(params, seed)
        
        self.tau = params.get("tau", 1)
        self.q = params.get("q", 0.4)
        self.beta = params.get("beta", 10)
        self.warm_up_T = params.get("warm_up_period", 100)
        
        # Initialize attributes to None
        self.adj_matrix = None
        self.profiles = None
        self.base_values = None
        self.neighbors = None
        self.neighbor_counts = None
        self.initial_behaviors = None
        self.payoffs_a = None
        self.payoffs_b = None
        self.tau_individual = None
        
    def initialize_environment(self) -> None:
        """Initialize network structure, user profiles, and initial behaviors."""
        # Network data mapping for real networks
        NETWORK_DATA = {
            3366: ("Krupina", "krupina"),
            18246: ("Topolcany", "topolcany"),
            42971: ("Zilina", "zilina")
        }

        # Load real network data if available, otherwise generate synthetic
        if self.N in NETWORK_DATA:
            self._load_real_network(NETWORK_DATA[self.N])
        else:
            self._generate_synthetic_network()

        self._precompute_neighbors()

        # Initialize behaviors with specified proportion of Opinion B
        behaviors = np.ones(self.N)
        num_b = int(self.N * self.q)
        b_indices = self.rng.choice(self.N, size=num_b, replace=False)
        behaviors[b_indices] = -1
        self.initial_behaviors = behaviors.copy()

        # Store payoffs and treatment effects
        self.payoffs_a = self.base_values['payoff_a'].to_dict()
        self.payoffs_b = self.base_values['payoff_b'].to_dict()
        self.tau_individual = self.tau * self.base_values['treatment_multiplier'].values
            
    def generate_outcomes(self, W: np.ndarray) -> np.ndarray:
        """Generate outcomes for belief adoption model."""
        # Create local RNG with fixed seed for reproducibility
        local_rng = np.random.default_rng(self.seed)

        # Start with stored initial behaviors
        behaviors = self.initial_behaviors.copy()
        behaviors_dict = {i: behaviors[i] for i in range(self.N)}

        # Initialize outcomes matrix
        Y = np.zeros((self.N, self.T), dtype=np.int8)

        # Run warmup period with fixed seed
        for _ in range(self.warm_up_T):
            update_order = local_rng.permutation(self.N)
            for v in update_order:
                behaviors_dict[v] = self._update_behavior(
                    v, 
                    self.payoffs_a[v], 
                    self.payoffs_b[v], 
                    behaviors_dict,
                    local_rng
                )

        # Main simulation
        for t in range(self.T):
            # Apply treatments and update payoffs
            treated_nodes = np.where(W[:, t])[0]
            current_payoffs_a = self.payoffs_a.copy()
            for v in treated_nodes:
                current_payoffs_a[v] += self.tau_individual[v]

            # Update behaviors
            update_order = local_rng.permutation(self.N)
            for v in update_order:
                behaviors_dict[v] = self._update_behavior(
                    v, 
                    current_payoffs_a[v], 
                    self.payoffs_b[v], 
                    behaviors_dict,
                    local_rng
                )
                Y[v, t] = 1 if behaviors_dict[v] == 1 else 0

        return Y
            
    def _load_real_network(self, network_info: Tuple[str, str]) -> None:
        """Load real network data from files."""
        city_name, _ = network_info
        data_path = DataPathManager.get_environment_data_path('belief_adoption_model', '')
    
        try:
            # Load base values
            self.base_values = pd.read_csv(
                data_path / f'{city_name}_base_values.txt',
                sep='\t',
                comment='#',
                header=None,
                names=['payoff_a', 'payoff_b', 'treatment_multiplier']
            )
            
            # Load user profiles and relationships
            self.profiles = pd.read_csv(data_path / f'{city_name}_profiles.txt')
            relationships = pd.read_csv(data_path / f'{city_name}_relationships.txt')
            
            # Create user index mapping
            unique_users = self.profiles['user_id'].unique()
            user_to_index = {user: idx for idx, user in enumerate(unique_users)}
            
            # Create adjacency matrix
            rows = [user_to_index[src] for src in relationships['source']]
            cols = [user_to_index[tgt] for tgt in relationships['target']]
            data = np.ones(len(rows), dtype=np.int8)
            
            self.adj_matrix = sparse.csr_matrix(
                (data, (rows, cols)), 
                shape=(self.N, self.N)
            )
            
            # Update profile indices
            self.profiles['index'] = self.profiles['user_id'].map(user_to_index)
            self.profiles = self.profiles.sort_values('index')
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing data file for {city_name}: {e}")
            
    def _generate_synthetic_network(self) -> None:
        """Generate synthetic network when real data is unavailable."""
        # Generate random sparse network
        density = 0.01
        n_edges = int(self.N * self.N * density)
        
        rows = self.rng.integers(0, self.N, size=n_edges)
        cols = self.rng.integers(0, self.N, size=n_edges)
        data = np.ones(n_edges, dtype=np.int8)
        
        self.adj_matrix = sparse.csr_matrix(
            (data, (rows, cols)), 
            shape=(self.N, self.N)
        )
        
        try:
            # Load Zilina base values and for synthetic network
            data_path = DataPathManager.get_environment_data_path('belief_adoption_model', '')
            self.base_values = pd.read_csv(
                data_path / 'Zilina_base_values.txt',
                sep='\t',
                comment='#',
                header=None,
                names=['payoff_a', 'payoff_b', 'treatment_multiplier']
            ).head(self.N)
            # Load sample  and generate synthetic values
            self.profiles = pd.read_csv(
                data_path / 'Zilina_profiles.txt'
            ).head(self.N)
            self.profiles['index'] = np.arange(self.N)

        except FileNotFoundError:
            raise FileNotFoundError("Sample profiles not found for synthetic network")
            
    def _precompute_neighbors(self) -> None:
        """Precompute neighbor relationships for efficient access during simulation."""
        self.neighbors = [
            np.where(self.adj_matrix[i].toarray().flatten() != 0)[0]
            for i in range(self.N)
        ]
        self.neighbor_counts = np.array([len(n) for n in self.neighbors])
        
    def _update_behavior(self, 
                        v: int, 
                        payoff_a: float, 
                        payoff_b: float, 
                        behaviors_dict: Dict[int, int],
                        rng: np.random.Generator) -> int:
        """Update behavior for a single node based on payoffs and neighbor influence."""
        if self.neighbor_counts[v] > 0:
            h = (payoff_a - payoff_b) / (payoff_a + payoff_b)
            neighbor_influence = sum(behaviors_dict[n] for n in self.neighbors[v])
            beta_term = self.beta * (h * self.neighbor_counts[v] + neighbor_influence)

            # Handle large beta_term values to avoid overflow
            if beta_term > 35:  # exp(35) is about 1.6e15
                p_choose_a = 1.0
            elif beta_term < -35:
                p_choose_a = 0.0
            else:
                p_choose_a = 1 / (1 + np.exp(-2 * beta_term))

            return 1 if rng.random() < p_choose_a else -1
        return behaviors_dict[v]