import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any, ClassVar

class BaseEnvironment(ABC):
    """Base class for all experimental environments.
    
    This abstract base class defines the interface and common functionality
    that all simulation environments must implement.
    
    Attributes:
        ENVIRONMENT_TYPE (ClassVar[str]): String identifier for the environment type
        N (int): Number of units
        T (int): Number of time periods
        stage_time_blocks (List[int]): Time points where treatment regime changes
        probs (List[float]): Treatment probabilities for each regime
        desired_probs_1 (List[float]): First desired treatment regime probabilities
        desired_probs_2 (List[float]): Second desired treatment regime probabilities
        desired_stage_time_blocks (List[int]): Time points for desired treatment regimes
        desired_T (int): Number of time periods for desired scenarios
        seed (Optional[int]): Random seed for reproducibility
        rng (np.random.Generator): Random number generator instance
    """
    
    ENVIRONMENT_TYPE: ClassVar[str] = ""
    
    def __init__(self, params: Dict[str, Any], seed: Optional[int] = None):
        """Initialize base environment with common parameters.
        
        Args:
            params: Dictionary containing environment parameters including:
                - N: Number of units
                - stage_time_blocks: Time points for regime changes
                - design: Treatment probabilities
                - desired_design_1: First desired treatment regime
                - desired_design_2: Second desired treatment regime
                - desired_stage_time_blocks: Time points for desired regimes
            seed: Random seed for reproducibility
        """
        self.N = params["N"]
        self.stage_time_blocks = params["stage_time_blocks"]
        self.T = self.stage_time_blocks[-1]
        self.probs = params["design"]
        self.desired_probs_1 = params["desired_design_1"]
        self.desired_probs_2 = params["desired_design_2"]
        self.desired_stage_time_blocks = params["desired_stage_time_blocks"]
        self.desired_T = self.desired_stage_time_blocks[-1]
        self.params = params
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    @abstractmethod
    def generate_outcomes(self, W: np.ndarray) -> np.ndarray:
        """Generate outcomes based on treatment assignment.
        
        This abstract method must be implemented by each environment to define
        how outcomes are generated given a treatment assignment matrix.
        
        Args:
            W: Treatment assignment matrix of shape (N, T)
            
        Returns:
            Outcome matrix of shape (N, T)
        """
        pass
    
    @abstractmethod
    def initialize_environment(self) -> None:
        """Initialize environment-specific parameters and structures.
        
        This abstract method must be implemented by each environment to set up
        any required internal state before simulation can begin.
        """
        pass
    
    def run_simulation(self, staggered: bool = True) -> Tuple[np.ndarray, ...]:
        """Run a complete simulation including desired treatment scenarios.
        
        This method coordinates the overall simulation process:
        1. Initializes the environment
        2. Generates treatment assignments
        3. Simulates outcomes for observed and desired scenarios
        
        Args:
            staggered: Whether to use staggered rollout design
            
        Returns:
            Tuple of:
            - W: Observed treatment matrix
            - Y: Observed outcome matrix
            - W1: First desired treatment matrix
            - Y1: First desired outcome matrix
            - W2: Second desired treatment matrix
            - Y2: Second desired outcome matrix
        """
        self.initialize_environment()
        
        # Generate treatment matrices using appropriate time points
        W = self._generate_W(self.probs, self.stage_time_blocks, self.T, staggered)
        W1 = self._generate_W(self.desired_probs_1, self.desired_stage_time_blocks, self.desired_T, staggered)
        W2 = self._generate_W(self.desired_probs_2, self.desired_stage_time_blocks, self.desired_T, staggered)
        
        Y = self.generate_outcomes(W)
        Y1 = self.generate_outcomes(W1)
        Y2 = self.generate_outcomes(W2)
        
        return W, Y, W1, Y1, W2, Y2
    
    def _generate_W(self, 
                   probs: np.ndarray, 
                   stage_time_blocks: np.ndarray,
                   T: int,
                   staggered: bool) -> np.ndarray:
        """Generate treatment allocation matrix.
        
        Creates a treatment assignment matrix based on specified probabilities
        and time points, supporting both staggered and independent assignments.
        
        Args:
            probs: Treatment probabilities for different phases
            stage_time_blocks: Time points where treatment probabilities change
            T: Total number of time periods
            staggered: Whether to use staggered rollout
            
        Returns:
            Treatment allocation matrix of shape (N, T)
        """
        from ..treatment_generator import generate_W
        return generate_W(
            self.N, T, probs, stage_time_blocks, 
            staggered=staggered, 
            seed=self.seed
        )