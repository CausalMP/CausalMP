import numpy as np
from typing import Dict, Tuple, Optional, Any
from ..simulator.environments import create_environment, BaseEnvironment

def generate_data_from_design(
    environment_params: Dict[str, Any],
    staggered: bool = True,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, ...]:
    """Generate simulation data using specified environment design.
    
    Parameters
    ----------
    environment_params : dict
        Dictionary containing environment configuration parameters including:
        - setting: String identifier for the environment type
        - N: Number of units
        - stage_time_blocks: Time points for regime changes
        - design: Treatment probabilities
        - desired_design_1: First desired treatment regime
        - desired_design_2: Second desired treatment regime
        - desired_stage_time_blocks: Time points for desired regimes
        Additional parameters specific to chosen environment type
    staggered : bool, default=True
        Whether to use staggered rollout design
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        (W, Y, W1, Y1, W2, Y2) where:
        - W: Observed treatment matrix
        - Y: Observed outcome matrix
        - W1: First desired treatment matrix
        - Y1: First desired outcome matrix
        - W2: Second desired treatment matrix
        - Y2: Second desired outcome matrix
    """
    # Create appropriate environment instance
    env = create_environment(environment_params, seed=seed)
    
    # Run simulation
    return env.run_simulation(staggered=staggered)