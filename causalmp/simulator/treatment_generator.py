import numpy as np
from typing import List, Union, Optional

def generate_W(
    N: int,
    T: int,
    probs: List[float],
    stage_time_blocks: List[int],
    staggered: bool = True,
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate a treatment assignment matrix with an option for staggered rollout.
    
    Parameters
    ----------
    N : int
        Number of units
    T : int
        Total number of time periods
    probs : array-like
        List of treatment probabilities, where each value represents the probability
        of treatment assignment during its corresponding time block
    stage_time_blocks : array-like
        List of time points where treatment probability changes. Must be same length 
        as probs. Each value indicates the start of a new probability block.
    staggered : bool, default=True
        If True, units keep their initial random draw throughout time (staggered rollout)
        If False, new random draws are made each period (independent assignment)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    numpy.ndarray, shape (N, T)
        Binary treatment assignment matrix where 1 indicates treatment and 0 control
        
    Raises
    ------
    ValueError
        If probability values are invalid or dimensions do not match
    """
    # Input validation
    if len(probs) != len(stage_time_blocks):
        raise ValueError("probs and stage_time_blocks must have same length")
    if not all(0 <= p <= 1 for p in probs):
        raise ValueError("All treatment probabilities must be between 0 and 1")
        
    # Initialize treatment matrix
    W = np.zeros((N, T))
    current_index = 0
    
    # Set random seed for reproducibility
    rng = np.random.default_rng(seed)
    
    # For staggered design, generate one random draw per unit
    # These values determine when each unit enters treatment
    unit_random_values = rng.random(N)
    
    # Generate treatment assignments for each time period
    for t in range(T):
        # Update probability index if we've reached next time threshold
        if current_index < len(stage_time_blocks) - 1 and t >= stage_time_blocks[current_index]:
            current_index += 1
            
        # Get random values for this period
        if staggered:
            # Use fixed random values for staggered design
            period_random_values = unit_random_values
        else:
            # Generate new random values for independent assignment
            period_random_values = rng.random(N)
            
        # Assign treatment based on current probability threshold
        current_prob = probs[current_index]
        W[:,t] = (period_random_values < current_prob)
    
    return W