import numpy as np
from typing import List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

def validate_time_blocks_inputs(
    time_values: Union[List[int], np.ndarray],
    n_time_blocks: Optional[int]
) -> None:
    """Validate input for time blocks generator."""
    if not np.all(np.diff(np.array(time_values)) >= 0):
        raise ValueError("time_values must be in ascending order")
    
    if n_time_blocks is not None and n_time_blocks < 1:
        raise ValueError("n_time_blocks must be positive if specified")

    if not np.all(np.mod(time_values, 1) == 0):
        raise ValueError("time_values must be integers")
    
def time_blocks_generator(
    time_values: Union[List[int], np.ndarray],
    n_time_blocks: Optional[int] = None
) -> List[Tuple[int, int]]:
    """Generate time blocks and their corresponding intervals for cross-validation.

    Parameters
    ----------
    time_values : list or numpy.ndarray
        Time values marking regime changes
    n_time_blocks : int or None
        If None, returns blocks based on original time_values
        If int, generates that many equally spaced blocks

    Returns
    -------
    list
        List of tuples representing intervals [(start1, end1), (start2, end2), ...]
        
    Raises
    ------
    ValueError
        If inputs are invalid or inconsistent
    """
    logger.debug(f"Generating time blocks with n_time_blocks={n_time_blocks}")
    
    # Input validation
    validate_time_blocks_inputs(time_values, n_time_blocks)
    
    # Convert input to numpy array
    time_values = np.array(time_values)
    
    # Generate the time blocks
    if n_time_blocks is None:
        t_blocks = time_values
        # Add 0 at the beginning if not present
        if t_blocks[0] != 0:
            t_blocks = np.concatenate(([0], t_blocks))
    else:
        # Generate equally spaced values
        time_max = time_values.max()
        t_blocks = np.linspace(0, time_max, n_time_blocks + 1, dtype=int)
    
    # Generate intervals
    time_blocks_list = [(t_blocks[i], t_blocks[i+1]) 
                    for i in range(len(t_blocks)-1)]
    
    logger.debug(f"Generated {len(time_blocks_list)} time blocks")
    return time_blocks_list