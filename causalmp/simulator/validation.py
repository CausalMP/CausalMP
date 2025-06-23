from typing import Dict, Any, List, Optional
import numpy as np

def validate_environment(environment: Dict[str, Any]) -> None:
    """Validate environment configuration dictionary.
    
    Parameters
    ----------
    environment : dict
        Environment configuration dictionary
        
    Raises
    ------
    ValueError
        If any validation check fails
    """
    # Check required base parameters
    _validate_required_params(environment)
    
    # Validate parameter types
    _validate_parameter_types(environment)
    
    # Validate time blocks
    _validate_time_blocks(
        environment['stage_time_blocks'],
        environment['desired_stage_time_blocks']
    )
    
    # Validate probability designs
    _validate_probability_designs(
        environment['design'],
        environment['desired_design_1'],
        environment['desired_design_2']
    )

def _validate_required_params(environment: Dict[str, Any]) -> None:
    """Validate presence of required parameters."""
    required_params = {
        'setting',
        'N',
        'stage_time_blocks',
        'design',
        'desired_design_1',
        'desired_design_2',
        'desired_stage_time_blocks'
    }
    
    missing_params = required_params - set(environment.keys())
    if missing_params:
        raise ValueError(f"Missing required parameters: {missing_params}")

def _validate_parameter_types(environment: Dict[str, Any]) -> None:
    """Validate parameter types."""
    # Check N
    if not isinstance(environment['N'], int) or environment['N'] <= 0:
        raise ValueError("N must be a positive integer")
    
    # Check setting
    if not isinstance(environment['setting'], str):
        raise ValueError("setting must be a string")
    
    # Check time blocks
    if not (isinstance(environment['stage_time_blocks'], (list, np.ndarray)) and
            isinstance(environment['desired_stage_time_blocks'], (list, np.ndarray))):
        raise ValueError("Time blocks must be lists or numpy arrays")

def _validate_time_blocks(
    stage_blocks: List[int],
    desired_blocks: List[int]
) -> None:
    """Validate time block configurations."""
    # Check if time blocks are sorted
    if not all(x < y for x, y in zip(stage_blocks[:-1], stage_blocks[1:])):
        raise ValueError("stage_time_blocks must be in ascending order")
        
    if not all(x < y for x, y in zip(desired_blocks[:-1], desired_blocks[1:])):
        raise ValueError("desired_stage_time_blocks must be in ascending order")
    
    # Check if time blocks are positive
    if min(stage_blocks) < 0 or min(desired_blocks) < 0:
        raise ValueError("Time blocks must be non-negative")
    
    # Check if time blocks cover the same range
    if stage_blocks[-1] != desired_blocks[-1]:
        raise ValueError("stage_time_blocks and desired_stage_time_blocks must cover the same range")

def _validate_probability_designs(
    design: List[float],
    desired_1: List[float],
    desired_2: List[float]
) -> None:
    """Validate probability design configurations."""
    for name, probs in [
        ('design', design),
        ('desired_design_1', desired_1),
        ('desired_design_2', desired_2)
    ]:
        # Check probability values
        if not all(0 <= p <= 1 for p in probs):
            raise ValueError(f"All probabilities in {name} must be between 0 and 1")
        
        # Check length matches time blocks
        if len(probs) < 2:
            raise ValueError(f"{name} must contain at least 2 probability values")