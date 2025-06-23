import numpy as np
from typing import Dict, Optional, Tuple
import logging

from ..simulator.outcome_generator import generate_data_from_design

logger = logging.getLogger(__name__)

def cmp_simulator(
    environment: Dict,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Main interface function for simulation environments.
    
    This function provides a unified interface for running simulations across
    different environments.
    
    Parameters
    ----------
    environment : dict
        Dictionary specifying simulation environment configuration including:
        - setting: String identifier for environment type
        - N: Number of units
        - stage_time_blocks: Time points for regime changes
        - design: Treatment probabilities
        - desired_design_1: First desired treatment regime
        - desired_design_2: Second desired treatment regime
        - desired_stage_time_blocks: Time points for desired regimes
        Additional parameters specific to chosen environment
    seed : int, optional
        Random seed for reproducibility. If None, default seed is used
        
    Returns
    -------
    tuple
        (W, Y, desired_W_1, desired_Y_1, desired_W_2, desired_Y_2) where:
        - W: Treatment matrix (N × T)
        - Y: Outcome matrix (N × T)
        - desired_W_1: First desired treatment matrix (N × T)
        - desired_Y_1: First desired outcome matrix (N × T)
        - desired_W_2: Second desired treatment matrix (N × T)
        - desired_Y_2: Second desired outcome matrix (N × T)
    """
    logger.info("Initializing simulation environment")
    
    # Log environment settings
    logger.debug(f"Environment setting: {environment.get('setting')}")
    logger.debug(f"Number of units: {environment.get('N')}")
    logger.debug(f"Stage time blocks: {environment.get('stage_time_blocks')}")
    logger.debug(f"Treatment design: {environment.get('design')}")
    
    if seed is not None:
        logger.debug(f"Using random seed: {seed}")
    
    logger.debug("Running simulation")
    results = generate_data_from_design(environment, staggered=True, seed=seed)
    
    # Log simulation results
    W, Y, W1, Y1, W2, Y2 = results
    logger.debug(f"Generated data shapes - W: {W.shape}, Y: {Y.shape}")
    logger.debug(f"Generated counterfactual data shapes - W1: {W1.shape}, W2: {W2.shape}")
    
    logger.info("Simulation completed successfully")
    return results