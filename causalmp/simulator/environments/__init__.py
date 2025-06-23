from typing import Dict, Optional, Any

# Import base environment
from .base_environment import BaseEnvironment

# Import all environment implementations
from .belief_adoption import BeliefAdoptionEnvironment
from .auction import AuctionEnvironment
from .nyc_taxi import NYCTaxiRoutesEnvironment
from .exercise import ExerciseEnvironment
from .data_center import DataCenterEnvironment
from ..validation import validate_environment

# Define mapping of environment types to their classes
ENVIRONMENT_REGISTRY = {
    BeliefAdoptionEnvironment.ENVIRONMENT_TYPE: BeliefAdoptionEnvironment,
    AuctionEnvironment.ENVIRONMENT_TYPE: AuctionEnvironment,
    NYCTaxiRoutesEnvironment.ENVIRONMENT_TYPE: NYCTaxiRoutesEnvironment,
    ExerciseEnvironment.ENVIRONMENT_TYPE: ExerciseEnvironment,
    DataCenterEnvironment.ENVIRONMENT_TYPE: DataCenterEnvironment,
}

def create_environment(params: Dict[str, Any], seed: Optional[int] = None) -> BaseEnvironment:
    """Factory function to create appropriate environment instance.
    
    Args:
        params: Dictionary containing environment parameters including:
            - setting: String identifier for the environment type
            - N: Number of units
            - stage_time_blocks: Time points for regime changes
            - design: Treatment probabilities
            - desired_design_1: First desired treatment regime
            - desired_design_2: Second desired treatment regime
            - desired_stage_time_blocks: Time points for desired regimes
            Additional parameters specific to each environment type
        seed: Random seed for reproducibility
        
    Returns:
        Instance of appropriate environment class
        
    Raises:
        ValueError: If environment type is not specified or unknown
    """
    env_type = params.get("setting")
    if not env_type:
        raise ValueError("Environment setting not specified in params")
    
    if env_type not in ENVIRONMENT_REGISTRY:
        valid_types = "\n- ".join([""] + list(ENVIRONMENT_REGISTRY.keys()))
        raise ValueError(
            f"Unknown environment type: {env_type}\n"
            f"Valid environment types are:{valid_types}"
        )
    
    validate_environment(params)
    # Create and return appropriate environment instance
    return ENVIRONMENT_REGISTRY[env_type](params, seed)

# Make key components available at package level
__all__ = [
    'BaseEnvironment',
    'BeliefAdoptionEnvironment',
    'AuctionEnvironment',
    'NYCTaxiRoutesEnvironment',
    'ExerciseEnvironment',
    'DataCenterEnvironment',
    'create_environment',
    'ENVIRONMENT_REGISTRY'
]