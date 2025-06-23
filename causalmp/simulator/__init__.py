from typing import Dict, List, Optional, Any

# Import key components
from .environments import (
    BaseEnvironment,
    BeliefAdoptionEnvironment,
    AuctionEnvironment,
    NYCTaxiRoutesEnvironment,
    ExerciseEnvironment,
    DataCenterEnvironment,
    create_environment,
    ENVIRONMENT_REGISTRY
)
from .treatment_generator import generate_W
from .outcome_generator import generate_data_from_design

# Make common functions and classes available at package level
__all__ = [
    # Environments
    'BaseEnvironment',
    'BeliefAdoptionEnvironment',
    'AuctionEnvironment',
    'NYCTaxiRoutesEnvironment',
    'ExerciseEnvironment',
    'DataCenterEnvironment',
    'create_environment',
    'ENVIRONMENT_REGISTRY',
    
    # Generators
    'generate_W',
    'generate_data_from_design'
]

# Version info
__version__ = '0.1.0'