"""
causalmp Interface Module

This module provides the main interface functions for the causalmp package,
enabling easy access to counterfactual estimation, simulation, and experimental
running capabilities.

Main Functions:
    - cmp_estimator: Counterfactual estimation interface
    - cmp_simulator: Simulation environment interface
    - cmp_runner: Combined estimation and simulation interface
"""

from .cmp_estimator import cmp_estimator
from .cmp_simulator import cmp_simulator
from .cmp_runner import cmp_runner

__all__ = [
    'cmp_estimator',
    'cmp_simulator',
    'cmp_runner'
]

# Interface version tracking
__version__ = '0.1.0'

# Example usage
EXAMPLE_USAGE = """
# Example: Running counterfactual estimation
from causalmp import cmp_estimator

estimates = cmp_estimator(
    Y=outcomes,
    W=treatments,
    desired_W=desired_treatments,
    main_param_ranges=config_ranges,
    n_time_blocks=5,
    n_validation_batch=2
)

# Example: Running simulation
from causalmp import cmp_simulator

W, Y, W1, Y1, W2, Y2 = cmp_simulator(
    environment=env_config
)

# Example: Running full experiment
from causalmp import cmp_runner

observed, cfes, ttes = cmp_runner(
    environment=env_config,
    main_param_ranges=config_ranges,
    n_time_blocks=5,
    n_validation_batch=2,
    visualize_results=True
)

# Example: Configuring logging
from causalmp import configure
configure({
    'logging': {
        'level': 'INFO',  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': 'causalmp.log'  # Optional: log to file
    }
})
"""

def get_version():
    """Return the current version of the interface."""
    return __version__

def get_example_usage():
    """Return example usage of the interface functions."""
    return EXAMPLE_USAGE

# Optional Settings
DEFAULT_CONFIG = {
    'visualization': {
        'figsize': (24, 6),
        'style': 'whitegrid',
        'dpi': 300
    },
    'parallel': {
        'default_processes': None,  # Auto-detect
        'chunk_size': 1
    },
    'logging': {
        'level': 'WARNING',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
}

# Environment Types
AVAILABLE_ENVIRONMENTS = {
    'Belief Adoption Model': 'Social network belief propagation simulation',
    'Auction Model': 'Multi-bidder auction market simulation',
    'NYC Taxi Routes': 'NYC taxi zone route dynamics simulation',
    'Exercise Encouragement Program': 'Social network exercise program simulation',
    'Data Center Model': 'Server farm task allocation simulation'
}

def get_available_environments():
    """Return dictionary of available simulation environments."""
    return AVAILABLE_ENVIRONMENTS

def configure(settings: dict) -> None:
    """Configure global interface settings.
    
    Parameters
    ----------
    settings : dict
        Dictionary of settings to update
    """
    global DEFAULT_CONFIG
    
    for category, values in settings.items():
        if category in DEFAULT_CONFIG:
            DEFAULT_CONFIG[category].update(values)
        else:
            DEFAULT_CONFIG[category] = values
    
    # Apply logging configuration if present
    if 'logging' in settings:
        from ..logging_config import configure_logging
        log_settings = settings['logging']
        configure_logging(
            level=log_settings.get('level', 'WARNING'),
            log_format=log_settings.get('format', None),
            log_file=log_settings.get('file', None)
        )