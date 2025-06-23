"""
causalmp: A Python Package for Counterfactual Estimation and Simulation

This package provides tools for counterfactual estimation, simulation environments,
and experimental analysis in causal inference scenarios. It is designed to be
modular, allowing users to work with estimation or simulation components
independently or combined.

Main Components:
---------------
1. Estimator:
   - Counterfactual Evolution (CFE) estimation
   - Semi-recursive estimation for detrending the observed outcomes
   - Cross-validation for model selection
   - Feature engineering and batch processing
   - Classic estimators (DinM, HT, Basic CMP)

2. Simulator:
   - Multiple simulation environments
   - Staggered rollout support
   - Customizable parameters

3. Runner:
   - Combined simulation and estimation
   - Result visualization
   - Multi-run experiments
   - Parallel processing

Main Interfaces:
---------------
cmp_estimator : Main estimation interface
cmp_simulator : Main simulation interface
cmp_runner : Combined experiment interface
"""

# Import component requirements from the standalone file
from .component_requirements import (
    COMPONENT_REQUIREMENTS, 
    verify_component_requirements,
    _check_installation
)

# Import main interfaces
from .interface import (
    cmp_estimator,
    cmp_simulator,
    cmp_runner
)

# Import key components from estimator
from .estimator import (
    CFEEstimator,
    CounterfactualEstimator,
    CFESemiRecursiveEstimator,
    CrossValidator,
    dinm_estimate,
    ht_estimate,
    basic_cmp_estimate
)

# Import simulator components
from .simulator import (
    BaseEnvironment,
    BeliefAdoptionEnvironment,
    AuctionEnvironment,
    NYCTaxiRoutesEnvironment,
    ExerciseEnvironment,
    DataCenterEnvironment,
    create_environment
)

# Import runner components
from .runner import (
    SingleExperimentRunner,
    MultiExperimentRunner,
    ParallelExperimentRunner,
    ResultVisualizer
)

from .logging_config import configure_logging
import logging

# Version info
__version__ = '0.1.0'

# Define what gets imported with *
__all__ = [
    # Main interfaces
    'cmp_estimator',
    'cmp_simulator',
    'cmp_runner',
    
    # Estimator components
    'CFEEstimator',
    'CounterfactualEstimator',
    'CFESemiRecursiveEstimator',
    'CrossValidator',
    'dinm_estimate',
    'ht_estimate',
    'basic_cmp_estimate',
    
    # Simulator components
    'BaseEnvironment',
    'BeliefAdoptionEnvironment',
    'AuctionEnvironment',
    'NYCTaxiRoutesEnvironment',
    'ExerciseEnvironment',
    'DataCenterEnvironment',
    'create_environment',
    
    # Runner components
    'SingleExperimentRunner',
    'MultiExperimentRunner',
    'ParallelExperimentRunner',
    'ResultVisualizer',
    
    # Utility functions
    'get_version',
    'get_citation',
    'get_installation_status',
    'get_available_environments',
    'get_example_config',
    'configure',
    'verify_component_requirements'
]

# Package information
def get_version():
    """Return package version."""
    return __version__

def get_citation():
    """Return citation information for the package."""
    citation = """
@article{shirani2025can,
  title={Can We Validate Counterfactual Estimations in the Presence of General Network Interference?},
  author={Shirani, Sadegh and Luo, Yuwei and Overman, William and Xiong, Ruoxuan and Bayati, Mohsen},
  journal={arXiv preprint arXiv:2502.01106},
  year={2025}
}
@article{shirani2024causal,
  title={Causal message-passing for experiments with unknown and general network interference},
  author={Shirani, Sadegh and Bayati, Mohsen},
  journal={Proceedings of the National Academy of Sciences},
  volume={121},
  number={40},
  pages={e2322232121},
  year={2024},
  publisher={National Academy of Sciences}
}
    """
    return citation.strip()

def get_installation_status():
    """Print the installation status of all components."""
    components = verify_component_requirements()
    
    print("CausalMP Installation Status:")
    print("============================")
    
    # First print individual components
    for component in COMPONENT_REQUIREMENTS.keys():
        info = components[component]
        print(f"\n{component.upper()}:")
        if info['installed']:
            print(f"✓ Installed (version: {info['version']})")
            print("Dependencies:")
            for pkg, ver in COMPONENT_REQUIREMENTS[component]['dependencies'].items():
                print(f"  - {pkg} (>= {ver})")
        else:
            print("✗ Not fully installed")
            if info['missing']:
                print("Missing dependencies:")
                for dep in info['missing']:
                    print(f"  - {dep}")
            if info['outdated']:
                print("Outdated dependencies:")
                for dep in info['outdated']:
                    print(f"  - {dep}")
    
    # Then print overall status
    print("\nOVERALL STATUS:")
    if components['all']['installed']:
        print("✓ All components are installed and ready to use")
    else:
        print("✗ Some components are missing or outdated")
        if components['all']['missing']:
            print("Missing dependencies:")
            for dep in components['all']['missing']:
                print(f"  - {dep}")
        if components['all']['outdated']:
            print("Outdated dependencies:")
            for dep in components['all']['outdated']:
                print(f"  - {dep}")

# Configure logging
configure_logging(level=logging.INFO)

# Check installation status on import
INSTALLED_COMPONENTS = _check_installation()

def get_available_environments():
    """Return information about available simulation environments."""
    from .simulator.environments import ENVIRONMENT_REGISTRY
    
    env_info = {}
    for env_name, env_class in ENVIRONMENT_REGISTRY.items():
        env_info[env_name] = {
            'description': env_class.__doc__.split('\n')[0] if env_class.__doc__ else "No description",
            'parameters': getattr(env_class, 'REQUIRED_PARAMS', [])
        }
    
    return env_info

def get_example_environment(environment_type=None):
    """Return example environment for a specific environment or all environments."""
    examples = {
        "Belief Adoption Model": {
            'N': 3366,
            'setting': "Belief Adoption Model",
            'design': [0, 0.1, 0.2, 0.5],
            'stage_time_blocks': [1, 3, 5, 7],
            'desired_design_1': [0, 0],
            'desired_design_2': [0, 1],
            'desired_stage_time_blocks': [1, 7],
            'tau': 1
        },
        "Auction Model": {
            'N': 200,
            'setting': "Auction Model",
            'design': [0, 0.1, 0.2, 0.5],
            'stage_time_blocks': [1, 6, 11, 16],
            'desired_design_1': [0, 0],
            'desired_design_2': [0, 1],
            'desired_stage_time_blocks': [1, 16],
            'tau': 0.1
        },
        "NYC Taxi Routes": {
            'N': 18768,
            'setting': "NYC Taxi Routes",
            'design': [0, 0.1, 0.2, 0.5],
            'stage_time_blocks': [1, 29, 57, 85],
            'desired_design_1': [0, 0],
            'desired_design_2': [0, 1],
            'desired_stage_time_blocks': [1, 85],
            'tau_unit_coeff': 1
        },
        "Exercise Encouragement Program": {
            'N': 30162,
            'setting': "Exercise Encouragement Program",
            'design': [0, 0.1, 0.2, 0.5],
            'stage_time_blocks': [1, 8, 15, 22],
            'desired_design_1': [0, 0],
            'desired_design_2': [0, 1],
            'desired_stage_time_blocks': [1, 22],
            'tau_unit_coeff': 1
        },
        "Data Center Model": {
            'N': 2000,
            'setting': "Data Center Model",
            'design': [0, 0.1, 0.2, 0.5],
            'stage_time_blocks': [1, 25, 49, 73],
            'desired_design_1': [0, 0],
            'desired_design_2': [0, 1],
            'desired_stage_time_blocks': [1, 73],
            'tau': 0.1
        }
    }    
    if environment_type:
        if environment_type in examples:
            return examples[environment_type]
        else:
            raise ValueError(f"Unknown environment type: {environment_type}. Available types: {list(examples.keys())}")
    
    return examples

def get_example_config():
    """Return example configuration for quick start."""
    return {
        'environment': {
            'setting': 'Belief Adoption Model',
            'N': 3366,
            'stage_time_blocks': [1, 3, 5, 7],
            'design': [0, 0.1, 0.2, 0.5],
            'desired_design_1': [0, 0],
            'desired_design_2': [0, 1],
            'desired_stage_time_blocks': [1, 7],
            'tau': 1
        },
        'estimation': {
            'n_validation_batch': 2,
            'time_blocks': [(0, 3), (3, 5), (5, 7)],
            'detrending_options': [True],
            'detrending_param_ranges': {
                'n_lags_Y_range': [1],
                'interaction_term_p_range': [None],
                'interaction_term_u_range': [1],
                'n_batch_range': [1],
                'batch_size_range': [3366],
                'ridge_alpha_range': [1e-4]
            },
            'main_param_ranges': {
                'n_lags_Y_range': [1],
                'n_lags_W_range': [1],
                'moment_order_p_Y_range': [1],
                'moment_order_p_W_range': [1],
                'moment_order_u_Y_range': [1],
                'moment_order_u_W_range': [1],
                'interaction_term_p_range': [None],
                'interaction_term_u_range': [None, 1],
                'n_batch_range': [100, 500, 1000],
                'batch_size_range': [168, 337, 673, 1010, 1683],
                'ridge_alpha_range': [1e-4, 1e-2, 1, 100]
            }
        }
    }