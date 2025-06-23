"""
causalmp Runner Module

This module provides components for running and visualizing experiments that combine
simulation environments and counterfactual estimators. The module includes tools for
executing single experiments, parallel experiments, and visualizing results.

Components:
    - SingleExperimentRunner: Execute individual experimental trials
    - ParallelExperimentRunner: Execute multiple trials in parallel
    - ResultVisualizer: Visualize experimental results
"""

from .experiment_runners import ExperimentRunner, SingleExperimentRunner, MultiExperimentRunner
from .parallel_runner import ParallelExperimentRunner
from .visualizer import ResultVisualizer

__all__ = [
    # Runner classes
    'ExperimentRunner',
    'SingleExperimentRunner',
    'MultiExperimentRunner',
    'ParallelExperimentRunner',
    
    # Visualization
    'ResultVisualizer',
]

# Version tracking
__version__ = '0.1.0'

def get_example_usage():
    """Return example usage of runner components."""
    return """
    # Example: Running a single experiment
    from causalmp.runner import SingleExperimentRunner
    
    runner = SingleExperimentRunner(
        environment=env_config,
        grouped_configurations=grouped_configurations,
        time_blocks_list=time_blocks
    )
    observed, cfes, ttes = runner.run(Kbatch_validation=2)
    
    # Example: Running multiple sequential experiments
    from causalmp.runner import MultiExperimentRunner
    
    multi_runner = MultiExperimentRunner(
        environment=env_config,
        grouped_configurations=grouped_configurations,
        time_blocks_list=time_blocks
    )
    all_observed, all_cfes, all_ttes = multi_runner.run(n_runs=5)
    
    # Example: Running parallel experiments
    from causalmp.runner import ParallelExperimentRunner
    
    parallel_runner = ParallelExperimentRunner(
        environment=env_config,
        grouped_configurations=grouped_configurations,
        time_blocks_list=time_blocks,
        n_processes=4
    )
    all_observed, all_cfes, all_ttes = parallel_runner.run(N_runs=10)
    
    # Example: Visualizing results
    from causalmp.runner import ResultVisualizer
    
    visualizer = ResultVisualizer()
    visualizer.plot_results(
        Observed_outcomes=all_observed, 
        CFEs=all_cfes, 
        TTEs=all_ttes,
        filename="experiment_results.pdf"
    )
    """