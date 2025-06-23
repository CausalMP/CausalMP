from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

from ..runner import (
    MultiExperimentRunner,
    ParallelExperimentRunner,
    ResultVisualizer
)
from ..estimator import config_utils

logger = logging.getLogger(__name__)

def cmp_runner(
    environment: Dict,
    main_param_ranges: Dict,
    n_validation_batch: int = 1,
    n_time_blocks: int = None,
    detrending_options: Optional[List[bool]] = None,
    detrending_param_ranges: Optional[Dict] = None,
    visualize_results: bool = True,
    visualization_path: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    visualize_ccv: bool = False,
    return_model_terms: bool = False,
    n_runs: int = 1,
    n_processes: Optional[int] = None,
) -> Tuple[Dict, Dict, Dict]:
    """Main interface function combining simulation and estimation.
    
    This function provides a unified interface for running experiments that combine
    simulation and estimation, with support for multiple runs, visualization, and
    parallel processing.
    
    Parameters
    ----------
    environment : dict
        Simulation environment configuration
    main_param_ranges : dict
        Parameter ranges for main estimator configuration
    n_validation_batch: int, default=1
        Number of test batches for validation
    n_time_blocks : int, default=None
        Number of time blocks for cross-validation
    detrending_options : list of bool, optional
        If provided, specifies whether to use detrending
    detrending_param_ranges : dict, optional
        Required if detrending_options is provided
    visualize_results : bool, default=True
        Whether to visualize final results
    visualization_path : str, optional
        If provided, filename and format to save visualization
    data_path : str or Path, optional
        If provided, path to read the data instead of generating it
    visualize_ccv : bool, default=False
        Whether to visualize cross-validation results
    return_model_terms : bool, default=False
        Whether to display configuration and model terms
    n_runs : int, default=1
        Number of experimental runs
    n_processes : int, optional
        Number of parallel processes to use
        
    Returns
    -------
    tuple
        (all_observed_outcomes, all_CFEs, all_TTEs) as dictionaries containing:
        - Time: time points
        - Values: outcomes/effects
        - Run: run identifier
        - Label: result type
        
    Notes
    -----
    Results dictionaries can be easily converted to pandas DataFrames for analysis.
    """
    logger.info("Initializing experiment runner")
    
    # Validate inputs
    logger.debug("Validating input parameters")
    _validate_inputs(
        environment,
        main_param_ranges,
        n_time_blocks,
        n_validation_batch,
        detrending_options,
        detrending_param_ranges,
        n_runs,
        n_processes
    )
    
    # Generate configurations
    logger.debug("Generating configurations")
    grouped_configurations = config_utils.generate_configurations(
        detrending_options=detrending_options or [False],
        detrending_param_ranges=detrending_param_ranges or {},
        main_param_ranges=main_param_ranges
    )
    logger.info(f"Generated {sum(len(configs) for configs in grouped_configurations.values())} configurations")
    
    # Generate time blocks
    logger.debug("Generating time blocks")
    time_blocks_list=_generate_time_blocks(n_time_blocks, environment)
    logger.info(f"Generated time blocks: {time_blocks_list}")

    # Initialize appropriate runner
    if n_runs == 1 or n_processes is None or n_processes == 1:
        logger.info("Using sequential experiment runner")
        runner = MultiExperimentRunner(
            environment=environment,
            grouped_configurations=grouped_configurations,
            time_blocks_list=time_blocks_list,
            data_path=data_path
        )
    else:
        logger.info(f"Using parallel experiment runner with {n_processes} processes")
        runner = ParallelExperimentRunner(
            environment=environment,
            grouped_configurations=grouped_configurations,
            time_blocks_list=time_blocks_list,
            n_processes=n_processes,
            data_path=data_path
        )
    
    logger.info(f"Running {n_runs} experiments")
    
    # Run experiments
    all_observed_outcomes, all_CFEs, all_TTEs = runner.run(
        n_runs=n_runs,
        n_validation_batch=n_validation_batch,
        visualize_ccv=visualize_ccv,
        return_model_terms=return_model_terms,
    )
    
    # Visualize results if requested
    if visualize_results:
        logger.info("Visualizing results")
        _visualize_results(
            all_observed_outcomes,
            all_CFEs,
            all_TTEs,
            filename=visualization_path
        )
    
    logger.info("Experiment completed successfully")
    return all_observed_outcomes, all_CFEs, all_TTEs

def _validate_inputs(
    environment: Dict,
    main_param_ranges: Dict,
    n_time_blocks: int,
    n_validation_batch: int,
    detrending_options: Optional[List[bool]],
    detrending_param_ranges: Optional[Dict],
    n_runs: int,
    n_processes: Optional[int]
) -> None:
    """Validate experiment parameters."""
    # Validate environment configuration
    required_env_params = {
        'setting', 'N', 'stage_time_blocks', 'design',
        'desired_design_1', 'desired_design_2', 'desired_stage_time_blocks'
    }
    missing_params = required_env_params - set(environment.keys())
    if missing_params:
        logger.error(f"Missing required environment parameters: {missing_params}")
        raise ValueError(f"Missing required environment parameters: {missing_params}")
    
    # Validate main configuration ranges
    config_utils.validate_parameter_ranges(
        main_param_ranges,
        prefix="Main configuration: "
    )
    
    # Validate detrending configuration
    if detrending_options is not None:
        if not all(isinstance(x, bool) for x in detrending_options):
            logger.error("detrending_options must contain only boolean values")
            raise ValueError("detrending_options must contain only boolean values")
            
        if any(detrending_options) and detrending_param_ranges is None:
            logger.error("detrending_param_ranges required when detrending is enabled")
            raise ValueError("detrending_param_ranges required when detrending is enabled")
            
        if detrending_param_ranges:
            config_utils.validate_parameter_ranges(
                detrending_param_ranges,
                prefix="Detrending configuration: ",
                is_detrending=True
            )
    
    # Validate numeric parameters
    if n_time_blocks is not None and (not isinstance(n_time_blocks, int) or n_time_blocks < 2):
        logger.error(f"n_time_blocks must be a positive integer of at least 2, got {n_time_blocks}")
        raise ValueError("n_time_blocks must be a positive integer of at least 2")
        
    if not isinstance(n_validation_batch, int) or n_validation_batch < 1:
        logger.error(f"n_validation_batch must be a positive integer, got {n_validation_batch}")
        raise ValueError("n_validation_batch must be a positive integer")
        
    if not isinstance(n_runs, int) or n_runs < 1:
        logger.error(f"n_runs must be a positive integer, got {n_runs}")
        raise ValueError("n_runs must be a positive integer")
        
    if n_processes is not None:
        if not isinstance(n_processes, int) or n_processes < 1:
            logger.error(f"n_processes must be a positive integer, got {n_processes}")
            raise ValueError("n_processes must be a positive integer")

def _generate_time_blocks(
    n_time_blocks: int,
    environment: Dict
) -> List[Tuple[int, int]]:
    """Generate time blocks for validation."""
    from ..estimator.time_block_utils import time_blocks_generator
    
    return time_blocks_generator(
        time_values=environment['stage_time_blocks'],
        n_time_blocks=n_time_blocks
    )

def _visualize_results(
    observed_outcomes: Dict,
    CFEs: Dict,
    TTEs: Dict,
    filename: Optional[str] = None
) -> None:
    """Visualize experimental results."""
    visualizer = ResultVisualizer()
    
    try:
        visualizer.plot_results(
            Observed_outcomes=observed_outcomes,
            CFEs=CFEs,
            TTEs=TTEs,
            filename=filename
        )
    except Exception as e:
        logger.error(f"Error in visualization: {str(e)}")
        raise