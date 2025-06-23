import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging

from ..estimator import (
    CrossValidator,
    CFEEstimator,
    config_utils,
    time_blocks_generator
)

logger = logging.getLogger(__name__)

def cmp_estimator(
    Y: np.ndarray,
    W: np.ndarray,
    desired_W: np.ndarray,
    main_param_ranges: Dict,
    desired_W_2: Optional[np.ndarray] = None,
    n_time_blocks: Optional[int] = None,
    n_validation_batch: int = 1,
    time_blocks: Optional[List[Tuple[int, int]]] = None,
    detrending_options: Optional[List[bool]] = None,
    detrending_param_ranges: Optional[Dict] = None,
    visualize_ccv: bool = False,
    return_model_terms: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict, List], Tuple[np.ndarray, np.ndarray, Dict, List]]:
    """Main interface function for counterfactual estimation.
    
    This function provides a unified interface for counterfactual estimation,
    handling model selection through cross-validation and generating counterfactual
    predictions for one or two desired treatment patterns.
    
    Parameters
    ----------
    Y : numpy.ndarray, shape (N, T)
        Panel data of outcomes
    W : numpy.ndarray, shape (N, T)
        Panel data of treatments
    desired_W : numpy.ndarray, shape (N, T)
        First desired treatment panel data for counterfactual prediction
    main_param_ranges : dict
        Dictionary of parameter ranges for main estimator configuration
    desired_W_2 : numpy.ndarray, shape (N, T), optional
        Second desired treatment panel data for counterfactual prediction
    n_time_blocks : int, optional
        Number of time blocks for cross-validation. Required if time_blocks is None.
    n_validation_batch: int, default=1
        Number of test batches for validation
    time_blocks : list of tuples, optional
        Pre-defined time blocks for validation in the format [(start1, end1), (start2, end2), ...].
        If provided, n_time_blocks is ignored.
    detrending_options : list of bool, optional
        If provided, specifies whether to use detrending
    detrending_param_ranges : dict, optional
        Required if detrending_options is provided, specifies detrending parameters
    visualize_ccv : bool, default=False
        Whether to visualize cross-validation results
    return_model_terms : bool, default=False
        Whether to return best configuration and model terms
        
    Returns
    -------
    numpy.ndarray or tuple
        If return_model_terms=False and desired_W_2 is None:
            numpy.ndarray: Estimated counterfactual evolution for desired_W (1 x T)
        If return_model_terms=True and desired_W_2 is None:
            tuple: (estimates, best_config, best_model_terms)
        If desired_W_2 is provided and return_model_terms=False:
            tuple: (estimates_1, estimates_2)
        If desired_W_2 is provided and return_model_terms=True:
            tuple: (estimates_1, estimates_2, best_config, best_model_terms)
            
    Raises
    ------
    ValueError
        If inputs are invalid or inconsistent
    """
    logger.info("Initializing counterfactual estimation")
    
    # Input validation
    logger.debug("Validating input parameters")
    _validate_inputs(
        Y, W, desired_W,
        detrending_options,
        detrending_param_ranges,
        main_param_ranges,
        n_time_blocks,
        time_blocks,
        desired_W_2
    )
    
    # Set default detrending if not provided
    if detrending_options is None:
        logger.debug("No detrending options provided, using default (False)")
        detrending_options = [False]
        detrending_param_ranges = {}
    
    # Use provided time blocks or generate them
    if time_blocks is not None:
        logger.info(f"Using provided time blocks: {time_blocks}")
    else:
        # Generate time blocks for validation
        logger.debug("Generating time blocks for validation")
        time_max = Y.shape[1]
        time_blocks = time_blocks_generator(
            time_values=[time_max],
            n_time_blocks=n_time_blocks
        )
        logger.info(f"Generated {len(time_blocks)} time blocks: {time_blocks}")
    
    # Generate configurations
    logger.debug("Generating configurations")
    grouped_configurations = config_utils.generate_configurations(
        detrending_options=detrending_options,
        detrending_param_ranges=detrending_param_ranges,
        main_param_ranges=main_param_ranges
    )
    logger.info(f"Generated {sum(len(configs) for configs in grouped_configurations.values())} configurations")
    
    logger.debug("Performing cross-validation")
    
    # Initialize cross-validator
    cross_validator = CrossValidator()
    
    # Find best configuration
    best_config, best_model_terms, best_score = cross_validator.validate(
        Y=Y,
        W=W,
        grouped_configurations=grouped_configurations,
        time_blocks_list=time_blocks,
        n_validation_batch=n_validation_batch,
        visualize=visualize_ccv
    )
    logger.info(f"Cross-validation completed successfully and the best score is {best_score}")
    
    logger.debug("Estimating counterfactual evolution")

    # Create and fit estimator with best configuration
    estimator = CFEEstimator(best_config)
    
    # Generate counterfactual predictions for the first desired treatment
    predictions_1 = estimator.estimate(Y, W, desired_W)
    
    # If a second desired treatment is provided, generate predictions for it too
    if desired_W_2 is not None:
        logger.info("Estimating counterfactual evolution for second desired treatment")
        predictions_2 = estimator.estimate(
            Y=Y,
            W=W,
            desired_W=desired_W_2
        )
        
        logger.info("Both counterfactual estimations completed successfully")
        
        # Return results based on model_terms flag
        if return_model_terms:
            return predictions_1, predictions_2, best_config, best_model_terms
        return predictions_1, predictions_2
    
    logger.info("Counterfactual estimation completed successfully")
    
    # Return results based on model_terms flag for single desired treatment
    if return_model_terms:
        return predictions_1, best_config, best_model_terms
    return predictions_1

def _validate_inputs(
    Y: np.ndarray,
    W: np.ndarray,
    desired_W: np.ndarray,
    detrending_options: Optional[List[bool]],
    detrending_param_ranges: Optional[Dict],
    main_param_ranges: Optional[Dict],
    n_time_blocks: Optional[int] = None,
    time_blocks: Optional[List[Tuple[int, int]]] = None,
    desired_W_2: Optional[np.ndarray] = None
) -> None:
    """Validate input parameters.
    
    Parameters
    ----------
    Y : numpy.ndarray
        Outcome matrix
    W : numpy.ndarray
        Treatment matrix
    desired_W : numpy.ndarray
        Desired treatment matrix
    detrending_options : list of bool or None
        Detrending options
    detrending_param_ranges : dict or None
        Detrending configuration ranges
    main_param_ranges : dict or None
        Main parameter configuration ranges
    n_time_blocks : int or None
        Number of time blocks for cross-validation
    time_blocks : list of tuples or None
        Pre-defined time blocks for validation
    desired_W_2 : numpy.ndarray or None
        Second desired treatment matrix
        
    Raises
    ------
    ValueError
        If inputs are invalid
    """
    # Validate matrix shapes
    if Y.shape != W.shape:
        logger.error(f"Shape mismatch: Y {Y.shape}, W {W.shape}")
        raise ValueError(f"Shape mismatch: Y {Y.shape}, W {W.shape}")
    
    if desired_W.shape != W.shape:
        logger.error(f"Shape mismatch: desired_W {desired_W.shape}, W {W.shape}")
        raise ValueError(f"Shape mismatch: desired_W {desired_W.shape}, W {W.shape}")
    
    if desired_W_2 is not None and desired_W_2.shape != W.shape:
        logger.error(f"Shape mismatch: desired_W_2 {desired_W_2.shape}, W {W.shape}")
        raise ValueError(f"Shape mismatch: desired_W_2 {desired_W_2.shape}, W {W.shape}")
    
    # Check for missing/invalid values
    if np.any(np.isnan(Y)) or np.any(np.isnan(W)):
        logger.error("Missing values detected in Y or W")
        raise ValueError("Missing values detected in Y or W")
    
    if not np.all(np.isin(W, [0, 1])):
        logger.error("W must contain only binary values")
        raise ValueError("W must contain only binary values")
    
    if not np.all(np.isin(desired_W, [0, 1])):
        logger.error("desired_W must contain only binary values")
        raise ValueError("desired_W must contain only binary values")
    
    if desired_W_2 is not None and not np.all(np.isin(desired_W_2, [0, 1])):
        logger.error("desired_W_2 must contain only binary values")
        raise ValueError("desired_W_2 must contain only binary values")
    
    # Check time blocks parameters
    if time_blocks is None and n_time_blocks is None:
        logger.error("Either n_time_blocks or time_blocks must be provided")
        raise ValueError("Either n_time_blocks or time_blocks must be provided")
    
    if time_blocks is not None:
        # Validate time blocks format
        if not isinstance(time_blocks, list) or not all(isinstance(block, tuple) and len(block) == 2 for block in time_blocks):
            logger.error("time_blocks must be a list of (start, end) tuples")
            raise ValueError("time_blocks must be a list of (start, end) tuples")
        
        # Validate time blocks values
        T = Y.shape[1]
        for start, end in time_blocks:
            if not (isinstance(start, int) and isinstance(end, int)):
                logger.error("Time block values must be integers")
                raise ValueError("Time block values must be integers")
            if start < 0 or end > T or start >= end:
                logger.error(f"Invalid time block: ({start}, {end}). Must satisfy 0 <= start < end <= {T}")
                raise ValueError(f"Invalid time block: ({start}, {end}). Must satisfy 0 <= start < end <= {T}")
    elif n_time_blocks is not None:
        if not isinstance(n_time_blocks, int) or n_time_blocks < 2:
            logger.error(f"n_time_blocks must be a positive integer of at least 2, got {n_time_blocks}")
            raise ValueError(f"n_time_blocks must be a positive integer of at least 2, got {n_time_blocks}")
    
    # Check detrending configuration
    if detrending_options is not None:
        if not all(isinstance(x, bool) for x in detrending_options):
            logger.error("detrending_options must contain only boolean values")
            raise ValueError("detrending_options must contain only boolean values")
            
        if any(detrending_options) and detrending_param_ranges is None:
            logger.error("detrending_param_ranges required when detrending is enabled")
            raise ValueError("detrending_param_ranges required when detrending is enabled")
    
    # Validate main parameter ranges
    if main_param_ranges is None:
        logger.error("main_param_ranges must be provided")
        raise ValueError("main_param_ranges must be provided")
    
    # Validate that initial treatment patterns match
    if main_param_ranges is not None:
        # Calculate maximum possible lag
        max_n_lags_Y = max(main_param_ranges.get('n_lags_Y_range', [0]))
        max_n_lags_W = max(main_param_ranges.get('n_lags_W_range', [0]))
        
        # If detrending is enabled, consider its lags too
        if detrending_options is not None and any(detrending_options) and detrending_param_ranges is not None:
            max_n_lags_Y = max(max_n_lags_Y, max(detrending_param_ranges.get('n_lags_Y_range', [0])))
        
        # Calculate max_lag as per the formula in CFEEstimator.fit
        max_lag = max(max_n_lags_Y, max_n_lags_W - 1)
        
        # Check if initial treatment patterns match
        if max_lag > 0 and not np.array_equal(W[:, :max_lag], desired_W[:, :max_lag]):
            error_msg = (
                f"Initial {max_lag} columns of W and desired_W must match for proper counterfactual estimation. "
                "These columns represent pre-treatment conditions to initialize the recursive estimation."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if desired_W_2 is not None and max_lag > 0 and not np.array_equal(W[:, :max_lag], desired_W_2[:, :max_lag]):
            error_msg = (
                f"Initial {max_lag} columns of W and desired_W_2 must match for proper counterfactual estimation. "
                "These columns represent pre-treatment conditions to initialize the recursive estimation."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    logger.debug("Input validation successful")