from typing import Dict, List, Set
from itertools import product
import logging

logger = logging.getLogger(__name__)

def _validate_configuration_inputs(
    detrending_options: List[bool],
    detrending_param_ranges: Dict,
    main_param_ranges: Dict
) -> None:
    """Validate configuration generation inputs."""
    if not isinstance(detrending_options, (list, tuple)) or not all(isinstance(x, bool) for x in detrending_options):
        raise ValueError("detrending_options must be a list of boolean values")
        
    validate_parameter_ranges(main_param_ranges, prefix="Main ")
    if any(detrending_options):
        validate_parameter_ranges(detrending_param_ranges, prefix="Detrending ", is_detrending=True)

    logger.debug(f"Input parameters - Detrending options: {detrending_options}, Detrending param ranges: {detrending_param_ranges}, Main param ranges: {main_param_ranges}")
    logger.debug("Validation of configuration inputs successful")

def validate_parameter_ranges(
    params: Dict,
    prefix: str = "",
    is_detrending: bool = False
) -> None:
    """Validate parameter ranges in configuration dictionaries.
    
    Parameters
    ----------
    params : dict
        Dictionary containing parameter ranges
    prefix : str, default=""
        Prefix for error messages
    is_detrending : bool, default=False
        Whether this is for detrending configuration
        
    Raises
    ------
    ValueError
        If any parameter ranges are invalid
    """
    # Define required ranges based on configuration type
    if is_detrending:
        required_ranges = {
            'n_lags_Y_range': 'number of Y lags',
            'interaction_term_p_range': 'population interaction terms',
            'interaction_term_u_range': 'unit interaction terms',
            'n_batch_range': 'number of batches',
            'batch_size_range': 'batch size',
            'ridge_alpha_range': 'ridge alpha'
        }
    else:
        required_ranges = {
            'n_lags_Y_range': 'number of Y lags',
            'n_lags_W_range': 'number of W lags',
            'moment_order_p_Y_range': 'population moment order for Y',
            'moment_order_u_Y_range': 'unit moment order for Y',
            'moment_order_p_W_range': 'population moment order for W',
            'moment_order_u_W_range': 'unit moment order for W',
            'interaction_term_p_range': 'population interaction terms',
            'interaction_term_u_range': 'unit interaction terms',
            'n_batch_range': 'number of batches',
            'batch_size_range': 'batch size',
            'ridge_alpha_range': 'ridge alpha'
        }
    
    # Check for required ranges
    missing_ranges = [desc for name, desc in required_ranges.items() if name not in params]
    if missing_ranges:
        raise ValueError(f"{prefix}Missing required ranges: {', '.join(missing_ranges)}")
        
    # Check for empty ranges
    empty_ranges = [desc for name, desc in required_ranges.items() if not params[name]]
    if empty_ranges:
        raise ValueError(f"{prefix}Empty ranges for: {', '.join(empty_ranges)}")
            
    # Validate numeric ranges
    for range_name, values in params.items():
        if not all(isinstance(x, (int, float, type(None))) for x in values):
            raise ValueError(f"{prefix}Invalid type in {range_name}")
        if any(x is not None and x < 0 for x in values):
            raise ValueError(f"{prefix}Negative values not allowed in {range_name}")
            
    # Validate moment orders if not detrending
    if not is_detrending:
        for order in (params['moment_order_p_Y_range'] + params['moment_order_u_Y_range']
                      + params['moment_order_p_W_range'] + params['moment_order_u_W_range']):
            if order > 4:
                raise ValueError("Moment orders cannot exceed 4")

def generate_configurations(
    detrending_options: List[bool], 
    detrending_param_ranges: Dict, 
    main_param_ranges: Dict
) -> Dict:
    """Generate nested configurations grouped by detrending settings.
    
    Parameters
    ----------
    detrending_options : list
        List of boolean values for detrending option
    detrending_param_ranges : dict
        Dictionary containing parameter ranges for detrending
    main_param_ranges : dict
        Dictionary containing parameter ranges for main estimation
        
    Returns
    -------
    dict
        Dictionary with detrending configurations as keys and lists of complete 
        configurations as values
        
    Raises
    ------
    ValueError
        If configurations are invalid
    """
    logger.debug("Generating configurations")
    
    # Validate input parameters
    _validate_configuration_inputs(detrending_options, detrending_param_ranges, main_param_ranges)
    
    # Initialize configurations dictionary
    grouped_configurations = {}
    
    # Extract ranges
    main_param_ranges = _extract_main_param_ranges(main_param_ranges)
    
    # Generate configurations for each combination
    for moment_order_p_Y in main_param_ranges['moment_order_p_Y_range']:
        for moment_order_u_Y in main_param_ranges['moment_order_u_Y_range']:
            for moment_order_p_W in main_param_ranges['moment_order_p_W_range']:
                for moment_order_u_W in main_param_ranges['moment_order_u_W_range']:
                    # Filter interaction terms
                    valid_interactions = _get_valid_interactions(
                        moment_order_p_Y,
                        moment_order_u_Y,
                        main_param_ranges['interaction_term_p_range'],
                        main_param_ranges['interaction_term_u_range']
                    )
                    
                    # Generate main combinations
                    main_combinations = _generate_main_combinations(
                        main_param_ranges,
                        moment_order_p_Y,
                        moment_order_u_Y,
                        moment_order_p_W,
                        moment_order_u_W,
                        valid_interactions
                    )
                    
                    # Process detrending options
                    for detrending in detrending_options:
                        detrend_key = _process_detrending_option(
                            detrending,
                            detrending_param_ranges,
                            main_combinations,
                            grouped_configurations
                        )
                        
    logger.debug(f"Generated {len(grouped_configurations)} configuration groups")
    return grouped_configurations

def _extract_main_param_ranges(main_param_ranges: Dict) -> Dict:
    """Extract and validate main parameter ranges."""
    return {
        'n_lags_Y_range': main_param_ranges['n_lags_Y_range'],
        'n_lags_W_range': main_param_ranges['n_lags_W_range'],
        'moment_order_p_Y_range': main_param_ranges['moment_order_p_Y_range'],
        'moment_order_u_Y_range': main_param_ranges['moment_order_u_Y_range'],
        'moment_order_p_W_range': main_param_ranges['moment_order_p_W_range'],
        'moment_order_u_W_range': main_param_ranges['moment_order_u_W_range'],
        'interaction_term_p_range': main_param_ranges['interaction_term_p_range'],
        'interaction_term_u_range': main_param_ranges['interaction_term_u_range'],
        'n_batch_range': main_param_ranges['n_batch_range'],
        'batch_size_range': main_param_ranges['batch_size_range'],
        'ridge_alpha_range': main_param_ranges['ridge_alpha_range']
    }

def _get_valid_interactions(
    moment_order_p_Y: int,
    moment_order_u_Y: int,
    interaction_range_p: List,
    interaction_range_u: List
) -> Dict:
    """Get valid interaction terms based on moment orders.
    
    Note: We only consider interactions between first moment of W (mean) and different moments of Y.
    """
    return {
        'p': [term for term in interaction_range_p if term is None or term <= moment_order_p_Y],
        'u': [term for term in interaction_range_u if term is None or term <= moment_order_u_Y]
    }

def _generate_main_combinations(
    main_param_ranges: Dict,
    moment_order_p_Y: int,
    moment_order_u_Y: int,
    moment_order_p_W: int,
    moment_order_u_W: int,
    valid_interactions: Dict
) -> List:
    """Generate all valid main parameter combinations."""
    return list(product(
        main_param_ranges['n_lags_Y_range'],
        main_param_ranges['n_lags_W_range'],
        [moment_order_p_Y],
        [moment_order_u_Y],
        [moment_order_p_W],
        [moment_order_u_W],
        valid_interactions['p'],
        valid_interactions['u'],
        main_param_ranges['n_batch_range'],
        main_param_ranges['batch_size_range'],
        main_param_ranges['ridge_alpha_range']
    ))

def _process_detrending_option(
    detrending: bool,
    detrending_param_ranges: Dict,
    main_combinations: List,
    grouped_configurations: Dict
) -> None:
    """Process detrending option and update configurations."""
    if detrending:
        detrend_combinations = list(product(
            detrending_param_ranges['n_lags_Y_range'],
            detrending_param_ranges['interaction_term_p_range'],
            detrending_param_ranges['interaction_term_u_range'],
            detrending_param_ranges['n_batch_range'],
            detrending_param_ranges['batch_size_range'],
            detrending_param_ranges['ridge_alpha_range']
        ))
        
        for detrend_params in detrend_combinations:
            detrend_config = _create_detrend_config(detrend_params)
            detrend_key = tuple(sorted(detrend_config.items()))
            if detrend_key not in grouped_configurations:
                grouped_configurations[detrend_key] = []
            
            for main_param_ranges in main_combinations:
                grouped_configurations[detrend_key].append({
                    'detrending_config': detrend_config,
                    'main_config': _create_main_config(main_param_ranges)
                })
    else:
        detrend_key = None
        if detrend_key not in grouped_configurations:
            grouped_configurations[detrend_key] = []
        
        for main_param_ranges in main_combinations:
            grouped_configurations[detrend_key].append({
                'detrending_config': {},
                'main_config': _create_main_config(main_param_ranges)
            })

def _create_detrend_config(params: tuple) -> Dict:
    """Create detrending configuration from parameters."""
    n_lags_Y, interaction_term_p, interaction_term_u, n_batch, batch_size, ridge_alpha = params
    return {
        'detrending': True,
        'n_lags_Y': n_lags_Y,
        'interaction_term_p': interaction_term_p,
        'interaction_term_u': interaction_term_u,
        'n_batch': n_batch,
        'batch_size': batch_size,
        'ridge_alpha': ridge_alpha
    }

def _create_main_config(params: tuple) -> Dict:
    """Create main configuration from parameters."""
    n_lags_Y, n_lags_W, moment_p_Y, moment_u_Y, moment_p_W, moment_u_W, interaction_term_p, interaction_term_u, n_batch, batch_size, ridge_alpha = params
    return {
        'n_lags_Y': n_lags_Y,
        'n_lags_W': n_lags_W,
        'moment_order_p_Y': moment_p_Y,
        'moment_order_u_Y': moment_u_Y,
        'moment_order_p_W': moment_p_W,
        'moment_order_u_W': moment_u_W,
        'interaction_term_p': interaction_term_p,
        'interaction_term_u': interaction_term_u,
        'n_batch': n_batch,
        'batch_size': batch_size,
        'ridge_alpha': ridge_alpha
    }