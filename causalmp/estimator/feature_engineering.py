import numpy as np
from typing import List, Optional, Tuple, Union
import logging

from .batch_generator import BatchGenerator
from .statistical_moments import StatisticalMoments

class FeatureEngineering:
    """Handle feature creation and management for counterfactual estimation.
    
    This class manages the creation of features for counterfactual estimation,
    including moment-based features, interaction terms, and LaTeX model terms
    for documentation.
    """
    
    def __init__(self):
        """Initialize FeatureEngineering with logger."""
        self.logger = logging.getLogger(__name__)
    
    def create_moment_features(
        self,
        n_lags_Y: int,
        n_lags_W: int,
        moment_order_p_Y: int,
        moment_order_u_Y: int,
        moment_order_p_W: int,
        moment_order_u_W: int,
        interaction_term_p: Optional[int],
        interaction_term_u: Optional[int],
        args: Optional[List[np.ndarray]] = None
    ) -> Union[List[np.ndarray], List[str]]:
        """
        Creates moment-based features including both outcome and treatment lags.
        
        Parameters
        ----------
        n_lags_Y : int
            Number of lags for outcome variables
        n_lags_W : int
            Number of lags for treatment variables 
        moment_order_p_Y : int
            Highest order of moments for population level Y
        moment_order_u_Y : int
            Highest order of moments for unit level Y
        moment_order_p_W : int
            Highest order of moments for population level W
        moment_order_u_W : int
            Highest order of moments for unit level W
        interaction_term_p : int or None
            Maximum order of population-level interaction terms
        interaction_term_u : int or None
            Maximum order of batch-level interaction terms
        args : sequence of arrays, optional
            Input arrays containing moments
        
        Returns
        -------
        list
            Either list of features or list of LaTeX model terms
        """
        self.logger.debug("Entering create_moment_features method.")
        features = []
        model_terms = ["\\nu_{t+1} \sim "]
        
        # Population level outcome moments
        for m in range(moment_order_p_Y):
            for lag in range(n_lags_Y):
                if args is not None:
                    start_idx = m * n_lags_Y
                    features.append(args[start_idx + lag])
                
                # LaTeX notation
                if m == 0:
                    term = f"\\nu_{{t-{lag}}}" if lag > 0 else "\\nu_t"
                else:
                    term = f"\\nu_{{t-{lag}}}^{{({m+1})}}" if lag > 0 else f"\\nu_t^{{({m+1})}}"
                model_terms.append(term)
        
        # Population level treatment moments
        treatment_offset = moment_order_p_Y * n_lags_Y
        for m in range(moment_order_p_W):
            for lag in range(n_lags_W):
                if args is not None:
                    start_idx = treatment_offset + m * n_lags_W
                    features.append(args[start_idx + lag])
                
                if m == 0:
                    if lag == 0:
                        term = "p_{t+1}"
                    elif lag == 1:
                        term = "p_{t}"
                    else:
                        term = f"p_{{t-{lag-1}}}"
                else:
                    if lag == 0:
                        term = f"p_{{t+1}}^{{({m+1})}}"
                    elif lag == 1:
                        term = f"p_{{t}}^{{({m+1})}}"
                    else:
                        term = f"p_{{t-{lag-1}}}^{{({m+1})}}"
                model_terms.append(term)
        
        # Batch/unit level outcome moments
        batch_offset = treatment_offset + moment_order_p_W * n_lags_W
        for m in range(moment_order_u_Y):
            for lag in range(n_lags_Y):
                if args is not None:
                    start_idx = batch_offset + m * n_lags_Y
                    features.append(args[start_idx + lag])
                
                if m == 0:
                    term = f"\\nu_{{t-{lag}}}^S" if lag > 0 else "\\nu_t^S"
                else:
                    term = f"\\nu_{{t-{lag}}}^{{S,({m+1})}}" if lag > 0 else f"\\nu_t^{{S,({m+1})}}"
                model_terms.append(term)
        
        # Batch/unit level treatment moments
        batch_treatment_offset = batch_offset + moment_order_u_Y * n_lags_Y
        for m in range(moment_order_u_W):
            for lag in range(n_lags_W):
                if args is not None:
                    start_idx = batch_treatment_offset + m * n_lags_W
                    features.append(args[start_idx + lag])
                
                if m == 0:
                    if lag == 0:
                        term = "p_{t+1}^S"
                    elif lag == 1:
                        term = "p_{t}^S"
                    else:
                        term = f"p_{{t-{lag-1}}}^S"
                else:
                    if lag == 0:
                        term = f"p_{{t+1}}^{{S,({m+1})}}"
                    elif lag == 1:
                        term = f"p_{{t}}^{{S,({m+1})}}"
                    else:
                        term = f"p_{{t-{lag-1}}}^{{S,({m+1})}}"
                model_terms.append(term)
        
        # Population level interaction terms
        if interaction_term_p is not None:
            for m in range(min(interaction_term_p, moment_order_p_Y)):
                if args is not None:
                    moment_idx = m * n_lags_Y
                    features.append(args[moment_idx] * args[treatment_offset])
                
                if m == 0:
                    term = "\\nu_t \\times p_{t+1}"
                else:
                    term = f"\\nu_t^{{({m+1})}} \\times p_{{t+1}}"
                model_terms.append(term)
        
        # Batch/unit level interaction terms
        if interaction_term_u is not None:
            for m in range(min(interaction_term_u, moment_order_u_Y)):
                if args is not None:
                    batch_moment_idx = batch_offset + m * n_lags_Y
                    features.append(args[batch_moment_idx] * args[batch_treatment_offset])
                
                if m == 0:
                    term = "\\nu_t^S \\times p_{t+1}^S"
                else:
                    term = f"\\nu_t^{{S,({m+1})}} \\times p_{{t+1}}^S"
                model_terms.append(term)
        
        if args is None:
            self.logger.debug("Exiting create_moment_features method.")
            return model_terms
        self.logger.debug("Exiting create_moment_features method.")
        return features

    def prepare_block_features(
        self,
        Y_block: np.ndarray,
        W_block: np.ndarray,
        n_lags_Y: int,
        n_lags_W: int,
        moment_order_p_Y: int,
        moment_order_u_Y: int,
        moment_order_p_W: int,
        moment_order_u_W: int,
        n_batch: int,
        batch_size: int,
        psi: Optional[callable] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepares features for a single contiguous block of data using batch processing.
        
        Parameters
        ----------
        Y_block : numpy.ndarray, shape (N, T_block)
            Block of outcome data
        W_block : numpy.ndarray, shape (N, T_block)
            Block of treatment data
        n_lags_Y : int
            Number of outcome lags
        n_lags_W : int
            Number of treatment lags
        moment_order_p_Y : int
            Order of population-level Y moments
        moment_order_u_Y : int
            Order of unit-level Y moments
        moment_order_p_W : int
            Order of population-level W moments
        moment_order_u_W : int
            Order of unit-level W moments
        n_batch  : int
            Number of batches
        batch_size : int
            Size of each batch
        psi : callable, optional
            Custom feature engineering function
            
        Returns
        -------
        tuple
            (X_block, y_block) feature and target matrices
        """        
        self.logger.debug("Entering prepare_block_features method.")
        
        # Ensure sufficient data for lags
        if Y_block.shape[1] <= n_lags_Y:
            raise ValueError(f"Block length {Y_block.shape[1]} insufficient for {n_lags_Y} lags")
        if W_block.shape[1] <= n_lags_W:
            raise ValueError(f"Block length {W_block.shape[1]} insufficient for {n_lags_W} lags")
        
        # Calculate effective lag periods
        max_lag = max(n_lags_Y, n_lags_W)
        
        # Generate normalized batch weights
        batch_weights_matrix= BatchGenerator.generate(
            Y_block.shape[0], n_batch, batch_size, partitioning=False
        )
        
        # Initialize feature lists
        features = []
        
        # Population weights for statistics
        pop_weights = np.ones((1, Y_block.shape[0])) / Y_block.shape[0]
        
        # Population-level outcome moments
        for m in range(moment_order_p_Y):
            for lag in range(1, n_lags_Y + 1):
                Y_slice = Y_block[:, max_lag-lag:-lag]
                pop_stats = StatisticalMoments.calculate(
                    pop_weights, Y_slice,
                    moments_to_calculate=set(['mean', 'variance', 'skewness', 'kurtosis'][:m+1])
                )
                moment_type = ['mean', 'variance', 'skewness', 'kurtosis'][m]
                moment_stat = pop_stats[moment_type]
                pop_feature = np.ones((n_batch, 1)).dot(moment_stat.reshape(1, -1)).T.reshape(-1, 1)
                features.append(pop_feature)
        
        # Population-level treatment moments
        for m in range(moment_order_p_W):
            for lag in range(n_lags_W):
                W_slice = W_block[:, max_lag-lag:W_block.shape[1]-lag]
                pop_stats = StatisticalMoments.calculate(
                    pop_weights, W_slice,
                    moments_to_calculate=set(['mean', 'variance', 'skewness', 'kurtosis'][:m+1])
                )
                moment_type = ['mean', 'variance', 'skewness', 'kurtosis'][m]
                moment_stat = pop_stats[moment_type]
                pop_feature = np.ones((n_batch, 1)).dot(moment_stat.reshape(1, -1)).T.reshape(-1, 1)
                features.append(pop_feature)
        
        # Batch-level outcome moments
        for m in range(moment_order_u_Y):
            for lag in range(1, n_lags_Y + 1):
                Y_slice = Y_block[:, max_lag-lag:-lag]
                batch_stats = StatisticalMoments.calculate(
                    batch_weights_matrix, Y_slice,
                    moments_to_calculate=set(['mean', 'variance', 'skewness', 'kurtosis'][:m+1])
                )
                moment_type = ['mean', 'variance', 'skewness', 'kurtosis'][m]
                moment_stat = batch_stats[moment_type]
                batch_feature = moment_stat.T.reshape(-1, 1)
                features.append(batch_feature)
        
        # Batch-level treatment moments
        for m in range(moment_order_u_W):
            for lag in range(n_lags_W):
                W_slice = W_block[:, max_lag-lag:W_block.shape[1]-lag]
                batch_stats = StatisticalMoments.calculate(
                    batch_weights_matrix, W_slice,
                    moments_to_calculate=set(['mean', 'variance', 'skewness', 'kurtosis'][:m+1])
                )
                moment_type = ['mean', 'variance', 'skewness', 'kurtosis'][m]
                moment_stat = batch_stats[moment_type]
                batch_feature = moment_stat.T.reshape(-1, 1)
                features.append(batch_feature)
        
        # Calculate target moments
        Y_next = Y_block[:, max_lag:]
        target_stats = StatisticalMoments.calculate(batch_weights_matrix, Y_next)
        
        # Prepare target matrix
        max_moment_order = max(moment_order_p_Y, moment_order_u_Y)
        y_moments = [target_stats['mean'].T.reshape(-1, 1)]
        if max_moment_order >= 2:
            y_moments.append(target_stats['variance'].T.reshape(-1, 1))
        if max_moment_order >= 3:
            y_moments.append(target_stats['skewness'].T.reshape(-1, 1))
        if max_moment_order >= 4:
            y_moments.append(target_stats['kurtosis'].T.reshape(-1, 1))
        
        # Apply feature engineering if provided
        if psi is not None:
            X_block = np.column_stack(psi(*features))
        else:
            X_block = np.column_stack(features)
        
        y_block = np.column_stack(y_moments)
        
        self.logger.debug("Exiting prepare_block_features method.")
        return X_block, y_block