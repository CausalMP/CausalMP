import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import logging

from .batch_generator import BatchGenerator
from .model_training import ModelTraining
from .cfe_semi_recursive import CFESemiRecursiveEstimator
from .feature_engineering import FeatureEngineering

class CrossValidator:
    """Handle cross-validation for counterfactual evolution estimation.
    
    This class manages the cross-validation process for finding optimal model
    configurations, including optional detrending and main estimation parameters.
    """
    
    def __init__(self, loss_function: Optional[Callable] = None):
        """
        Initialize CrossValidator.
        
        Parameters
        ----------
        loss_function : callable, optional
            Custom loss function of form f(true_values, predicted_values) -> float
            If None, uses root mean squared error
        """
        self.loss_function = loss_function or self._default_loss_function
        self.model_trainer = ModelTraining()
        self.feature_engineer = FeatureEngineering()
        self.logger = logging.getLogger(__name__)
    
    def validate(
        self,
        Y: np.ndarray,
        W: np.ndarray,
        grouped_configurations: Dict,
        time_blocks_list: List[Tuple[int, int]],
        n_validation_batch: int = 1,
        visualize: bool = False
    ) -> Tuple[Dict, List]:
        """
        Performs cross-validation for counterfactual evolution estimation.
        
        Parameters
        ----------
        Y : numpy.ndarray, shape (N, T)
            Matrix of outcomes
        W : numpy.ndarray, shape (N, T)
            Matrix of treatments
        grouped_configurations : dict
            Dictionary of configurations grouped by detrending settings
        time_blocks_list : list of tuples
            List of (start, end) time blocks for validation
        n_validation_batch: int, default=1
            Number of validation batches to create
        visualize : bool, default=False
            Whether to create visualization of results
            
        Returns
        -------
        tuple
            - best_config : dict
                Configuration parameters of best performing estimator
            - best_model_terms : list
                LaTeX formatted terms describing model structure
            - best_score : float
            The validation score of the best model
        """
        # Get dimensions
        N = Y.shape[0]
        
        # Sort units by treatment exposure
        self.logger.debug("Sorting units by treatment exposure")
        unit_exposures = np.sum(W, axis=1)
        sorted_indices = np.argsort(-unit_exposures)
        ordered_W = W[sorted_indices]
        ordered_Y = Y[sorted_indices]
        
        # Generate validation batches
        self.logger.debug(f"Generating {n_validation_batch} validation batches")
        batching_matrix = BatchGenerator.generate(N, n_validation_batch, partitioning=True)
        
        # Calculate ground truth by batch averaging
        CFE = np.dot(batching_matrix, ordered_Y)
        
        # Initialize tracking variables
        best_score = float('inf')
        best_config = None
        best_estimation_CFE = None
        best_model_terms = [[], []]  # [detrending terms, main estimation terms]
        
        # Cross-validation process
        self.logger.debug("Starting cross-validation process")
        for detrend_key, configs in grouped_configurations.items():
            self.logger.debug(f"Processing detrending configuration: {detrend_key}")
            
            # Apply detrending if specified
            if detrend_key is not None:
                # Extract detrending configuration
                detrending_config = configs[0]['detrending_config']
                
                # Estimate trend using zero treatment
                trend_estimation, detrending_model_terms = self._estimate_trend(
                    ordered_Y, ordered_W,
                    detrending_config
                )
                
                # Remove trend component
                ordered_Y_detrended = ordered_Y - trend_estimation
            else:
                ordered_Y_detrended = ordered_Y
                detrending_model_terms = []
            
            # Evaluate each configuration
            for config in configs:
                self.logger.debug(f"Evaluating configuration: {config}")
                estimation_CFE = np.zeros_like(CFE)
                
                # Estimate CFE for each validation block
                for time_block in time_blocks_list:
                    estimation_CFE[:, time_block[0]:time_block[1]], main_model_terms = self._estimate_block(
                        ordered_Y_detrended, ordered_W,
                        batching_matrix, config, time_block
                    )
                
                # Reapply trend if detrending was used
                if detrend_key is not None:
                    estimation_CFE = estimation_CFE + trend_estimation
                
                # Calculate performance score
                score = self.loss_function(CFE, estimation_CFE)
                
                # Update best model if current score is better
                if score < best_score:
                    self.logger.info(f"New best score: {score}")
                    best_score = score
                    best_config = config
                    best_estimation_CFE = estimation_CFE
                    best_model_terms[0] = detrending_model_terms
                    best_model_terms[1] = main_model_terms
        
        # Visualize results if requested
        if visualize:
            self.logger.debug("Visualizing results")
            self._visualize_results(CFE, best_estimation_CFE)
        
        self.logger.debug("Cross-validation process completed")
        return best_config, best_model_terms, best_score
    
    def _default_loss_function(
        self,
        true: np.ndarray,
        pred: np.ndarray
    ) -> float:
        """Default root mean squared error loss function."""
        return np.sqrt(np.mean(np.square(true - pred)))
    
    def _estimate_trend(
        self,
        Y: np.ndarray,
        W: np.ndarray,
        config: Dict
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Estimates trend using zero treatment counterfactual.
        
        Parameters
        ----------
        Y : numpy.ndarray
            Outcome matrix
        W : numpy.ndarray
            Treatment matrix
        config : dict
            Detrending configuration
            
        Returns
        -------
        tuple
            (trend_estimation, model_terms)
        """
        cfe_sr_estimator = CFESemiRecursiveEstimator(config)
        trend_estimation, detrending_model_terms = cfe_sr_estimator.estimate(
            Y=Y,
            W=W,
            desired_W=np.zeros_like(W),
            return_model_terms=True
        )
        return trend_estimation, detrending_model_terms
    
    def _estimate_block(
        self,
        Y: np.ndarray,
        W: np.ndarray,
        batch_matrix: np.ndarray,
        config: Dict,
        time_block: Tuple[int, int]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Estimates CFE for a specific time block.
        
        Parameters
        ----------
        Y : numpy.ndarray
            Outcome matrix
        W : numpy.ndarray
            Treatment matrix
        batch_matrix : numpy.ndarray
            Batch weights matrix for the validation sets
        config : dict
            Estimation configuration
        time_block : tuple
            Time block (start, end)
            
        Returns
        -------
        tuple
            (block_estimation, model_terms)
        """
        # Get key parameters from config
        main_config = config['main_config']
        n_lags_Y = main_config['n_lags_Y']
        n_lags_W = main_config['n_lags_W']
        moment_order_p_Y = main_config['moment_order_p_Y']
        moment_order_u_Y = main_config['moment_order_u_Y']
        moment_order_p_W = main_config['moment_order_p_W']
        moment_order_u_W = main_config['moment_order_u_W']
        interaction_term_p = main_config.get('interaction_term_p')
        interaction_term_u = main_config.get('interaction_term_u')
        
        # Create psi function with interaction terms
        psi = lambda *args: self.feature_engineer.create_moment_features(
            n_lags_Y, n_lags_W,
            moment_order_p_Y, moment_order_u_Y,
            moment_order_p_W, moment_order_u_W,
            interaction_term_p,
            interaction_term_u,
            args
        )
        
        # Split data for training
        y_train_pre, y_train_post, w_train_pre, w_train_post, w_test, y_pre, w_pre = \
            self._split_data(Y, W, time_block, n_lags_Y, n_lags_W)
        
        # Extract only the parameters needed for training
        train_params = {
            'Y_train_1': y_train_pre,
            'Y_train_2': y_train_post,
            'W_train_1': w_train_pre,
            'W_train_2': w_train_post,
            'n_lags_Y': n_lags_Y,
            'n_lags_W': n_lags_W,
            'moment_order_p_Y': moment_order_p_Y,
            'moment_order_u_Y': moment_order_u_Y,
            'moment_order_p_W': moment_order_p_W,
            'moment_order_u_W': moment_order_u_W,
            'n_batch': main_config['n_batch'],
            'batch_size': main_config['batch_size'],
            'ridge_alpha': main_config['ridge_alpha'],
            'psi': psi
        }
        
        # Train model
        model = self.model_trainer.train(**train_params)
        
        # Extract prediction parameters
        predict_params = {
            'model': model,
            'Y_pre': y_pre,
            'W_pre': w_pre,
            'desired_W': w_test,
            'n_lags_Y': n_lags_Y,
            'n_lags_W': n_lags_W,
            'moment_order_p_Y': moment_order_p_Y,
            'moment_order_u_Y': moment_order_u_Y,
            'moment_order_p_W': moment_order_p_W,
            'moment_order_u_W': moment_order_u_W,
            'batch_weights': batch_matrix,
            'time_block': time_block,
            'psi': psi
        }
        
        # Generate predictions
        block_estimation = self.model_trainer.predict(**predict_params)
        
        # Generate model terms
        model_terms = self._generate_model_terms(main_config)
        
        return block_estimation, model_terms
    
    def _split_data(
        self,
        Y: np.ndarray,
        W: np.ndarray,
        time_block: Tuple[int, int],
        n_lags_Y: int,
        n_lags_W: int
    ) -> Tuple[np.ndarray, ...]:
        """
        Splits data into training and test sets based on time block.
        
        Parameters
        ----------
        Y : numpy.ndarray
            Outcome matrix
        W : numpy.ndarray
            Treatment matrix
        time_block : tuple
            Time block indices
        n_lags_Y : int
            Number of outcome lags
        n_lags_W : int
            Number of treatment lags
            
        Returns
        -------
        tuple
            Training and test data splits
        """
        T = Y.shape[1]
        pre_period_length = max(n_lags_Y, n_lags_W - 1)
        
        if time_block[0] == 0 and time_block[1] == T:  # The entire period is the training set
            t_test_start = pre_period_length
            y_train_pre = Y
            y_train_post = None
            w_train_pre = W
            w_train_post = None
        elif time_block[0] == 0:  # Block at start
            t_test_start = pre_period_length
            y_train_pre = None
            y_train_post = Y[:, time_block[1]:]
            w_train_pre = None
            w_train_post = W[:, time_block[1]:]
        elif time_block[1] == T:  # Block at end
            t_test_start = time_block[0]
            y_train_pre = Y[:, :time_block[0]]
            y_train_post = None
            w_train_pre = W[:, :time_block[0]]
            w_train_post = None
        else:  # Block in middle
            t_test_start = time_block[0]
            y_train_pre = Y[:, :time_block[0]]
            y_train_post = Y[:, time_block[1]:]
            w_train_pre = W[:, :time_block[0]]
            w_train_post = W[:, time_block[1]:]
        
        # Get test and pre-period data
        w_test = W[:, t_test_start:time_block[1]]
        y_pre = Y[:, t_test_start-pre_period_length:t_test_start]
        w_pre = W[:, t_test_start-pre_period_length:t_test_start]
        
        return y_train_pre, y_train_post, w_train_pre, w_train_post, w_test, y_pre, w_pre
    
    def _generate_model_terms(self, config: Dict) -> List[str]:
        """Generates LaTeX model terms from configuration."""
        return FeatureEngineering().create_moment_features(
            n_lags_Y=config['n_lags_Y'],
            n_lags_W=config['n_lags_W'],
            moment_order_p_Y=config['moment_order_p_Y'],
            moment_order_u_Y=config['moment_order_u_Y'],
            moment_order_p_W=config['moment_order_p_W'],
            moment_order_u_W=config['moment_order_u_W'],
            interaction_term_p=config.get('interaction_term_p'),
            interaction_term_u=config.get('interaction_term_u'),
            args=None
        )
    
    def _visualize_results(
        self,
        true_CFE: np.ndarray,
        estimated_CFE: np.ndarray
    ) -> None:
        """Creates visualization comparing ground truth vs estimated CFE."""
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        plt.figure(figsize=(10, 6))
        x_values = np.arange(true_CFE.shape[1])
        n_batch = true_CFE.shape[0]
        
        # Create color maps for true and estimated values with wider color range
        true_colors = cm.Blues(np.linspace(0.3, 1.0, n_batch))
        est_colors = cm.Reds(np.linspace(0.3, 1.0, n_batch))
        
        # Different marker styles for each batch
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', '8']
        
        # Plot each batch with a different color, marker and label
        for i in range(n_batch):
            # Use different markers for each batch (cycling if more batches than markers)
            marker_idx = i % len(markers)
            
            plt.plot(x_values, true_CFE[i], '-', color=true_colors[i], marker=markers[marker_idx],
                    label=f'Ground-truth CFE (Batch {i+1})', linewidth=2, markersize=6)
            plt.plot(x_values, estimated_CFE[i], '--', color=est_colors[i], marker=markers[marker_idx],
                    label=f'Estimated CFE (Batch {i+1})', linewidth=2, markersize=6)
        
        plt.title('Ground-truth vs Best Estimated CFE Comparison', fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('CFE Value', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()