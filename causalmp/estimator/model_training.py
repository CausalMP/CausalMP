import numpy as np
from sklearn.linear_model import Ridge
from typing import Optional, Tuple, Callable
import logging

from .feature_engineering import FeatureEngineering
from .statistical_moments import StatisticalMoments

class ModelTraining:
    """Manage model training and prediction for counterfactual estimation.
    
    This class handles the training of Ridge regression models for counterfactual
    estimation, including feature preparation, model fitting, and prediction.
    """
    
    def __init__(self):
        """Initialize ModelTraining with required components."""
        self.feature_engineer = FeatureEngineering()
        self.logger = logging.getLogger(__name__)
        
    def train(
        self,
        Y_train_1: np.ndarray,
        W_train_1: np.ndarray,
        n_lags_Y: int,
        n_lags_W: int,
        moment_order_p_Y: int,
        moment_order_u_Y: int,
        moment_order_p_W: int,
        moment_order_u_W: int,
        n_batch: int,
        batch_size: int,
        psi: Callable,
        ridge_alpha: float = 1e-4,
        Y_train_2: Optional[np.ndarray] = None,
        W_train_2: Optional[np.ndarray] = None,
    ) -> Ridge:
        """Train a Ridge regression model using features from one or two training blocks.
        
        Parameters
        ----------
        Y_train_1 : numpy.ndarray
            First block of outcome data
        W_train_1 : numpy.ndarray
            First block of treatment data
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
        psi : callable
            Feature engineering function
        ridge_alpha : float, default=1e-4
            Ridge regression penalty
        Y_train_2 : numpy.ndarray, optional
            Second block of outcome data
        W_train_2 : numpy.ndarray, optional
            Second block of treatment data
            
        Returns
        -------
        sklearn.linear_model.Ridge
            Trained Ridge regression model
        """
        self.logger.debug("Preparing training features")
        
        # Initialize feature lists
        feature_blocks = []
        target_blocks = []
        
        # Process first training block
        if Y_train_1 is not None:
            self.logger.debug("Processing first training block")
            X_block, y_block = self.feature_engineer.prepare_block_features(
                Y_block=Y_train_1,
                W_block=W_train_1,
                n_lags_Y=n_lags_Y,
                n_lags_W=n_lags_W,
                moment_order_p_Y=moment_order_p_Y,
                moment_order_u_Y=moment_order_u_Y,
                moment_order_p_W=moment_order_p_W,
                moment_order_u_W=moment_order_u_W,
                n_batch=n_batch,
                batch_size=batch_size,
                psi=psi
            )
            feature_blocks.append(X_block)
            target_blocks.append(y_block)
        
        # Process second training block if provided    
        if Y_train_2 is not None:
            self.logger.debug("Processing second training block")
            X_block, y_block = self.feature_engineer.prepare_block_features(
                Y_block=Y_train_2,
                W_block=W_train_2,
                n_lags_Y=n_lags_Y,
                n_lags_W=n_lags_W,
                moment_order_p_Y=moment_order_p_Y,
                moment_order_u_Y=moment_order_u_Y,
                moment_order_p_W=moment_order_p_W,
                moment_order_u_W=moment_order_u_W,
                n_batch=n_batch,
                batch_size=batch_size,
                psi=psi
            )
            feature_blocks.append(X_block)
            target_blocks.append(y_block)
        
        # Combine features and targets
        X = np.vstack(feature_blocks)
        y = np.vstack(target_blocks)
        
        self.logger.debug(f"Training Ridge regression model with alpha={ridge_alpha}")
        
        # Fit and return Ridge regression model
        model = Ridge(alpha=ridge_alpha)
        model.fit(X, y)
        
        return model

    def predict(
        self,
        model: Ridge,
        Y_pre: np.ndarray,
        W_pre: np.ndarray,
        desired_W: np.ndarray,
        n_lags_Y: int,
        n_lags_W: int,
        moment_order_p_Y: int,
        moment_order_u_Y: int,
        moment_order_p_W: int,
        moment_order_u_W: int,
        psi: Callable,
        batch_weights: Optional[np.ndarray] = None,
        time_block: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Generate counterfactual predictions using fitted model and dynamic reconstruction.
        
        Parameters
        ----------
        model : sklearn.linear_model.Ridge
            Trained Ridge regression model
        Y_pre : numpy.ndarray
            Historical outcome data for initialization
        W_pre : numpy.ndarray
            Historical treatment data for initialization
        desired_W : numpy.ndarray
            Target treatment matrix
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
        psi : callable
            Feature engineering function
        batch_weights : numpy.ndarray, optional
            Weight matrix for batch-specific predictions
        time_block : tuple, optional
            Time block for prediction slicing
            
        Returns
        -------
        numpy.ndarray
            Matrix of predicted counterfactual outcomes
        """
        self.logger.debug("Generating predictions")
        
        # Setup initialization
        T_desired = desired_W.shape[1]
        if batch_weights is None:
            batch_weights = np.ones((1, Y_pre.shape[0])) / Y_pre.shape[0]
        
        # Combine historical and desired treatments
        Prediction_W = np.concatenate([W_pre, desired_W], axis=1)
        
        self.logger.debug(f"Prediction_W shape: {Prediction_W.shape}")
        
        # Get number of test batches
        Kbatch_validation = batch_weights.shape[0]
        
        # Determine lag requirements
        max_lag = max(n_lags_Y, n_lags_W-1)
        
        # Initialize prediction arrays
        estimation_CFE = np.zeros((Kbatch_validation, T_desired + max_lag))
        
        # Initialize moment tracking
        max_moment_order_Y = max(moment_order_p_Y, moment_order_u_Y)
        max_moment_order_W = max(moment_order_p_W, moment_order_u_W)
        pop_moments_Y = self._initialize_moments(T_desired + max_lag, max_moment_order_Y)
        unit_moments_Y = self._initialize_moments(T_desired + max_lag, max_moment_order_Y)
        pop_moments_W = self._initialize_moments(T_desired + max_lag, max_moment_order_W)
        unit_moments_W = self._initialize_moments(T_desired + max_lag, max_moment_order_W)
        
        # Generate predictions for each batch
        for i in range(Kbatch_validation):
            self.logger.debug(f"Processing batch {i+1}/{Kbatch_validation}")
            
            # Process current batch
            current_batch = batch_weights[i, :]
            
            # Calculate initial population statistics for Y
            pop_stats_Y = StatisticalMoments.calculate(
                np.ones((1, Y_pre.shape[0])) / Y_pre.shape[0],
                Y_pre,
                moments_to_calculate=set(['mean', 'variance', 'skewness', 'kurtosis'][:moment_order_p_Y])
            )
            
            # Calculate initial batch statistics for Y
            unit_stats_Y = StatisticalMoments.calculate(
                current_batch.reshape(1, -1),
                Y_pre,
                moments_to_calculate=set(['mean', 'variance', 'skewness', 'kurtosis'][:moment_order_u_Y])
            )
            
            # Calculate initial population statistics for W
            pop_stats_W = StatisticalMoments.calculate(
                np.ones((1, Prediction_W.shape[0])) / Prediction_W.shape[0],
                Prediction_W,
                moments_to_calculate=set(['mean', 'variance', 'skewness', 'kurtosis'][:moment_order_p_W])
            )
            
            # Calculate initial batch statistics for W
            unit_stats_W = StatisticalMoments.calculate(
                current_batch.reshape(1, -1),
                Prediction_W,
                moments_to_calculate=set(['mean', 'variance', 'skewness', 'kurtosis'][:moment_order_u_W])
            )
            
            # Set initial conditions
            self._set_initial_conditions(pop_moments_Y, pop_stats_Y, max_lag)
            self._set_initial_conditions(unit_moments_Y, unit_stats_Y, max_lag)
            self._set_initial_conditions(pop_moments_W, pop_stats_W, T_desired + max_lag)
            self._set_initial_conditions(unit_moments_W, unit_stats_W, T_desired + max_lag)
            
            # Dynamic reconstruction
            for t in range(max_lag - 1, T_desired + max_lag - 1):
                # Generate features
                pop_features, batch_features = self._generate_features(
                    t, pop_moments_Y, unit_moments_Y, pop_moments_W, unit_moments_W,
                    n_lags_Y, n_lags_W,
                    moment_order_p_Y, moment_order_u_Y,
                    moment_order_p_W, moment_order_u_W
                )
                
                # Make predictions
                X_t_pop = np.column_stack(psi(*pop_features))
                pop_predictions = model.predict(X_t_pop).reshape(-1)
                                
                X_t_batch = np.column_stack(psi(*batch_features))
                batch_predictions = model.predict(X_t_batch).reshape(-1)
                                
                # Update moments
                self._update_moments(
                    pop_moments_Y, unit_moments_Y,
                    pop_predictions, batch_predictions,
                    t, moment_order_p_Y, moment_order_u_Y
                )
            
            # Store predictions
            estimation_CFE[i, :] = unit_moments_Y['mean']
            
        self.logger.debug(f"Final estimation shape: {estimation_CFE.shape}")
        self.logger.debug(f"Final estimation: {estimation_CFE}")
        
        # Return appropriate time slice
        if time_block is None:
            return estimation_CFE[0, :T_desired+max_lag]
        elif time_block[0] == 0:
            return estimation_CFE[:, :T_desired+max_lag]
        else:
            return estimation_CFE[:, max_lag:T_desired+max_lag]
    
    def _initialize_moments(self, length: int, max_order: int) -> dict:
        """Initialize moment tracking dictionaries."""
        moments = {
            'mean': np.zeros(length),
            'variance': np.zeros(length) if max_order >= 2 else None,
            'skewness': np.zeros(length) if max_order >= 3 else None,
            'kurtosis': np.zeros(length) if max_order >= 4 else None
        }
        return moments
    
    def _set_initial_conditions(
        self,
        moments: dict,
        stats: dict,
        max_lag: int
    ) -> None:
        """Set initial conditions for moment tracking."""
        for moment_type, values in stats.items():
            if moments[moment_type] is not None:
                moments[moment_type][:max_lag] = values.flatten()[-max_lag:]
    
    def _generate_features(
        self,
        t: int,
        pop_moments_Y: dict,
        unit_moments_Y: dict,
        pop_moments_W: dict,
        unit_moments_W: dict,
        n_lags_Y: int,
        n_lags_W: int,
        moment_order_p_Y: int,
        moment_order_u_Y: int,
        moment_order_p_W: int,
        moment_order_u_W: int
    ) -> Tuple[list, list]:
        """Generate population and batch-level features."""
        pop_features = []
        batch_features = []
        
        # Population-level Y features
        for m in range(moment_order_p_Y):
            moment_type = ['mean', 'variance', 'skewness', 'kurtosis'][m]
            if pop_moments_Y[moment_type] is not None:
                for lag in range(n_lags_Y):
                    feature = pop_moments_Y[moment_type][t-lag].reshape(-1,1)
                    pop_features.append(feature)
                    batch_features.append(feature)
        
        # Population-level W features
        for m in range(moment_order_p_W):
            moment_type = ['mean', 'variance', 'skewness', 'kurtosis'][m]
            if pop_moments_W[moment_type] is not None:
                for lag in range(n_lags_W):
                    feature = pop_moments_W[moment_type][t-lag+1].reshape(-1,1)
                    pop_features.append(feature)
                    batch_features.append(feature)
        
        # Batch-level Y features
        for m in range(moment_order_u_Y):
            moment_type = ['mean', 'variance', 'skewness', 'kurtosis'][m]
            if unit_moments_Y[moment_type] is not None:
                for lag in range(n_lags_Y):
                    pop_feat = pop_moments_Y[moment_type][t-lag].reshape(-1,1)
                    batch_feat = unit_moments_Y[moment_type][t-lag].reshape(-1,1)
                    pop_features.append(pop_feat)
                    batch_features.append(batch_feat)
        
        # Batch-level W features
        for m in range(moment_order_u_W):
            moment_type = ['mean', 'variance', 'skewness', 'kurtosis'][m]
            if unit_moments_W[moment_type] is not None:
                for lag in range(n_lags_W):
                    pop_feat = pop_moments_W[moment_type][t-lag+1].reshape(-1,1)
                    batch_feat = unit_moments_W[moment_type][t-lag+1].reshape(-1,1)
                    pop_features.append(pop_feat)
                    batch_features.append(batch_feat)
        
        return pop_features, batch_features
    
    def _update_moments(
        self,
        pop_moments_Y: dict,
        unit_moments_Y: dict,
        pop_predictions: np.ndarray,
        batch_predictions: np.ndarray,
        t: int,
        moment_order_p_Y: int,
        moment_order_u_Y: int
    ) -> None:
        """Update moment tracking with new predictions."""
        # Update population-level Y moments
        for m in range(moment_order_p_Y):
            moment_type = ['mean', 'variance', 'skewness', 'kurtosis'][m]
            if pop_moments_Y[moment_type] is not None:
                pop_moments_Y[moment_type][t+1] = pop_predictions[m]
        
        # Update batch-level Y moments
        for m in range(moment_order_u_Y):
            moment_type = ['mean', 'variance', 'skewness', 'kurtosis'][m]
            if unit_moments_Y[moment_type] is not None:
                unit_moments_Y[moment_type][t+1] = batch_predictions[m]