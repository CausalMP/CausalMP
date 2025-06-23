import numpy as np
from typing import Dict, Optional, Tuple, Union, Callable
import logging
from abc import ABC, abstractmethod

from .feature_engineering import FeatureEngineering
from .model_training import ModelTraining
from .batch_generator import BatchGenerator

class CounterfactualEstimator(ABC):
    """Base class for all counterfactual estimators."""
    
    def __init__(self, config: Dict):
        """
        Initialize the estimator with configuration parameters.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary containing estimation parameters
        """
        self.config = config
    
    @abstractmethod
    def estimate(
        self, 
        Y: np.ndarray,
        W: np.ndarray,
        desired_W: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Estimate counterfactual outcomes.
        
        Parameters
        ----------
        Y : numpy.ndarray, shape (N, T)
            Matrix of observed outcomes
        W : numpy.ndarray, shape (N, T)
            Matrix of observed treatments
        desired_W : numpy.ndarray, shape (N, T)
            Target treatment matrix
            
        Returns
        -------
        numpy.ndarray
            Estimated counterfactual outcomes
        """
        pass

class CFEEstimator(CounterfactualEstimator):
    """
    Main Counterfactual Evolution (CFE) estimator using moment-based features.
    
    This implementation uses moment-based features and Ridge regression for 
    estimation, with optional detrending capabilities.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the CFE estimator.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary containing:
            main_config : dict
                - n_lags_Y : int, number of outcome lags
                - n_lags_W : int, number of treatment lags
                - moment_order_p_Y : int, order of population-level Y moments
                - moment_order_u_Y : int, order of unit-level Y moments
                - moment_order_p_W : int, order of population-level W moments
                - moment_order_u_W : int, order of unit-level W moments
                - interaction_term_p : int or None, population interaction terms
                - interaction_term_u : int or None, unit interaction terms
                - n_batch  : int, number of batches
                - batch_size : int, size of each batch
                - ridge_alpha : float, Ridge regression penalty
            detrending_config : dict or empty dict, optional
                Same structure as main_config if detrending is used
        """
        super().__init__(config)
        self.feature_engineer = FeatureEngineering()
        self.model_trainer = ModelTraining()
        self.batch_generator = BatchGenerator()
        self.logger = logging.getLogger(__name__)
        
    def fit(
        self,
        Y: np.ndarray,
        W: np.ndarray,
        psi: Optional[Callable] = None,
        return_model_terms: bool = False
    ) -> Union['CFEEstimator', Tuple['CFEEstimator', list]]:
        """
        Fit the CFE estimator using observed outcomes and treatments.
        
        Parameters
        ----------
        Y : numpy.ndarray, shape (N, T)
            Matrix of observed outcomes
        W : numpy.ndarray, shape (N, T)
            Matrix of observed treatments
        psi : callable, optional
            Custom feature engineering function
        return_model : bool, default=False
            Whether to return model terms
            
        Returns
        -------
        CFEEstimator or tuple
            If return_model=False:
                Returns self for method chaining
            If return_model=True:
                Tuple (self, model_terms) where model_terms describes the features
        """
        self.N, self.T = np.shape(Y)
        
        # Setup configurations
        config = self.config['main_config']
        self.n_lags_Y = config['n_lags_Y']
        self.n_lags_W = config['n_lags_W']
        self.moment_order_p_Y = config['moment_order_p_Y']
        self.moment_order_u_Y = config['moment_order_u_Y']
        self.moment_order_p_W = config['moment_order_p_W']
        self.moment_order_u_W = config['moment_order_u_W']
        self.interaction_term_p = config['interaction_term_p']
        self.interaction_term_u = config['interaction_term_u']
        self.n_batch = config['n_batch']
        self.batch_size = config['batch_size']
        self.ridge_alpha = config['ridge_alpha']
        
        self.logger.debug("Setting up configurations")
        
        # Initialize feature engineering function
        self.psi = psi if psi is not None else lambda *args: self.feature_engineer.create_moment_features(
            self.n_lags_Y, self.n_lags_W,
            self.moment_order_p_Y, self.moment_order_u_Y,
            self.moment_order_p_W, self.moment_order_u_W,
            self.interaction_term_p, self.interaction_term_u,
            args
        )
        
        # Handle detrending if specified
        if self.config.get('detrending_config'):
            self.logger.debug("Applying detrending")
            # Create detrending estimator
            from .cfe_semi_recursive import CFESemiRecursiveEstimator
            self.detrend_estimator = CFESemiRecursiveEstimator(
                self.config['detrending_config']
            )
            
            # Estimate and remove trend
            trend_result = self.detrend_estimator.estimate(
                Y=Y,
                W=W,
                desired_W=np.zeros_like(W),
                return_model_terms=return_model_terms
            )
            
            self.trend_estimation = trend_result[0] if return_model_terms else trend_result
            # Update Y for training
            Y = Y - self.trend_estimation
        
        # Generate model terms if requested
        model_terms = None
        if return_model_terms:
            model_terms = self.psi(args=None)
        
        # Sort units by treatment exposure
        unit_exposures = np.sum(W, axis=1)
        sorted_indices = np.argsort(-unit_exposures)
        ordered_W = W[sorted_indices]
        ordered_Y = Y[sorted_indices]
        
        # Split data into pre-period and training sets
        self.pre_period_length = max(self.n_lags_Y, self.n_lags_W - 1)
        self.Y_pre = ordered_Y[:, :self.pre_period_length]
        self.W_pre = ordered_W[:, :self.pre_period_length]
        Y_train = ordered_Y
        W_train = ordered_W
        
        # Train model
        self.logger.debug("Training model with parameters: %s", {
            'n_lags_Y': self.n_lags_Y,
            'n_lags_W': self.n_lags_W,
            'ridge_alpha': self.ridge_alpha
        })
        self.model = self.model_trainer.train(
            Y_train_1=Y_train,
            Y_train_2=None,
            W_train_1=W_train,
            W_train_2=None,
            n_lags_Y=self.n_lags_Y,
            n_lags_W=self.n_lags_W,
            moment_order_p_Y=self.moment_order_p_Y,
            moment_order_u_Y=self.moment_order_u_Y,
            moment_order_p_W=self.moment_order_p_W,
            moment_order_u_W=self.moment_order_u_W,
            n_batch=self.n_batch,
            batch_size=self.batch_size,
            psi=self.psi,
            ridge_alpha=self.ridge_alpha
        )

        self.logger.debug(f"The trained model coefficients are: \n{self.model.coef_} \nand the model terms are: \n{model_terms}")
        
        if return_model_terms:
            return self, model_terms
        return self

    def predict(
        self,
        desired_W: np.ndarray
    ) -> np.ndarray:
        """
        Predict counterfactual evolution for desired treatment pattern.
        
        Parameters
        ----------
        desired_W : numpy.ndarray, shape (N, T)
            Target treatment matrix
            
        Returns
        -------
        numpy.ndarray
            Predicted counterfactual outcomes
            
        Raises
        ------
        ValueError
            If model has not been fitted
        """
        if not hasattr(self, 'model'):
            raise ValueError("Model must be fitted before making predictions")
            
        # Extract desired treatment for prediction period
        W_desired = desired_W[:, self.pre_period_length:]
        
        # Generate predictions
        self.logger.debug("Generating predictions for desired treatment")
        estimation_CFE = self.model_trainer.predict(
            model=self.model,
            Y_pre=self.Y_pre,
            W_pre=self.W_pre,
            desired_W=W_desired,
            n_lags_Y=self.n_lags_Y,
            n_lags_W=self.n_lags_W,
            moment_order_p_Y=self.moment_order_p_Y,
            moment_order_u_Y=self.moment_order_u_Y,
            moment_order_p_W=self.moment_order_p_W,
            moment_order_u_W=self.moment_order_u_W,
            psi=self.psi
        )
        
        # Reapply trend if detrending was used
        if hasattr(self, 'trend_estimation'):
            self.logger.info("Reapplying trend")
            estimation_CFE = estimation_CFE + self.trend_estimation
            
        return estimation_CFE

    def estimate(
        self, 
        Y: np.ndarray, 
        W: np.ndarray, 
        desired_W: np.ndarray,
        psi: Optional[Callable] = None,
        return_model: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
        """
        Estimate counterfactual evolution using moment-based features.
        
        This method combines fit and predict steps for convenience.
        
        Parameters
        ----------
        Y : numpy.ndarray, shape (N, T)
            Matrix of observed outcomes
        W : numpy.ndarray, shape (N, T)
            Matrix of observed treatments
        desired_W : numpy.ndarray, shape (N, T)
            Target treatment matrix
        psi : callable, optional
            Custom feature engineering function
        return_model : bool, default=False
            Whether to return model terms along with estimates
            
        Returns
        -------
        numpy.ndarray or tuple
            If return_model=False:
                numpy.ndarray of counterfactual estimates
            If return_model=True:
                Tuple (estimates, model_terms)
        """
        # Fit the model
        fit_result = self.fit(Y, W, psi, return_model)
        
        # Get model terms if requested
        model_terms = None
        if return_model:
            _, model_terms = fit_result
        
        # Generate predictions
        predictions = self.predict(desired_W)
        
        # Return results
        if return_model:
            return predictions, model_terms
        return predictions