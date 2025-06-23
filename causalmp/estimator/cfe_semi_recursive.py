import numpy as np
from typing import Dict, Optional, Tuple, Union
import logging
from sklearn.linear_model import Ridge

from .batch_generator import BatchGenerator

class CFESemiRecursiveEstimator:
    """Semi-recursive estimation implementation for counterfactual evolution.
    
    This estimator uses a semi-recursive approach for counterfactual estimation,
    particularly useful for trend estimation.
    """
    
    def __init__(self, config: Dict):
        """Initialize the semi-recursive estimator.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary containing:
            - n_lags_Y: int, number of lags for outcome variables
            - interaction_term_p: bool, population-level interactions
            - interaction_term_u: bool, unit-level interactions
            - n_batch: int, number of batches
            - batch_size: int, size of each batch
            - ridge_alpha: float, regularization parameter
        """
        self.config = config
        self._validate_config()
        self.logger = logging.getLogger(__name__)
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")
            
    def fit(
        self,
        Y: np.ndarray,
        W: np.ndarray,
        return_model_terms: bool = False
    ) -> Union['CFESemiRecursiveEstimator', Tuple['CFESemiRecursiveEstimator', list]]:
        """Fit the semi-recursive estimator to the data."""
        # Get dimensions
        self.N, self.T = np.shape(Y)
        self.logger.debug(f"Fitting model with N={self.N}, T={self.T}")
        
        # Unpack configuration parameters
        self.n_lags_Y = self.config["n_lags_Y"]
        self.interaction_term_p = self.config["interaction_term_p"]
        self.interaction_term_u = self.config["interaction_term_u"]
        self.n_batch = self.config["n_batch"]
        self.batch_size = self.config["batch_size"]
        self.ridge_alpha = self.config["ridge_alpha"]
        
        # Sort units by treatment exposure
        unit_exposures = np.sum(W, axis=1)
        sorted_indices = np.argsort(-unit_exposures)
        self.ordered_W = W[sorted_indices]
        self.ordered_Y = Y[sorted_indices]
        
        # Generate batch assignments
        self.batching_matrix = BatchGenerator.generate(
            self.N,
            n_batch=self.n_batch,
            batch_size=self.batch_size
        )
        
        # Compute batch statistics
        self.batch_sample_mean = np.dot(self.batching_matrix, self.ordered_Y)
        self.batch_treatment = np.dot(self.batching_matrix, self.ordered_W)
        
        # Initialize feature containers
        features = []
        model_terms = ["\\nu_{t+1} \sim "] if return_model_terms else None
        
        # Generate features and model terms
        features, model_terms = self._generate_features(return_model_terms)
        self.logger.debug(f"Generated model terms: {model_terms}")
        
        # Prepare feature matrix and target variable
        X = np.column_stack(features)
        y = self.batch_sample_mean[:, self.n_lags_Y:].T.reshape(-1,)
        
        # Train Ridge regression model
        self.logger.debug("Training Ridge regression model")
        self.model = Ridge(alpha=self.ridge_alpha).fit(X, y)
        self.coefs = self.model.coef_
        self.logger.debug(f"Model coefficients: {self.coefs}")
        
        if return_model_terms:
            return self, model_terms
        return self
    
    def predict(
        self,
        desired_W: np.ndarray
    ) -> np.ndarray:
        """Generate predictions using the fitted semi-recursive model.
        
        Parameters
        ----------
        desired_W : numpy.ndarray, shape (N, T)
            Matrix of desired treatments
            
        Returns
        -------
        numpy.ndarray
            Predicted counterfactual outcomes
        """
        if not hasattr(self, 'model'):
            raise ValueError("Model must be fitted before prediction")
        
        # Initialize arrays for reconstruction
        nu_hat = np.mean(self.ordered_Y, axis=0)
        pi_hat = np.mean(self.ordered_W, axis=0)
        Desired_pi = np.mean(desired_W, axis=0)
        estimation_CFE = np.zeros_like(nu_hat)
        
        # Set initial conditions
        estimation_CFE[:self.n_lags_Y] = nu_hat[:self.n_lags_Y]
        
        # Recursive prediction
        for t in range(self.n_lags_Y-1, self.T-1):
            estimation_CFE[t+1] = nu_hat[t+1]
            coef_idx = 0
            
            # Population-level adjustments
            for lag in range(self.n_lags_Y):
                estimation_CFE[t+1] += self.coefs[coef_idx] * (
                    estimation_CFE[t-lag] - nu_hat[t-lag]
                )
                coef_idx += 1
            
            estimation_CFE[t+1] += self.coefs[coef_idx] * (
                Desired_pi[t+1] - pi_hat[t+1]
            )
            coef_idx += 1
            
            # Batch-level adjustments
            for lag in range(self.n_lags_Y):
                estimation_CFE[t+1] += self.coefs[coef_idx] * (
                    estimation_CFE[t-lag] - nu_hat[t-lag]
                )
                coef_idx += 1
            
            estimation_CFE[t+1] += self.coefs[coef_idx] * (
                Desired_pi[t+1] - pi_hat[t+1]
            )
            coef_idx += 1
            
            # Population-level interaction effect
            if self.interaction_term_p:
                estimation_CFE[t+1] += self.coefs[coef_idx] * (
                    Desired_pi[t+1] * estimation_CFE[t] - 
                    pi_hat[t+1] * nu_hat[t]
                )
                coef_idx += 1
            
            # Batch-level interaction effect
            if self.interaction_term_u:
                estimation_CFE[t+1] += self.coefs[coef_idx] * (
                    Desired_pi[t+1] * estimation_CFE[t] - 
                    pi_hat[t+1] * nu_hat[t]
                )
        
        return estimation_CFE
    
    def estimate(
        self,
        Y: np.ndarray,
        W: np.ndarray,
        desired_W: np.ndarray,
        return_model_terms: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
        """Estimate counterfactual outcomes using semi-recursive approach.
        
        Parameters
        ----------
        Y : numpy.ndarray, shape (N, T)
            Matrix of ordered observed outcomes
        W : numpy.ndarray, shape (N, T)
            Matrix of ordered observed treatments
        desired_W : numpy.ndarray, shape (N, T)
            Matrix of desired treatments
        return_model : bool, default=False
            Whether to return model terms
            
        Returns
        -------
        numpy.ndarray or tuple
            If return_model=False:
                numpy.ndarray of predicted outcomes
            If return_model=True:
                Tuple (predictions, model_terms)
        """
        # Fit the model
        fit_result = self.fit(Y, W, return_model_terms)
        
        # Get model terms if requested
        model_terms = None
        if return_model_terms:
            _, model_terms = fit_result
        
        # Generate predictions
        predictions = self.predict(desired_W)
        self.logger.debug(f"Predictions of detrending model: {predictions}")
        
        if return_model_terms:
            return predictions, model_terms
        return predictions
    
    def _generate_features(
        self,
        return_model_terms: bool = False
    ) -> Tuple[list, Optional[list]]:
        """Generate features and model terms for semi-recursive estimation.
        
        Parameters
        ----------
        return_model : bool, default=False
            Whether to generate model terms
            
        Returns
        -------
        tuple
            (features, model_terms)
            features : list of numpy.ndarray
            model_terms : list of str or None
        """
        features = []
        model_terms = ["\\nu_{t+1} \sim "] if return_model_terms else None
        
        # Population-Level Features
        # Add lagged outcome features
        for lag in range(1, self.n_lags_Y + 1):
            if return_model_terms:
                term = f"\\nu_{{t-{lag-1}}}" if lag > 1 else "\\nu_t"
                model_terms.append(term)
            
            pop_lag = np.ones((self.n_batch, 1)).dot(
                np.mean(self.ordered_Y[:, self.n_lags_Y-lag:-lag], axis=0).reshape(1, -1)
            ).T.reshape(-1, 1)
            features.append(pop_lag)
        
        # Add treatment feature
        if return_model_terms:
            model_terms.append("p_{t+1}")
        
        pi_vec = np.ones((self.n_batch, 1)).dot(
            np.mean(self.ordered_W[:, self.n_lags_Y:], axis=0).reshape(1, -1)
        ).T.reshape(-1, 1)
        features.append(pi_vec)
        
        # Batch-Level Features
        # Add lagged outcome features
        for lag in range(1, self.n_lags_Y + 1):
            if return_model_terms:
                term = f"\\nu_{{t-{lag-1}}}^S" if lag > 1 else "\\nu_t^S"
                model_terms.append(term)
            
            batch_lag = self.batch_sample_mean[:, self.n_lags_Y-lag:-lag].T.reshape(-1, 1)
            features.append(batch_lag)
        
        # Add treatment feature
        if return_model_terms:
            model_terms.append("p_{t+1}^S")
        
        S_pi_vec = self.batch_treatment[:, self.n_lags_Y:].T.reshape(-1, 1)
        features.append(S_pi_vec)
        
        # Interaction Terms
        if self.interaction_term_p:
            if return_model_terms:
                model_terms.append("\\nu_t \\times p_{t+1}")
            pop_interaction = features[0] * features[self.n_lags_Y]
            features.append(pop_interaction)
        
        if self.interaction_term_u:
            if return_model_terms:
                model_terms.append("\\nu_t^S \\times p_{t+1}^S")
            batch_interaction = features[self.n_lags_Y + 1] * features[-1]
            features.append(batch_interaction)
        
        return features, model_terms