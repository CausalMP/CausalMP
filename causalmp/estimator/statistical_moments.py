import numpy as np
from typing import Set, Dict, Optional

class StatisticalMoments:
    """Calculate and manage statistical moments for counterfactual estimation.
    
    This class handles the calculation of various statistical moments (mean,
    variance, skewness, kurtosis) for counterfactual estimation.
    
    Attributes
    ----------
    VALID_MOMENTS : set
        Set of valid moment types that can be calculated
    EPSILON : float
        Small constant to avoid division by zero
    """
    
    VALID_MOMENTS = {'mean', 'variance', 'skewness', 'kurtosis'}
    EPSILON = 1e-8
    
    @classmethod
    def calculate(
        cls,
        batch_weights_matrix: np.ndarray,
        Y: np.ndarray,
        moments_to_calculate: Optional[Set[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Calculate specified statistical moments using weighted averaging.
        
        Parameters
        ----------
        batch_weights_matrix: numpy.ndarray, shape (Nb, N)
            Normalized batch weights, where each row sums to 1
        Y : numpy.ndarray, shape (N, T)
            Panel data with N units over T time periods
        moments_to_calculate : set, optional
            Specify which moments to calculate. If None, calculates all moments.
            Options: 'mean', 'variance', 'skewness', 'kurtosis'
        
        Returns
        -------
        dict
            Dictionary containing the requested moments, each as (Nb by T) matrix
            
        Raises
        ------
        ValueError
            If invalid moments are not specified or inputs are invalid
        """
        # Set default moments if none specified
        if moments_to_calculate is None:
            moments_to_calculate = cls.VALID_MOMENTS.copy()
            
        # Validate inputs
        cls._validate_inputs(batch_weights_matrix, Y, moments_to_calculate)
        
        results = {}
        
        # Calculate mean if needed or if required for higher moments
        if 'mean' in moments_to_calculate or len(moments_to_calculate - {'mean'}) > 0:
            mean = np.dot(batch_weights_matrix, Y)
            if 'mean' in moments_to_calculate:
                results['mean'] = mean
        
        # Calculate variance if needed or if required for higher moments
        if 'variance' in moments_to_calculate or {'skewness', 'kurtosis'} & moments_to_calculate:
            raw_moment2 = np.dot(batch_weights_matrix, Y**2)
            variance = raw_moment2 - mean**2
            if 'variance' in moments_to_calculate:
                results['variance'] = variance
        
        # Calculate skewness if requested
        if 'skewness' in moments_to_calculate:
            raw_moment3 = np.dot(batch_weights_matrix, Y**3)
            variance_stable = np.where(variance > cls.EPSILON, variance, cls.EPSILON)
            skewness = np.where(
                variance > cls.EPSILON,
                (raw_moment3 - 3*mean*variance - mean**3) / (variance_stable**1.5),
                0.0  # Set skewness to 0 for very small variances
            )
            results['skewness'] = skewness
        
        # Calculate kurtosis if requested
        if 'kurtosis' in moments_to_calculate:
            raw_moment4 = np.dot(batch_weights_matrix, Y**4)
            variance_stable = np.maximum(variance**2, cls.EPSILON)
            kurtosis = (raw_moment4 - 4*mean*raw_moment3 + 
                       6*mean**2*raw_moment2 - 3*mean**4) / variance_stable
            results['kurtosis'] = kurtosis
        
        return results
    
    @staticmethod
    def _validate_inputs(
        batch_weights_matrix: np.ndarray,
        Y: np.ndarray,
        moments_to_calculate: Optional[Set[str]] = None
    ) -> None:
        """Validate inputs for moment calculations.
        
        Parameters
        ----------
        batch_weights_matrix: numpy.ndarray
            Normalized batch weights matrix
        Y : numpy.ndarray
            Panel data matrix
        moments_to_calculate : set, optional
            Set of moments to calculate
            
        Raises
        ------
        ValueError
            If inputs are invalid
        """
        if batch_weights_matrix.shape[1] != Y.shape[0]:
            raise ValueError(
                f"Incompatible shapes: batch_weights_matrix{batch_weights_matrix.shape} and Y {Y.shape}"
            )
        
        if not np.allclose(batch_weights_matrix.sum(axis=1), 1):
            raise ValueError("batch_weights_matrixrows must sum to 1")
            
        if moments_to_calculate is not None:
            if not moments_to_calculate.issubset(StatisticalMoments.VALID_MOMENTS):
                raise ValueError(
                    f"Invalid moments specified. Valid options are: "
                    f"{StatisticalMoments.VALID_MOMENTS}"
                )