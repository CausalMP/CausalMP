import numpy as np
from typing import List
from sklearn.linear_model import LinearRegression

def dinm_estimate(Y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Calculate the Difference in Means (DinM) treatment effect.
    
    This method implements the classic Difference in Means estimator, comparing
    outcomes between treated and control units at each time point.
    
    Parameters
    ----------
    Y : numpy.ndarray, shape (N, T)
        Matrix of outcomes for N units over T time periods
    W : numpy.ndarray, shape (N, T)
        Matrix of treatments (binary) for N units over T time periods
        
    Returns
    -------
    numpy.ndarray, shape (T,)
        Time-varying treatment effect estimates
        
    Notes
    -----
    - Returns nan for periods with no variation in treatment
    - Uses cross-sectional comparison at each time point
    """
    N, T = Y.shape
    TTE = np.full(T, np.nan)
    
    for t in range(T):
        n_treated = np.sum(W[:, t])
        n_control = N - n_treated
        
        if n_treated > 0 and n_control > 0:
            treated_mean = np.sum(Y[:, t] * W[:, t]) / n_treated
            control_mean = np.sum(Y[:, t] * (1 - W[:, t])) / n_control
            TTE[t] = treated_mean - control_mean
    
    return TTE

def ht_estimate(
    Y: np.ndarray,
    W: np.ndarray,
    stage_time_blocks: List[int],
    design: List[float]
) -> np.ndarray:
    """Implement the Horvitz-Thompson (HT) estimator.
    
    This method implements the Horvitz-Thompson estimator, which weights outcomes
    by the inverse probability of treatment to estimate causal effects.
    
    Parameters
    ----------
    Y : numpy.ndarray, shape (N, T)
        Matrix of outcomes for N units over T time periods
    W : numpy.ndarray, shape (N, T)
        Matrix of treatments (binary) for N units over T time periods
    stage_time_blocks : list
        Time points where treatment probability changes
    design : list
        Treatment probabilities corresponding to each time period
        
    Returns
    -------
    numpy.ndarray, shape (T,)
        Time-varying treatment effect estimates
        
    Notes
    -----
    - Returns nan for periods with zero or one treatment probability
    - Accounts for changing treatment probabilities over time
    - Uses inverse probability weighting
    """
    N, T = Y.shape
    TTE = np.full(T, np.nan)
    current_index = 0
    
    for t in range(T):
        # Update probability index if needed
        if current_index < len(stage_time_blocks) - 1 and t >= stage_time_blocks[current_index]:
            current_index += 1
            
        p = design[current_index]
        
        # Check if treatment probability allows for estimation
        if 0 < p < 1:
            # Calculate inverse probability weighted means
            treated_mean = np.mean(Y[:, t] * W[:, t]) / p
            control_mean = np.mean(Y[:, t] * (1 - W[:, t])) / (1 - p)
            TTE[t] = treated_mean - control_mean
    
    return TTE

def basic_cmp_estimate(Y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Calculate the TTE using basic Causal message-passing approach.
    
    This method implements a simplified version of the Causal Message Passing
    approach using a single regression on the mean outcomes.
    
    Parameters
    ----------
    Y : numpy.ndarray, shape (N, T)
        Matrix of outcomes for N units over T time periods
    W : numpy.ndarray, shape (N, T)
        Matrix of treatments (binary) for N units over T time periods
        
    Returns
    -------
    numpy.ndarray, shape (T,)
        Time-varying total treatment effect estimates
        
    Notes
    -----
    - Uses a single autoregressive model with treatment and interaction terms
    - Assumes first-order autoregressive dynamics
    - Recursively constructs counterfactual outcomes and treatment effects
    """
    # Calculate mean outcomes and treatments
    nu_hat = np.mean(Y, axis=0)
    p = np.mean(W, axis=0)
    T = Y.shape[1]
    
    # Create features and response for regression
    features = nu_hat[:T-1]
    response = nu_hat[1:]
    
    # Create feature matrix
    featMatrix = np.zeros((len(features), 3))
    featMatrix[:, 0] = p[1:T]  # Current treatment effect (lambda)
    featMatrix[:, 1] = features  # Autoregressive effect (xi)
    featMatrix[:, 2] = featMatrix[:, 0] * featMatrix[:, 1]  # Interaction term (gamma)
    
    # Fit linear regression
    reg = LinearRegression().fit(featMatrix, response)
    
    # Extract coefficients
    hat_lambda = reg.coef_[0]
    hat_xi = reg.coef_[1]
    hat_gamma = reg.coef_[2]
    
    # Calculate counterfactual outcomes and treatment effects
    TTE = np.zeros_like(nu_hat)
    nu_tilda = np.zeros_like(nu_hat)
    
    # Set initial conditions
    nu_tilda[0] = nu_hat[0]
    
    # Recursively calculate treatment effects
    for t in range(1, T):
        nu_tilda[t] = nu_hat[t] + hat_xi * (nu_tilda[t-1] - nu_hat[t-1]) + \
                     hat_lambda * (1 - p[t]) + hat_gamma * (nu_tilda[t-1] - p[t] * nu_hat[t-1])
        
        TTE[t] = TTE[t-1] * hat_xi + hat_lambda + hat_gamma * nu_tilda[t-1]
    
    return TTE