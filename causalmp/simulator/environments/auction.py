from typing import Dict, Optional, Any
import numpy as np
from .base_environment import BaseEnvironment

class AuctionEnvironment(BaseEnvironment):
    """Environment for simulating an auction model with multiple bidders.
    
    This environment implements a competitive market where multiple bidders participate 
    in auctions for objects. The auction mechanism creates a dynamic pricing system 
    where bidder interactions generate complex patterns of market influence.
    
    Attributes:
        ENVIRONMENT_TYPE (str): Identifier for the environment type
        tau (float): Treatment effect size
        sigma_v (float): Standard deviation for noise in valuations
        min_bid (float): Minimum bid increment
        sparsity (float): Sparsity parameter for bidder valuations
        auction_rounds (int): Number of auction rounds per time period
        valuation_mean (float): Desired mean for bidder valuations
        valuation_std_dev (float): Standard deviation for bidder valuations
        bidder_types (list): Array specifying the type of each bidder
        market_value_weight (float): Weight for market component in valuations
        private_value_weight (float): Weight for private component in valuations
        type_parameters (dict): Parameters for different bidder types
        valuations (np.ndarray): Matrix of bidder valuations for objects
    """
    
    ENVIRONMENT_TYPE = "Auction Model"
    
    def __init__(self, params: Dict[str, Any], seed: Optional[int] = None):
        """Initialize the auction environment."""
        super().__init__(params, seed)
        
        # Extract parameters specific to auction environment
        self.tau = params.get("tau", 0.1)
        self.sigma_v = params.get("sigma_v", 10)
        self.min_bid = params.get("min_bid", 1e-4)
        self.sparsity = params.get("sparsity", 0.75)
        self.auction_rounds = params.get("auction_rounds", 3)
        self.valuation_mean = params.get("valuation_mean", 200)
        self.valuation_std_dev = params.get("valuation_std_dev", 75)
        
        # Parameters for multi-component valuations
        self.market_value_weight = params.get("market_value_weight", 0.65)
        self.private_value_weight = params.get("private_value_weight", 0.35)
        self.bidder_types = None  # Will be generated in initialize_environment
        
        # Type-specific valuation parameters:
        # - standard: baseline bidders with no modification
        # - collector: enthusiasts who value items higher with more variability
        # - dealer: professional resellers with slightly lower, more consistent valuations
        # - investor: seeking financial return with higher but more variable valuations
        self.type_parameters = params.get("type_parameters", {
            "standard": {"mean_factor": 0.0, "std_dev_factor": 0.0},
            "collector": {"mean_factor": 0.25, "std_dev_factor": 0.35},  # 20% higher values, 30% more variable
            "dealer": {"mean_factor": -0.15, "std_dev_factor": -0.25},   # 10% lower values, 20% less variable
            "investor": {"mean_factor": 0.18, "std_dev_factor": 0.45}   # 15% higher values, 40% more variable
        })
        
        # Initialize matrices to None
        self.valuations = None
        self.noise = None

    def initialize_environment(self) -> None:
        """Initialize auction environment with bidder valuations and noise."""
        # Create local RNG with fixed seed for reproducibility
        local_rng = np.random.default_rng(self.seed)
        
        # Generate bidder types
        bidder_type_options = ["standard", "collector", "dealer", "investor"]
        bidder_type_probabilities = [0.5, 0.2, 0.2, 0.1]  # Standard is most common
        self.bidder_types = local_rng.choice(
            bidder_type_options,
            size=self.N,
            p=bidder_type_probabilities
        )
        
        # Calculate lognormal distribution base parameters
        mu = np.log(self.valuation_mean**2 / 
                   np.sqrt(self.valuation_std_dev**2 + self.valuation_mean**2))
        sigma = np.sqrt(np.log(1 + (self.valuation_std_dev**2 / 
                                   self.valuation_mean**2)))
        
        # Initialize valuations matrix
        self.valuations = np.zeros((self.N, self.N))
        
        # Generate market values (common component) for all objects
        market_values = local_rng.lognormal(
            mean=mu, 
            sigma=sigma, 
            size=self.N
        )
        
        # Generate bidder-specific valuations with type-specific components
        for i in range(self.N):
            bidder_type = self.bidder_types[i]
            type_params = self.type_parameters.get(bidder_type)
            
            # Calculate type-specific distribution parameters
            bidder_mean = self.valuation_mean * (1 + type_params["mean_factor"])
            bidder_std_dev = self.valuation_std_dev * (1 + type_params["std_dev_factor"])

            # Calculate lognormal parameters from these adjusted values
            bidder_mu = np.log(bidder_mean**2 / np.sqrt(bidder_std_dev**2 + bidder_mean**2))
            bidder_sigma = np.sqrt(np.log(1 + (bidder_std_dev**2 / bidder_mean**2)))
            
            # Generate private value component for this bidder
            private_component = local_rng.lognormal(
                mean=bidder_mu,
                sigma=bidder_sigma,
                size=self.N
            )
            
            # Combine market and private components with weights and convert to integers
            self.valuations[i] = np.round(
                self.market_value_weight * market_values + 
                self.private_value_weight * private_component
            ).astype(int)
        
        # Apply sparsity to valuations
        mask = local_rng.random((self.N, self.N)) > (1 - self.sparsity)
        self.valuations[mask] = -np.inf
        
        # Generate fixed noise matrix for all periods
        self.noise = local_rng.normal(0, self.sigma_v, (self.N, self.N))
        
    def generate_outcomes(self, W: np.ndarray) -> np.ndarray:
        """Generate outcomes through auction simulation."""
        return self._run_auction_simulation(W)
        
    def _run_auction_simulation(self, W: np.ndarray) -> np.ndarray:
        """Run auction simulation to generate panel data."""
        weights_history = []
        current_weights = np.zeros(self.N)
        
        # Initialize prices
        current_prices = np.zeros(self.N)
        
        # Initialize matching arrays
        matches = -np.ones(self.N, dtype=int)
        buyers_matched = set()
        
        for t in range(self.T):
            treatment = W[:, t]
            
            # Update observed values with treatment effects
            observed_values = ((1 + self.tau * treatment.reshape(-1, 1)) * 
                             self.valuations + self.noise)
            
            # Run multiple auction rounds per time period
            for _ in range(self.auction_rounds):
                # Initialize bids matrix
                bids = np.full((self.N, self.N), -np.inf)

                # Bidding phase
                for j in range(self.N):
                    values_minus_price = observed_values[:, j] - current_prices
                            
                    # Find top 2 values
                    top2_indices = np.argpartition(values_minus_price, -2)[-2:]
                    top2_indices = top2_indices[
                        np.argsort(values_minus_price[top2_indices])
                    ][::-1]
                    i_opt, i_alt = top2_indices

                    # Calculate bid if valuation is valid
                    if not np.isinf(observed_values[i_opt, j]):
                        bid = (current_prices[i_alt] + 
                            observed_values[i_opt, j] - 
                            observed_values[i_alt, j] + 
                            self.min_bid)
                        if bid > 0:
                            bids[i_opt, j] = bid

                # Matching phase
                for i in range(self.N):
                    j_opt = np.argmax(bids[i, :])
                    
                    if bids[i, j_opt] > current_prices[i]:                        
                        # Update matches when object receives competitive bid
                        if matches[i] != -1:
                            buyers_matched.discard(matches[i])

                        current_prices[i] = bids[i, j_opt]
                        matches[np.where(matches == j_opt)] = -1
                        matches[i] = j_opt
                        buyers_matched.add(j_opt)

                        current_weights[i] = observed_values[i, j_opt]
                        
                    if matches[i] == -1:
                        current_weights[i] = 0

            # Record history for this time period
            weights_history.append(current_weights.copy())
                    
        return np.array(weights_history).T