import numpy as np
from typing import Dict, Optional, Any
from .base_environment import BaseEnvironment

class DataCenterEnvironment(BaseEnvironment):
    """Environment for simulating a server farm with dynamic task allocation.
    
    This environment models a data center where servers process incoming tasks using 
    a join-the-shortest-queue routing policy. The system demonstrates interference 
    through shared resource allocation, even without explicit network connections.
    
    Attributes:
        ENVIRONMENT_TYPE (str): Identifier for the environment type
        base_arrival_rate (float): Base rate for task arrivals
        base_service_rate (float): Base rate for task processing
        tau (float): Treatment effect size (processing speed improvement)
        warm_up_T (int): Number of warm-up periods before measurement
        JSQ_sample_size (int): Sample size for join-shortest-queue policy
        n_job_type_total (int): Total number of different task types
        n_job_type_per_server (int): Maximum task types per server
        arrival_rates (np.ndarray): Time-varying arrival rates
        server_capabilities (List[Set[int]]): Task types each server can process
        compatible_servers (List[List[int]]): Servers compatible with each task type
    """
    
    ENVIRONMENT_TYPE = "Data Center Model"
    
    def __init__(self, params: Dict[str, Any], seed: Optional[int] = None):
        """Initialize the data center environment."""
        super().__init__(params, seed)
        
        # Extract parameters specific to data center environment
        self.base_arrival_rate = params.get("base_arrival_rate", 0.75)
        self.base_service_rate = params.get("base_service_rate", 1)
        self.tau = params.get("tau", 0.2)
        self.warm_up_T = params.get("warm_up_period", 24)
        self.JSQ_sample_size = params.get("JSQ_sample_size", 5)
        self.n_job_type_total = params.get("n_job_type_total", 10)
        self.n_job_type_per_server = params.get("n_job_type_per_server", 2)
        
        # Initialize attributes that will be set in initialize_environment
        self.arrival_rates = None
        self.server_capabilities = None
        self.compatible_servers = None
        
    def initialize_environment(self) -> None:
        """Initialize data center environment with server capabilities and arrival rates."""
        # Generate arrival rates for simulation period plus warm-up
        self.arrival_rates = self._generate_arrival_rates(
            self.T + self.warm_up_T + 1,
            self.base_arrival_rate
        )
        
        # Initialize server capabilities with fixed seed for consistency
        capability_rng = np.random.RandomState(42)
        self.server_capabilities = []
        
        for _ in range(self.N):
            n_tasks = max(1, int(capability_rng.poisson(self.n_job_type_per_server)))
            capabilities = capability_rng.choice(
                self.n_job_type_total,
                size=min(n_tasks, self.n_job_type_total),
                replace=False
            )
            self.server_capabilities.append(set(capabilities))
            
        # Precompute compatible servers for each task type
        self.compatible_servers = [
            [i for i, capabilities in enumerate(self.server_capabilities)
             if task_type in capabilities]
            for task_type in range(self.n_job_type_total)
        ]
        
    def generate_outcomes(self, W: np.ndarray) -> np.ndarray:
        """Generate outcomes through server farm simulation."""
        return self._simulate_server_farm(W)
    
    def _generate_arrival_rates(self, T_length: int, base_rate: float) -> np.ndarray:
        """Generate time-varying arrival rates with temporal patterns."""
        # Initialize arrival rate generation with fixed seed for consistency
        arrival_rate_rng = np.random.RandomState(42)
            
        # Initialize array
        lambda_t = np.zeros(T_length)
        
        # Daily pattern (24 hours)
        hours = np.arange(T_length) % 24
        
        # Define daily patterns
        daily_pattern = (
            # Morning ramp-up (6 AM - 9 AM)
            (1 + 0.2 * np.sin(np.pi * (hours - 6) / 6)) * (hours >= 6) * (hours < 9) +
            # Peak work hours (9 AM - 5 PM)
            1.15 * (hours >= 9) * (hours < 17) +
            # Evening decline (5 PM - 11 PM)
            (1 + 0.15 * np.cos(np.pi * (hours - 17) / 8)) * (hours >= 17) * (hours < 23) +
            # Night time (11 PM - 6 AM)
            0.6 * (hours >= 23) + 0.6 * (hours < 6)
        )
        
        # Combine patterns
        lambda_t = base_rate * daily_pattern
        
        # Add random fluctuations
        random_noise = 0.05 * arrival_rate_rng.randn(T_length)
        lambda_t = lambda_t * (1 + random_noise)
        
        # Ensure minimum rate
        lambda_t = np.maximum(lambda_t, 0.3 * base_rate)
        
        return lambda_t
    
    def _simulate_server_farm(self, W: np.ndarray) -> np.ndarray:
        """Run server farm simulation to generate utilization data."""
        # Create local RNG with fixed seed for reproducibility
        local_rng = np.random.default_rng(self.seed)

        # Initialize simulation state
        queues = np.zeros((self.N, 1))
        Y = np.zeros((self.N, self.T))
        service_rates_default = self.base_service_rate * np.ones((self.N,))
        server_status = np.zeros((self.N,))
        service_rates = service_rates_default.copy()

        # Time tracking
        current_time = -self.warm_up_T
        current_period = -self.warm_up_T
        current_period_Y = np.zeros(self.N)
        
        # Main simulation loop
        while current_time < self.T:
            # Generate next event
            hour_index = int(np.floor(current_time + self.warm_up_T))
            arrival_rate = self.arrival_rates[hour_index]
            # Calculate time until next event
            time_to_next = local_rng.exponential(
                1 / (self.N * arrival_rate + np.sum(server_status * service_rates))
            )
            next_event_time = current_time + time_to_next
            
            # Check period boundary crossing
            period = int(np.floor(current_time))
            next_period = int(np.floor(next_event_time))
            
            if next_period > period and period >= 0 and period < self.T:
                # Record Y for completed period
                time_in_period = period + 1 - current_time
                current_period_Y += time_in_period * server_status
                Y[:, period] = current_period_Y
                current_period_Y = np.zeros(self.N)
                
                # Update service rates for next period
                if period + 1 < self.T:
                    treatment = W[:, period]
                    service_rates = service_rates_default * (1 + self.tau * treatment)
            
            # Process event
            if local_rng.random() <= (self.N * arrival_rate) / (self.N * arrival_rate + np.sum(server_status * service_rates)):
                # Process arrival
                task_type = local_rng.integers(self.n_job_type_total)
                compatible_servers = self.compatible_servers[task_type]
                
                if compatible_servers:
                    # Select server using JSQ policy
                    sample_size = min(self.JSQ_sample_size, len(compatible_servers))
                    sampled_servers = local_rng.choice(compatible_servers, size=sample_size, replace=False)
                    sampled_queues = queues[sampled_servers]
                    shortest_queues = np.where(sampled_queues == np.min(sampled_queues))[0]
                    selected_idx = local_rng.integers(len(shortest_queues))
                    selected_server = sampled_servers[selected_idx]
                    
                    # Update server state
                    server_status[selected_server] = 1
                    queues[selected_server] += 1
            else:
                # Process service completion
                active_servers = np.where(server_status > 0)[0]
                if len(active_servers) > 0:
                    # Select server proportional to service rate
                    probs = service_rates[active_servers] / np.sum(service_rates[active_servers])
                    selected_server = local_rng.choice(active_servers, p=probs)
                    
                    # Update server state
                    queues[selected_server] -= 1
                    if queues[selected_server] == 0:
                        server_status[selected_server] = 0
            
            # Update Y tracking
            if period >= 0 and period < self.T:
                time_elapsed = min(next_event_time - current_time, period + 1 - current_time)
                current_period_Y += time_elapsed * server_status
            
            # Advance time
            current_time = next_event_time
            
        return Y