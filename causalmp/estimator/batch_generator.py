import numpy as np
from typing import Optional, Dict

class BatchGenerator:
    """Handle batch generation and normalization for counterfactual estimation.
    
    This class manages the creation of batch assignments for estimation,
    supporting both random sampling and partitioning approaches.
    """
    
    @staticmethod
    def generate(
        N: int,
        n_batch: int,
        batch_size: Optional[int] = None,
        partitioning: bool = False,
        seed: Optional[int] = 42
    ) -> np.ndarray:
        """Generate normalized batch matrices using sampling or partitioning.
        
        Parameters
        ----------
        N : int
            Number of units
        n_batch  : int
            Number of batches to generate
        batch_size : int, optional
            Target size for each batch when using random sampling
        partitioning : bool, default=False
            If True, creates equal-sized batches by partitioning units
            If False, uses systematic resampling with specified batch_size
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        numpy.ndarray, shape (n_batch, N)
            Normalized batch assignment matrix where each row sums to 1
            
        Raises
        ------
        ValueError
            If input parameters are invalid or incompatible
        """
        # Input validation
        BatchGenerator.validate_batch_parameters(N, n_batch, batch_size)
        if partitioning and n_batch  >= N:
            raise ValueError(f"When partitioning=True, n_batch  ({n_batch }) must be less than N ({N})")
        
        # Set random seed
        rng = np.random.default_rng(seed)
        
        # Initialize batch assignment matrix
        B = np.zeros((n_batch, N), dtype=int)
        
        if partitioning:
            B = BatchGenerator._create_partition_batches(N, n_batch, rng)
        else:
            B = BatchGenerator._create_random_batches(N, n_batch, batch_size, rng)
        
        # Normalize batch assignments
        batch_weights_matrix= BatchGenerator._normalize_batch_matrix(B)
        
        return batch_weights_matrix
    
    @staticmethod
    def _create_partition_batches(
        N: int,
        n_batch: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Create batches by partitioning units equally."""
        B = np.zeros((n_batch, N), dtype=int)
        batch_size = N // n_batch 
        
        # Assign units to batches sequentially
        for i in range(n_batch ):
            start_idx = i * batch_size
            # For last batch, include any remaining units
            end_idx = start_idx + batch_size if i < n_batch  - 1 else N
            B[i, start_idx:end_idx] = 1
            
        return B
    
    @staticmethod
    def _create_random_batches(
        N: int,
        n_batch: int,
        batch_size: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Create batches using random sampling strategy."""
        B = np.zeros((n_batch, N), dtype=int)
        
        # Create evenly spaced starting positions
        block1_start_list = np.linspace(0, N - batch_size, n_batch, dtype=int)
        
        # Generate batches
        for i in range(n_batch):
            # Get systematic block start position
            block1_start = block1_start_list[i]
            
            # Get random block start position
            block2_start = rng.integers(0, N - batch_size + 1)
            
            # Generate block indices
            block1_indices = np.arange(block1_start, block1_start + batch_size)
            block2_indices = np.arange(block2_start, block2_start + batch_size)
            
            # Combine and shuffle indices
            combined_indices = np.unique(np.concatenate([block1_indices, block2_indices]))
            rng.shuffle(combined_indices)
            
            # Calculate number of units to select
            selection_prob = batch_size / len(combined_indices)
            num_selected = rng.binomial(len(combined_indices), selection_prob)
            # Ensure at least one unit is selected
            num_selected = max(1, num_selected)
            
            # Select units randomly
            selected_indices = rng.choice(
                combined_indices,
                size=num_selected,
                replace=False
            )
            
            # Assign selected units to batch
            B[i, selected_indices] = 1
            
        return B
    
    @staticmethod
    def _normalize_batch_matrix(B: np.ndarray) -> np.ndarray:
        """Normalize batch matrix to ensure each row sums to 1."""
        B_row_sums = np.maximum(B.sum(axis=1, keepdims=True), 1)
        return B / B_row_sums
    
    @staticmethod
    def validate_batch_parameters(
        N: int,
        n_batch: int,
        batch_size: Optional[int] = None
    ) -> None:
        """Validate batch generation parameters.
        
        Parameters
        ----------
        N : int
            Number of units
        n_batch  : int
            Number of batches
        batch_size : int, optional
            Size of each batch
            
        Raises
        ------
        ValueError
            If parameters are invalid
        """
        if N <= 0:
            raise ValueError("N must be positive")
        if n_batch  <= 0:
            raise ValueError("n_batch  must be positive")
        if batch_size is not None:
            if batch_size <= 0:
                raise ValueError("batch_size must be positive")
            if batch_size > N:
                raise ValueError("batch_size cannot be larger than N")
    
    @staticmethod
    def get_batch_statistics(batch_weights_matrix: np.ndarray) -> Dict:
        """Calculate statistics about the generated batches.
        
        Parameters
        ----------
        batch_weights_matrix: numpy.ndarray
            Normalized batch assignment matrix
            
        Returns
        -------
        dict
            Dictionary containing:
            - average_batch_size : float
            - min_batch_size : float
            - max_batch_size : float
            - total_units_used : int
        """
        batch_sizes = batch_weights_matrix.sum(axis=1)
        return {
            'average_batch_size': np.mean(batch_sizes),
            'min_batch_size': np.min(batch_sizes),
            'max_batch_size': np.max(batch_sizes),
            'total_units_used': np.sum(batch_weights_matrix> 0)
        }