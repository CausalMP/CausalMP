import multiprocessing as mp
from functools import partial
import pandas as pd
import time
import traceback
from typing import Dict, Optional, Union, Tuple
from pathlib import Path

from .experiment_runners import ExperimentRunner, SingleExperimentRunner
from .visualizer import ResultVisualizer

class ParallelExperimentRunner(ExperimentRunner):
    """Handle parallel execution of experiments."""
    
    def __init__(
        self,
        environment: Dict,
        grouped_configurations: Dict,
        data_path: Optional[Union[str, Path]] = None,
        time_blocks_list: Optional[list] = None,
        n_processes: Optional[int] = None
    ):
        """
        Initialize parallel experiment runner.
        
        Parameters
        ----------
        environment : dict
            Environment configuration
        grouped_configurations : dict
            Configuration options for estimation
        data_path : str or Path, optional
            Base path for data files
        time_blocks_list : list, optional
            List of time blocks for validation
        n_processes : int, optional
            Number of parallel processes to use
        """
        super().__init__(environment, grouped_configurations, data_path, time_blocks_list)
        self.n_processes = n_processes
        
    def run(
        self,
        n_runs: int = 1,
        n_validation_batch: int = 1,
        visualize_ccv: bool = False,
        return_model_terms: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run multiple experimental trials in parallel.
        
        Parameters
        ----------
        n_runs : int, default=1
            Number of experimental runs
        n_validation_batch: int, default=1
            Number of validation batches
        visualize_ccv : bool, default=False
            Whether to to generate visualization plots of the best estimation in counterfactual-CV 
        return_model_terms : bool, default=False
            Whether to display configuration and model terms
            
        Returns
        -------
        tuple
            (all_Observed_outcomes, all_CFEs, all_TTEs) as DataFrames
        """
        if visualize_ccv:
            print("\nWarning: visualize_ccv=True is not supported in parallel processing.")
            print("Visualizations will be disabled. Use sequential runner with n_processes=1 for visualizations.\n")
    
        total_time = time.time()
        
        # Initialize combined DataFrames
        all_Observed_outcomes = pd.DataFrame({
            'Time': pd.Series(dtype='int64'),
            'outcome': pd.Series(dtype='float64'),
            'run': pd.Series(dtype='int16'),
            'label': pd.Series(dtype='str')
        })
        all_CFEs = pd.DataFrame({
            'Time': pd.Series(dtype='int64'),
            'CFE': pd.Series(dtype='float64'),
            'run': pd.Series(dtype='int16'),
            'type': pd.Series(dtype='str'),
            'label': pd.Series(dtype='str')
        })
        all_TTEs = pd.DataFrame({
            'Time': pd.Series(dtype='int64'),
            'TTE': pd.Series(dtype='float64'),
            'run': pd.Series(dtype='int16'),
            'label': pd.Series(dtype='str')
        })
        
        # Add collection for best configs and model terms
        best_configs_dict = {}
        best_model_terms_dict = {}
        
        # Set up single runner for parallel execution
        single_runner = SingleExperimentRunner(
            self.environment,
            self.grouped_configurations,
            self.data_path,
            self.time_blocks_list
        )
        
        # Create partial function with fixed parameters
        partial_run = partial(
            single_runner.run,
            n_validation_batch=n_validation_batch,
            visualize_ccv=False,  # Always False for parallel processing
        )
        
        # Run parallel processing
        with mp.Pool(processes=self.n_processes) as pool:
            try:
                # Process results as they complete
                for run_id, run_results in enumerate(pool.imap(partial_run, range(n_runs))):
                    Observed_outcomes, CFEs, TTEs, best_config, best_model_terms = run_results
                    
                    # Store best config and model terms for this run
                    best_configs_dict[run_id] = best_config
                    best_model_terms_dict[run_id] = best_model_terms
                    
                    # Concatenate results
                    all_Observed_outcomes = pd.concat(
                        [all_Observed_outcomes, Observed_outcomes],
                        ignore_index=True
                    )
                    all_CFEs = pd.concat([all_CFEs, CFEs], ignore_index=True)
                    all_TTEs = pd.concat([all_TTEs, TTEs], ignore_index=True)
                    
            except Exception as e:
                print("\nError in parallel processing:")
                print("="*50)
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print("\nFull traceback:")
                print(traceback.format_exc())
                print("="*50 + "\n")
            
        # Calculate and display timing statistics
        total_time = time.time() - total_time
        if n_runs > 0:
            average_time = total_time / n_runs
            print(f'Average time per run: {average_time:.2f} seconds')
            print(f'Total time: {total_time:.2f} seconds')
            print(f'Successfully completed {len(all_Observed_outcomes["run"].unique())} out of {n_runs} runs')
        
        # Print best configurations and model terms if requested
        if return_model_terms:
            result_visualizer = ResultVisualizer()
            print("\nBest Configurations and Model Terms by Run:")
            print("="*50)
            for run_id in sorted(best_configs_dict.keys()):
                print(f"\nRun {run_id + 1}:")
                result_visualizer.display_best_configuration(
                    best_configs_dict[run_id],
                    best_model_terms_dict[run_id]
                )
        
        return all_Observed_outcomes, all_CFEs, all_TTEs