import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import traceback
import time
import logging

from ..estimator import CrossValidator
from ..estimator import CFEEstimator
from ..estimator import dinm_estimate, ht_estimate, basic_cmp_estimate
from ..simulator import generate_data_from_design
from .visualizer import ResultVisualizer

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Base class for running experiments."""
    
    def __init__(
        self,
        environment: Dict,
        grouped_configurations: Dict,
        data_path: Optional[Union[str, Path]] = None,
        time_blocks_list: Optional[List[Tuple[int, int]]] = None
    ):
        self.environment = environment
        self.grouped_configurations = grouped_configurations
        self.data_path = Path(data_path) if data_path else None
        self.time_blocks_list = time_blocks_list
        self.cross_validator = CrossValidator()
        
    def _validate_inputs(self, Y, W, desired_W_1=None, desired_W_2=None):
        """Validate input data dimensions and values."""
        if not isinstance(Y, np.ndarray) or not isinstance(W, np.ndarray):
            raise ValueError("Y and W must be numpy arrays")
            
        if Y.shape != W.shape:
            raise ValueError(f"Shape mismatch: Y {Y.shape}, W {W.shape}")
            
        if np.any(np.isnan(Y)) or np.any(np.isnan(W)):
            raise ValueError("Missing values detected in Y or W")
            
        if not np.all(np.isin(W, [0, 1])):
            raise ValueError("W must contain only binary values")
            
        if desired_W_1 is not None:
            if desired_W_1.shape != W.shape:
                raise ValueError(f"Shape mismatch: desired_W_1 {desired_W_1.shape}")
            if not np.all(np.isin(desired_W_1, [0, 1])):
                raise ValueError("desired_W_1 must contain only binary values")
            
        if desired_W_2 is not None:
            if desired_W_2.shape != W.shape:
                raise ValueError(f"Shape mismatch: desired_W_2 {desired_W_2.shape}")
            if not np.all(np.isin(desired_W_2, [0, 1])):
                raise ValueError("desired_W_2 must contain only binary values")
        
        # Calculate maximum possible lag from configurations
        max_n_lags_Y = 0
        max_n_lags_W = 0
        
        for config_group in self.grouped_configurations.values():
            for config in config_group:
                main_config = config['main_config']
                max_n_lags_Y = max(max_n_lags_Y, main_config['n_lags_Y'])
                max_n_lags_W = max(max_n_lags_W, main_config['n_lags_W'])
                
                # Check detrending config if present
                if config['detrending_config'] and config['detrending_config'].get('detrending', False):
                    max_n_lags_Y = max(max_n_lags_Y, config['detrending_config'].get('n_lags_Y', 0))
        
        # Calculate max_lag
        max_lag = max(max_n_lags_Y, max_n_lags_W - 1)
        
        if max_lag > 0:
            # Check if initial treatment patterns match for desired_W_1
            if desired_W_1 is not None and not np.array_equal(W[:, :max_lag], desired_W_1[:, :max_lag]):
                raise ValueError(
                    f"Initial {max_lag} columns of W and desired_W_1 must match for proper counterfactual estimation. "
                    "These columns represent pre-treatment conditions to initialize the recursive estimation."
                )
            
            # Check if initial treatment patterns match for desired_W_2
            if desired_W_2 is not None and not np.array_equal(W[:, :max_lag], desired_W_2[:, :max_lag]):
                raise ValueError(
                    f"Initial {max_lag} columns of W and desired_W_2 must match for proper counterfactual estimation. "
                    "These columns represent pre-treatment conditions to initialize the recursive estimation."
                )

class SingleExperimentRunner(ExperimentRunner):
    """Handle single experiment execution."""
    
    def run(
        self,
        run_id: Optional[int] = 0,
        n_validation_batch: int = 1,
        visualize_ccv: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, Dict]:
        """
        Execute a single experimental run.
        
        Parameters
        ----------
        run_id : int, optional
            Identifier for the current run
        n_validation_batch: int, default=1
            Number of validation batches
        visualize_ccv : bool, default=False
            Whether to generate visualization plots of the best estimation in counterfactual-CV
            
        Returns
        -------
        tuple
            (Observed_outcomes, CFEs, TTEs, best_config, best_model_terms) as DataFrames and Dicts
        """        
        # Extract parameters
        N = self.environment["N"]
        T = self.environment["stage_time_blocks"][-1]
        
        # Initialize DataFrames
        Observed_outcomes = pd.DataFrame({
            'Time': pd.Series(dtype='int64'),
            'outcome': pd.Series(dtype='float64'),
            'run': pd.Series(dtype='int16'),
            'label': pd.Series(dtype='str')
        })
        
        CFEs = pd.DataFrame({
            'Time': pd.Series(dtype='int64'),
            'CFE': pd.Series(dtype='float64'),
            'run': pd.Series(dtype='int16'),
            'type': pd.Series(dtype='str'),
            'label': pd.Series(dtype='str')
        })
        
        TTEs = pd.DataFrame({
            'Time': pd.Series(dtype='int64'),
            'TTE': pd.Series(dtype='float64'),
            'run': pd.Series(dtype='int16'),
            'label': pd.Series(dtype='str')
        })
        
        try:
            # Generate or load data
            if self.data_path is None:
                W, Y, desired_W_1, desired_Y_1, desired_W_2, desired_Y_2 = \
                    generate_data_from_design(self.environment, staggered=True, seed=run_id)
            else:
                W, Y, desired_W_1, desired_Y_1, desired_W_2, desired_Y_2 = \
                    self._load_data(run_id, N, T)
            
            # Validate inputs
            self._validate_inputs(Y, W, desired_W_1, desired_W_2)
            
            # Find best configuration
            best_config, best_model_terms, best_score = self.cross_validator.validate(
                Y, W, self.grouped_configurations, self.time_blocks_list,
                n_validation_batch=n_validation_batch, visualize=visualize_ccv
            )
            logger.info(f"Cross-validation completed successfully and the best score is {best_score}")
            
            # Calculate estimations using best configuration
            results = self._calculate_estimations(
                Y, W, desired_Y_1, desired_Y_2,
                desired_W_1, desired_W_2, best_config, run_id
            )
            
            # Update DataFrames
            Observed_outcomes = pd.concat([Observed_outcomes, results['observed']], ignore_index=True)
            CFEs = pd.concat([CFEs, results['cfes']], ignore_index=True)
            TTEs = pd.concat([TTEs, results['ttes']], ignore_index=True)
            
        except Exception as e:
            print(f"\nError in run {run_id}:")
            print("="*50)
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nFull traceback:")
            print(traceback.format_exc())
            print("="*50 + "\n")
        
        return Observed_outcomes, CFEs, TTEs, best_config, best_model_terms

        
    def _load_data(self, run_id: int, N: int, T: int) -> Tuple[np.ndarray, ...]:
        """Load data from files."""
        file_paths = {
            'Y': self.data_path / f'seed{run_id}/experiment_n{N}_t{T}_seed{run_id}_panel_data.csv',
            'Y_1': self.data_path / f'seed{run_id}/treatment_n{N}_t{T}_seed{run_id}_panel_data.csv',
            'Y_0': self.data_path / f'seed{run_id}/control_n{N}_t{T}_seed{run_id}_panel_data.csv',
            'W': self.data_path / f'seed{run_id}/experiment_n{N}_t{T}_seed{run_id}_treatment_data.csv',
            'W_1': self.data_path / f'seed{run_id}/treatment_n{N}_t{T}_seed{run_id}_treatment_data.csv',
            'W_0': self.data_path / f'seed{run_id}/control_n{N}_t{T}_seed{run_id}_treatment_data.csv'
        }
        
        # Check for missing files
        missing = [f for f in file_paths.values() if not f.exists()]
        if missing:
            raise FileNotFoundError(f"Missing files: {missing}")
        
        # Read all files
        data = {k: pd.read_csv(v).iloc[:, 1:].to_numpy() for k, v in file_paths.items()}
        
        return (data['W'], data['Y'], data['W_0'], data['Y_0'], data['W_1'], data['Y_1'])
    
    def _calculate_estimations(
        self,
        Y: np.ndarray,
        W: np.ndarray,
        desired_Y_1: np.ndarray,
        desired_Y_2: np.ndarray,
        desired_W_1: np.ndarray,
        desired_W_2: np.ndarray,
        config: Dict,
        run_id: int
    ) -> Dict[str, pd.DataFrame]:
        """Calculate estimations using best configuration."""
        # Initialize CFE estimator with best config
        estimator = CFEEstimator(config)
        
        # Calculate CFEs
        Estimation_CFE_1 = estimator.estimate(Y, W, desired_W_1)
        Desired_CFE_1 = np.mean(desired_Y_1, axis=0)
        
        Estimation_CFE_2 = estimator.estimate(Y, W, desired_W_2)
        Desired_CFE_2 = np.mean(desired_Y_2, axis=0)
        
        # Calculate TTEs
        TTE_GT = Desired_CFE_2 - Desired_CFE_1
        TTE_DinM = dinm_estimate(Y, W)
        TTE_HT = ht_estimate(
            Y, W,
            self.environment["stage_time_blocks"],
            self.environment["design"]
        )
        TTE_CMP = Estimation_CFE_2 - Estimation_CFE_1
        TTE_BasicCMP = basic_cmp_estimate(Y, W)
        
        # Create results dictionary with DataFrames
        results = {
            'observed': self._create_observed_df(Y, run_id),
            'cfes': self._create_cfe_df(
                Desired_CFE_1, Estimation_CFE_1,
                Desired_CFE_2, Estimation_CFE_2,
                run_id
            ),
            'ttes': self._create_tte_df(
                TTE_GT, TTE_CMP, TTE_DinM, TTE_HT, TTE_BasicCMP,
                run_id
            )
        }
        
        return results
    
    def _create_observed_df(self, Y: np.ndarray, run_id: int) -> pd.DataFrame:
        """Create DataFrame for observed outcomes."""
        mean_df = pd.DataFrame({
            'Time': np.arange(Y.shape[1]),
            'outcome': np.mean(Y, axis=0),
            'run': run_id,
            'label': 'mean'
        })
        
        stdev_df = pd.DataFrame({
            'Time': np.arange(Y.shape[1]),
            'outcome': np.std(Y, axis=0),
            'run': run_id,
            'label': 'stdev'
        })
        
        return pd.concat([mean_df, stdev_df], ignore_index=True)
    
    def _create_cfe_df(
        self,
        CFE_1: np.ndarray,
        Est_CFE_1: np.ndarray,
        CFE_2: np.ndarray,
        Est_CFE_2: np.ndarray,
        run_id: int
    ) -> pd.DataFrame:
        """Create DataFrame for CFEs."""
        cfe_data = [
            {'CFE': CFE_1, 'type': 'CFE(0)', 'label': 'Ground Truth'},
            {'CFE': Est_CFE_1.ravel(), 'type': 'CFE(0)', 'label': 'Causal-MP'},
            {'CFE': CFE_2, 'type': 'CFE(1)', 'label': 'Ground Truth'},
            {'CFE': Est_CFE_2.ravel(), 'type': 'CFE(1)', 'label': 'Causal-MP'}
        ]
        
        dfs = []
        for data in cfe_data:
            df = pd.DataFrame({
                'Time': np.arange(len(data['CFE'])),
                'CFE': data['CFE'],
                'run': run_id,
                'type': data['type'],
                'label': data['label']
            })
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)
    
    def _create_tte_df(
        self,
        TTE_GT: np.ndarray,
        TTE_CMP: np.ndarray,
        TTE_DinM: np.ndarray,
        TTE_HT: np.ndarray,
        TTE_BasicCMP: np.ndarray,
        run_id: int
    ) -> pd.DataFrame:
        """Create DataFrame for TTEs."""
        tte_data = [
            {'TTE': TTE_GT.ravel(), 'label': 'GT'},
            {'TTE': TTE_CMP.ravel(), 'label': 'CMP'},
            {'TTE': TTE_DinM.ravel(), 'label': 'DM'},
            {'TTE': TTE_HT.ravel(), 'label': 'HT'},
            {'TTE': TTE_BasicCMP.ravel(), 'label': 'bCMP'}
        ]
        
        dfs = []
        for data in tte_data:
            df = pd.DataFrame({
                'Time': np.arange(len(data['TTE'])),
                'TTE': data['TTE'],
                'run': run_id,
                'label': data['label']
            })
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)
    
class MultiExperimentRunner(ExperimentRunner):
    """Handle multiple sequential experiment executions."""
    
    def run(
        self,
        n_runs: int,
        n_validation_batch: int = 1,
        visualize_ccv: bool = False,
        return_model_terms: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute multiple experimental runs sequentially.
        
        Parameters
        ----------
        n_runs : int
            Number of experiment runs to perform
        n_validation_batch: int, default=1
            Number of validation batches
        visualize_ccv : bool, default=False
            Whether to generate visualization plots of the best estimation in counterfactual-CV
        model_terms : bool, default=False
            Whether to display configuration and model terms
            
        Returns
        -------
        tuple
            (Observed_outcomes, CFEs, TTEs) as pandas DataFrames with results from all runs
        """
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
        
        # Set up single runner
        single_runner = SingleExperimentRunner(
            self.environment,
            self.grouped_configurations,
            self.data_path,
            self.time_blocks_list
        )
        
        # Track timing metrics
        start_time = time.time()
        completed_runs = 0
        
        # Perform runs sequentially
        for run_id in range(n_runs):
            try:
                print(f"Starting run {run_id+1}/{n_runs}")
                
                # Execute single run
                Observed_outcomes, CFEs, TTEs, best_config, best_model_terms = single_runner.run(
                    run_id=run_id,
                    n_validation_batch=n_validation_batch,
                    visualize_ccv=visualize_ccv
                )

                # Display best configuration
                if return_model_terms:
                    result_visualizer = ResultVisualizer()
                    result_visualizer.display_best_configuration(best_config, best_model_terms)
                
                # Concatenate results
                all_Observed_outcomes = pd.concat([all_Observed_outcomes, Observed_outcomes], ignore_index=True)
                all_CFEs = pd.concat([all_CFEs, CFEs], ignore_index=True)
                all_TTEs = pd.concat([all_TTEs, TTEs], ignore_index=True)
                
                completed_runs += 1
                print(f"Completed run {run_id+1}/{n_runs}")
                
            except Exception as e:
                print(f"\nError in run {run_id}:")
                print("="*50)
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print("\nFull traceback:")
                print(traceback.format_exc())
                print("="*50 + "\n")
        
        # Calculate and display timing statistics
        total_time = time.time() - start_time
        if completed_runs > 0:
            average_time = total_time / completed_runs
            print(f'Average time per run: {average_time:.2f} seconds')
            print(f'Total time: {total_time:.2f} seconds')
            print(f'Successfully completed {completed_runs} out of {n_runs} runs')
        
        return all_Observed_outcomes, all_CFEs, all_TTEs