import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
from IPython.display import Math, display

class ResultVisualizer:
    """Handle visualization of experiment results."""
    
    def __init__(self):
        """Initialize visualizer with default styling."""
        # Set default style
        sns.set_theme(style="whitegrid")
        
        # Set font sizes
        self.title_fs = 26
        self.label_fs = 22
        self.tick_fs = 22
        self.legend_fontsize = 22
        
        # Define consistent color palette for methods
        self.color_palette = {
            'GT': '#4C72B0',     # Ground Truth in both plots (deep blue)
            'Ground Truth': '#4C72B0',
            'CMP': '#DD8452',      # Causal-MP in both plots (deep orange)
            'Causal-MP': '#DD8452',
            'bCMP': '#55A868',    # Consistent colors for other methods (deep green)
            'DM': '#8172B3',      # deep purple
            'HT': '#C44E52'       # deep red
        }
    
    def display_best_configuration(
        self,
        best_config: dict,
        best_model_terms: list
    ) -> None:
        """
        Display the best configuration and model terms in a formatted way.
        
        Parameters
        ----------
        best_config : dict
            Dictionary containing the best configuration parameters
            Format: {'detrending_config': {...}, 'main_config': {...}}
        best_model_terms : list
            List containing model terms [detrending_terms, main_terms]
            Each element is a list of LaTeX-formatted terms
        """
        # Check if detrending was used
        has_detrending = bool(best_config.get('detrending_config'))
        
        print("\n" + "="*80)
        print("{:^80}".format("Best Configuration and Model Terms"))
        print("="*80 + "\n")
        
        if has_detrending:
            # Display detrending configuration
            print("-"*40)
            print("DETRENDING CONFIGURATION")
            print("-"*40)
            detrend_config = best_config['detrending_config']
            for key, value in detrend_config.items():
                print(f"  {key.replace('_', ' ').title():<30} : {value}")
                
            # Display detrending model
            print("\nDetrending Model:")
            if best_model_terms[0]:  # If detrending terms exist
                equation = best_model_terms[0][0]
                terms = best_model_terms[0][1:]
                if terms:
                    equation += " " + " + ".join(terms)
                display(Math(equation))
            
            print("\n" + "-"*80 + "\n")
        
        # Display main configuration
        print("-"*40)
        print("MAIN CONFIGURATION")
        print("-"*40)
        main_config = best_config['main_config']
        for key, value in main_config.items():
            print(f"  {key.replace('_', ' ').title():<30} : {value}")
            
        # Display main model
        print("\nMain Model:")
        if best_model_terms[-1]:  # Main terms always exist
            equation = best_model_terms[-1][0]
            terms = best_model_terms[-1][1:]
            if terms:
                equation += " " + " + ".join(terms)
            display(Math(equation))
        
        print("\n" + "="*80)
    
    def plot_results(
        self,
        Observed_outcomes: pd.DataFrame,
        CFEs: pd.DataFrame,
        TTEs: pd.DataFrame,
        filename: Optional[str] = None,
        layout: str = "1x4",
        y_lim: Optional[Tuple[float, float]] = None,
        x_ticks: Optional[List[int]] = None,
        n_periods_for_tte: Optional[int] = None
    ) -> None:
        """
        Plot experiment results including outcomes, TTEs, and CFEs.
        
        Parameters
        ----------
        Observed_outcomes : pd.DataFrame
            DataFrame with observed outcomes
        CFEs : pd.DataFrame
            DataFrame with counterfactual evolutions
        TTEs : pd.DataFrame
            DataFrame with treatment effects
        filename : str, optional
            If provided, saves plot to this file
        layout : str, default="1x4"
            Layout of plots ("1x4" or "2x2")
        y_lim : tuple, optional
            Y-axis limits for TTE plot
        x_ticks : list, optional
            Custom x-axis tick locations
        n_periods_for_tte : int, optional
            Number of periods to use for TTE calculation. If not provided, defaults to T/2
        """
        # Calculate statistics
        mean_data = Observed_outcomes[Observed_outcomes['label'] == 'mean']
        variance_data = Observed_outcomes[Observed_outcomes['label'] == 'stdev']
        
        # Create figure with appropriate layout
        if layout == "1x4":
            fig = plt.figure(figsize=(24, 6))
            gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1])
            gs_left = gs[0].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.5)
            ax1_mean = fig.add_subplot(gs_left[0])
            ax1_var = fig.add_subplot(gs_left[1])
            ax2 = fig.add_subplot(gs[1])
            ax3 = fig.add_subplot(gs[2])
            ax4 = fig.add_subplot(gs[3])
        else:  # 2x2 layout
            fig = plt.figure(figsize=(15, 12))
            gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.2)
            gs_top_left = gs[0, 0].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.4)
            ax1_mean = fig.add_subplot(gs_top_left[0])
            ax1_var = fig.add_subplot(gs_top_left[1])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])
        
        # Plot means
        self._plot_mean_outcomes(ax1_mean, mean_data)
        
        # Plot variance
        self._plot_variance(ax1_var, variance_data)
        
        # Plot TTEs
        self._plot_ttes(ax2, TTEs, y_lim, n_periods_for_tte)
        
        # Plot CFEs
        self._plot_cfes(ax3, ax4, CFEs)
        
        # Set x-ticks if provided
        if x_ticks is not None:
            for ax in [ax1_mean, ax1_var, ax3, ax4]:
                ax.set_xticks(x_ticks)
        
        # Adjust layout and save/show
        plt.tight_layout()
        if filename:
            # Extract format from filename or default to png
            format = filename.split('.')[-1].lower() if '.' in filename else 'png'
            # If filename doesn't include extension, append the default
            if '.' not in filename:
                filename = f"{filename}.{format}"
            plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.show()
    
    def _plot_mean_outcomes(self, ax, data):
        """Plot mean outcomes."""
        sns.lineplot(data=data, x='Time', y='outcome',
                    errorbar=('pi', 95), color='navy', ax=ax)
        ax.set_title('Mean of Observed Outcomes', fontsize=self.title_fs, pad=10)
        ax.set_xlabel('', fontsize=self.label_fs)
        ax.set_ylabel('')
        ax.tick_params(axis='both', which='major', labelsize=self.tick_fs)
    
    def _plot_variance(self, ax, data):
        """Plot variance outcomes."""
        sns.lineplot(data=data, x='Time', y='outcome',
                    errorbar=('pi', 95), color='darkred', ax=ax)
        ax.set_title('St. Dev. of Observed Outcomes', fontsize=self.title_fs, pad=10)
        ax.set_xlabel('Time', fontsize=self.label_fs)
        ax.set_ylabel('')
        ax.tick_params(axis='both', which='major', labelsize=self.tick_fs)
    
    def _plot_ttes(self, ax, TTEs, y_lim, n_periods_for_tte):
        """Plot treatment effects."""
        # Calculate n_periods_for_tte if not provided
        max_time = TTEs['Time'].max()
        if n_periods_for_tte is None:
            n_periods_for_tte = int(max_time / 2)
        
        # Get data for last time points
        last_times = sorted(TTEs['Time'].unique())[-n_periods_for_tte:]
        plot_data = TTEs[TTEs['Time'].isin(last_times)].copy()
        avg_TTE = plot_data.groupby(['label', 'run'])['TTE'].mean().reset_index()
        
        # Create box plot with consistent colors
        box_order = ['GT', 'CMP', 'bCMP', 'DM', 'HT']
        box_palette = {method: self.color_palette[method] for method in box_order}
        
        # Plot boxplots
        sns.boxplot(data=avg_TTE, y='TTE', x='label', hue='label',
                   order=box_order,
                   palette=box_palette,
                   showfliers=False, ax=ax, legend=False)
        
        # Calculate and plot means
        means = avg_TTE.groupby('label')['TTE'].mean()
        for i, method in enumerate(box_order):
            if method in means.index:
                ax.plot(i, means[method], 'x', color='black', markersize=8, markeredgewidth=1.5)
        
        ax.set_title('Total Treatment Effect', fontsize=self.title_fs, pad=10)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.tick_params(axis='both', which='major', labelsize=self.tick_fs)
        
        if y_lim is not None:
            ax.set_ylim(y_lim[0], y_lim[1])
    
    def _plot_cfes(self, ax3, ax4, CFEs):
        """Plot counterfactual evolutions."""
        # Map CFE labels to consistent colors
        cfe_labels = CFEs['label'].unique()
        cfe_palette = {label: self.color_palette.get(label, None) for label in cfe_labels}
        
        # Plot CFE(0)
        cfe0_data = CFEs[CFEs['type'] == 'CFE(0)']
        sns.lineplot(data=cfe0_data, x='Time', y='CFE',
                    hue='label', style='label',
                    palette=cfe_palette,
                    errorbar=('pi', 95), ax=ax3, legend=False)
        ax3.set_title('CFE of All Control', fontsize=self.title_fs, pad=10)
        ax3.set_xlabel('Time', fontsize=self.label_fs)
        ax3.set_ylabel('')
        ax3.tick_params(axis='both', which='major', labelsize=self.tick_fs)
        
        # Plot CFE(1)
        cfe1_data = CFEs[CFEs['type'] == 'CFE(1)']
        sns.lineplot(data=cfe1_data, x='Time', y='CFE',
                    hue='label', style='label',
                    palette=cfe_palette,
                    errorbar=('pi', 95), ax=ax4)
        ax4.set_title('CFE of All Treatment', fontsize=self.title_fs, pad=10)
        ax4.set_xlabel('Time', fontsize=self.label_fs)
        ax4.set_ylabel('')
        ax4.tick_params(axis='both', which='major', labelsize=self.tick_fs)
        ax4.legend(title='', fontsize=self.legend_fontsize, loc='lower right')
        
        # Synchronize y-axis limits
        y_min = min(ax3.get_ylim()[0], ax4.get_ylim()[0])
        y_max = max(ax3.get_ylim()[1], ax4.get_ylim()[1])
        ax3.set_ylim(y_min, y_max)
        ax4.set_ylim(y_min, y_max)