CausalMP - Interference Gym: Counterfactual Estimation and Simulation Package
=======================================================================

.. image:: https://img.shields.io/badge/python-3.8%2B-blue.svg
   :target: https://python.org
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

**CausalMP** is a Python package for counterfactual estimation and simulation in causal inference scenarios with network interference. It provides tools for analyzing treatment effects in complex experimental settings.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   environments/index
   examples/index
   
Key Features
============

🎯 **Estimation Methods**
   - Counterfactual Evolution (CFE) estimation with cross-validation
   - Semi-recursive estimation for detrending observed outcomes
   - Classic estimators (Difference-in-Means, Horvitz-Thompson, Basic CMP)

🌐 **Simulation Environments**
   - Belief Adoption Model (social network dynamics with real Slovak city data)
   - NYC Taxi Routes (transportation networks with TLC Trip Record Data)
   - Exercise Encouragement Program (health interventions with social influence)
   - Auction Model (competitive market dynamics)
   - Data Center Model (distributed service systems with JSQ routing)

🔧 **Flexible Architecture**
   - Modular design with independent components (estimator, simulator, runner)
   - Customizable environment parameters
   - Configurable estimation parameters

Installation
============

Development Installation (Current)
----------------------------------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/CausalMP/CausalMP.git
   cd CausalMP
   pip install -e .

Component-specific installation:

.. code-block:: bash

   pip install -e .[estimator]  # Estimator only
   pip install -e .[simulator]  # Simulator only
   pip install -e .[all]        # All components
   pip install -e .[dev]        # Development setup

Quick Start
===========

Basic Counterfactual Estimation
--------------------------------

.. code-block:: python

   import numpy as np
   from causalmp import cmp_estimator

   # Your panel data
   Y = np.random.randn(100, 20)  # 100 units, 20 time periods
   
   # Create treatment assignments with three treatment stages
   W = np.zeros((100, 20))
   # Stage 1: periods 0-6, 20% treatment probability
   W[:, 0:7] = np.random.binomial(1, 0.2, (100, 7))
   # Stage 2: periods 7-13, 50% treatment probability  
   W[:, 7:14] = np.random.binomial(1, 0.5, (100, 7))
   # Stage 3: periods 14-19, 70% treatment probability
   W[:, 14:20] = np.random.binomial(1, 0.7, (100, 6))

   # Desired scenarios
   desired_W_1 = np.zeros((100, 20))  # Control scenario
   desired_W_2 = np.ones((100, 20))   # Treatment scenario

   # Make the first column of all treatment scenarios the same
   W[:, 0] = desired_W_1[:, 0]
   desired_W_2[:, 0] = desired_W_1[:, 0]

   # Configuration for estimation
   main_param_ranges = {
       'n_lags_Y_range': [1],
       'n_lags_W_range': [1],
       'moment_order_p_Y_range': [1],
       'moment_order_p_W_range': [1],
       'moment_order_u_Y_range': [1],
       'moment_order_u_W_range': [1],
       'interaction_term_p_range': [None],
       'interaction_term_u_range': [None, 1],
       'n_batch_range': [100, 500],
       'batch_size_range': [10, 20],
       'ridge_alpha_range': [1e-4, 1e-2]
   }

   # Estimate counterfactuals
   predictions_1, predictions_2, best_config, best_terms = cmp_estimator(
       Y=Y,
       W=W,
       desired_W=desired_W_1,
       desired_W_2=desired_W_2,
       main_param_ranges=main_param_ranges,
       time_blocks=[(0, 7), (7, 14), (14, 20)],
       n_validation_batch=2,
       return_model_terms=True
   )

Simulation Environment
----------------------

.. code-block:: python

   from causalmp import cmp_simulator

   # Environment configuration
   environment = {
       'N': 3366,  # Krupina network size
       'setting': 'Belief Adoption Model',
       'design': [0, 0.1, 0.2, 0.5],  # Treatment probabilities
       'stage_time_blocks': [1, 3, 5, 7],  # Time blocks
       'desired_design_1': [0, 0],  # Control scenario
       'desired_design_2': [0, 1],  # Full treatment scenario
       'desired_stage_time_blocks': [1, 7],  # Counterfactual time blocks
       'tau': 1  # Treatment effect coefficient
   }

   # Run simulation
   W, Y, W1, Y1, W2, Y2 = cmp_simulator(environment=environment, seed=42)

Available Environments
======================

The package includes simulation environments with corresponding Jupyter notebooks:

**Belief Adoption Model**
   - ``nb_belief_adoption__Krupina.ipynb`` (N=3,366)
   - ``nb_belief_adoption__Topolcany.ipynb`` (N=18,246) 
   - ``nb_belief_adoption__Zilina.ipynb`` (N=42,971)

**NYC Taxi Routes**
   - ``nb_nyc_taxi_routes.ipynb`` (N=18,768 routes)

**Exercise Encouragement Program**
   - ``nb_exercise_encouragement_program.ipynb``

**Auction Model**
   - ``nb_auction_model.ipynb``

**Data Center Model**
   - ``nb_data_center_model.ipynb``

API Reference
=============

Main Interface Functions
------------------------

.. autofunction:: causalmp.cmp_estimator

.. autofunction:: causalmp.cmp_simulator

.. autofunction:: causalmp.cmp_runner

Estimator Classes
-----------------

.. autoclass:: causalmp.CFEEstimator
   :members:

.. autoclass:: causalmp.CounterfactualEstimator
   :members:

.. autoclass:: causalmp.CFESemiRecursiveEstimator
   :members:

.. autoclass:: causalmp.CrossValidator
   :members:

Classic Estimators
------------------

.. autofunction:: causalmp.dinm_estimate

.. autofunction:: causalmp.ht_estimate

.. autofunction:: causalmp.basic_cmp_estimate

Environment Classes
-------------------

.. autoclass:: causalmp.BaseEnvironment
   :members:

.. autoclass:: causalmp.BeliefAdoptionEnvironment
   :members:

.. autoclass:: causalmp.NYCTaxiRoutesEnvironment
   :members:

.. autoclass:: causalmp.ExerciseEnvironment
   :members:

.. autoclass:: causalmp.AuctionEnvironment
   :members:

.. autoclass:: causalmp.DataCenterEnvironment
   :members:

.. autofunction:: causalmp.create_environment

Configuration Examples
======================

From Notebooks
---------------

**Belief Adoption Model (Krupina)**

.. code-block:: python

   environment = {
       'N': 3366,
       'setting': "Belief Adoption Model",
       'design': [0, 0.1, 0.2, 0.5],
       'stage_time_blocks': [1, 3, 5, 7],
       'desired_design_1': [0, 0],
       'desired_design_2': [0, 1],
       'desired_stage_time_blocks': [1, 7],
       'tau': 1
   }

**NYC Taxi Routes**

.. code-block:: python

   environment = {
       'N': 18768,
       'setting': "NYC Taxi Routes",
       'design': [0, 0.1, 0.2, 0.5],
       'stage_time_blocks': [1, 29, 57, 85],
       'desired_design_1': [0, 0],
       'desired_design_2': [0, 1],
       'desired_stage_time_blocks': [1, 85],
       'tau': 1
   }

**Parameter Ranges (from notebooks)**

.. code-block:: python

   main_param_ranges = {
       'n_lags_Y_range': [1],
       'n_lags_W_range': [1],
       'moment_order_p_Y_range': [1],
       'moment_order_p_W_range': [1],
       'moment_order_u_Y_range': [1],
       'moment_order_u_W_range': [1],
       'interaction_term_p_range': [None],
       'interaction_term_u_range': [None, 1],
       'n_batch_range': [100, 500, 1000],
       'batch_size_range': [int(0.05 * N), int(0.1 * N), int(0.2 * N)],
       'ridge_alpha_range': [1e-4, 1e-2, 1, 100]
   }

Utility Functions
-----------------

.. autofunction:: causalmp.get_version

.. autofunction:: causalmp.get_citation

.. autofunction:: causalmp.get_installation_status

.. autofunction:: causalmp.get_available_environments

.. autofunction:: causalmp.get_example_environment

.. autofunction:: causalmp.get_example_config

.. autofunction:: causalmp.verify_component_requirements

License
=======

This project is licensed under the MIT License.

Citation
========

If you use CausalMP in your research, please cite:

.. code-block:: bibtex

   @article{shirani2025can,
     title={Can We Validate Counterfactual Estimations in the Presence of General Network Interference?},
     author={Shirani, Sadegh and Luo, Yuwei and Overman, William and Xiong, Ruoxuan and Bayati, Mohsen},
     journal={arXiv preprint arXiv:2502.01106},
     year={2025}
   }

   @article{shirani2024causal,
     title={Causal message-passing for experiments with unknown and general network interference},
     author={Shirani, Sadegh and Bayati, Mohsen},
     journal={Proceedings of the National Academy of Sciences},
     volume={121},
     number={40},
     pages={e2322232121},
     year={2024},
     publisher={National Academy of Sciences}
   }

Dependencies
============

**Core Dependencies:**
- numpy >= 1.20.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy
- matplotlib
- seaborn

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`