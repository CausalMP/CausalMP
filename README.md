# causalmp

A Python package for counterfactual estimation and simulation in causal inference scenarios. The package provides tools for researchers and practitioners to perform counterfactual analysis through both estimation and simulation approaches.

## Features

### Estimator Module
- Counterfactual Evolution (CFE) estimation
- Semi-recursive estimation
- Cross-validation for hyperparameter selection
- Classic estimators (DinM, HT)
- Feature engineering and batch processing
- Statistical moment calculation

### Simulator Module
- Multiple simulation environments:
  - Belief Adoption Model (social network influence)
  - Auction Model (market dynamics)
  - NYC Taxi Routes (transportation patterns)
  - Exercise Encouragement Program (health interventions)
  - Data Center Model (resource allocation)
- Staggered rollout support
- Customizable parameters
- Parallel execution capabilities

### Runner Module
- Combined simulation and estimation
- Result visualization
- Multi-run experiments
- Parallel processing support

## Installation

### Development Installation (Current)

Clone the repository and install in development mode:
```bash
git clone https://github.com/CausalMP/CausalMP.git
cd CausalMP
pip install -e .
```

Install with specific components:
```bash
# Estimator only
pip install -e .[estimator]

# Simulator only  
pip install -e .[simulator]

# All components
pip install -e .[all]

# Development setup with testing tools
pip install -e .[dev]
```

## Available Environments

1. **Belief Adoption Model**
   - Social network belief propagation
   - Treatment effects on belief adoption

2. **Auction Model**
   - Multi-bidder market dynamics
   - Treatment effects on object valuations

3. **NYC Taxi Routes**
   - Transportation network with pricing algorithm experiment
   - Treatment effects on route selection

4. **Exercise Encouragement Program**
   - Health intervention effects with social network influence
   - Behavioral change dynamics

5. **Data Center Model**
   - Distributed service system with join-the-shortest-queue routing policy
   - Treatment effects on system efficiency

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:
```bibtex
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
```

## Dependencies

- numpy
- pandas
- scikit-learn
- scipy
- matplotlib
- seaborn
- joblib (optional, for parallel processing)