# ANFIS Implementation

This repository contains the implementation of an Adaptive Neuro-Fuzzy Inference System (ANFIS) for a given dataset. The ANFIS model is trained using the provided Python files: `myANFIS.py` and `test.py`.

## Files:

1. **myANFIS.py**
   - This file contains the implementation of the ANFIS model, including functions for custom Gaussian bell-shaped membership functions, training the model, and calculating the outputs.

2. **test.py**
   - This file is an example script demonstrating how to use the ANFIS model. It loads data from `iris.csv`, scales the input data, trains the ANFIS model, and evaluates its performance.

## Usage:

To use the ANFIS model in your project, follow these steps:

1. Ensure you have the required dependencies installed by running:
   ```bash
   pip install numpy matplotlib scikit-fuzzy
