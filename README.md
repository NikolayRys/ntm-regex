# Algorithmic inference using Neural Turing Machines
Final project for University of London, Goldsmith, 2023 by Nikolay Rys.

## Main idea
NTM machines to learn regex from data. 

## Project structure
* The main code is located in the `ntm-regex.ipynb` notebook that contains the model definitions and experiments.
* The implementation for NTM for Tensorflow 2.0 is located in the `ntm.py` file.
* `datagen.py` contains the code for generating data for the experiments.
* In the `artefacts` folder you can find the saved models.
* In the `datasets` folder you can find the datasets used for the experiments.

## Requirements
* Python 3.9
* Tensorflow 2.4+
* CUDA 11.0+
* cuDNN 8.0+
* automata_toolkit 0.1.0

Licenced as LGPLv3 because the initial code is based on the [Neural Turing Machines by Mark Collier](https://github.com/MarkPKCollier/NeuralTuringMachine)