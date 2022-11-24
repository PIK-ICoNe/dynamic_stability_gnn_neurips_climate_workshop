# Companion to paper: Towards dynamic stability analysis of sustainable power grids using graph neural networks

This repository contains the relevant information to reproduce the results of the paper "Towards dynamical stability analysis of sustainable power grids using Graph Neural Networks" published at Tackling Climate Change with Machine Learning: workshop at NeurIPS 2022.

If you want to use the code to train similar models, you can check out the jupyter notebook in the directory ```ml_examples```, where you can use the ipynb to train models and the scripts include the automatic download of the datasets from Zenodo. 


## Generation of the datasets
The generation of the datasets occurs in multiple steps. To reproduce the results, we reccommend using Julia 1.5.3 and use the provided Project.toml and Manifest.toml to have the same environment. The corresponding path to activate the environment has to be set in all files. After manipulation of the paths, the scripts can be consecutively executed. Keep in mind that the dynamical computations are very expensive (~550,000 CPU hours for dataset20 and dataset100).

1. Grid generation: ```generate_grids_and_seeds.jl```
2. Dynamical computation: ```compute_dynamics.jl```
3. Preparation for  ML training: ```prepare4Pytorch.jl```
4. Preparation for linear regression and MLPs: ```prepare4Scikit.jl```

Prior to executing the scripts, the paths to store/loading the grid data and store/loading the dynamical results, as well as the number of desired nodes per grid must be adapted in ```generate_grids_and_seeds.jl``` and ```compute_dynamics.jl```. Furthermore, the path to the girds, dymnamical results and the output directory for the ML data has to be set in ```prepare4Pytorch.jl``` and ```prepare4Scikit.jl```.

## Traning of the ML models using the examples
To evaluate the reproducibility, we provide one Jupyter notebook and a python script to train the ArmaNet model after automatically downloading the datasets from Zenodo. We provide a conda environment file to generate a conda environment including all of the necessary software. 
To create the conda environment, the ENVNAME can be freely chosen. 
```
conda env create -n ENVNAME --file conda_environment.yml
```
The file conda_environment.yml is stored in training_model.
Afterwards the script train_ArmaNet.py or train_ArmaNet.ipynb can be executed.

## Training of models including parellalization with ray and hyperparameter optimization
The directory ```ml_extended``` contains the scripts for parellalization and hyperparameter optimization using ```ray```. After defining the relevant paths in ```start_ray.py```, this file has to be executed.


In case of any problems or questions, do not hesitate to contact us.
