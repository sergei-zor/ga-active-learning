# 3D-CNN for Specific Elastic Modulus Prediction

This repository contains the code for the paper:

*Does high entropy improve elastic properties of 3D lattice materials?—A genetic algorithm and active learning study*

by S. Zorkaltsev, J. Segurado, M.T. Pérez-Prado and M. Haranczyk

[Comput. Mater. Sci., vol. 262, p. 114332, 2026.](https://doi.org/10.1016/j.commatsci.2025.114332)

## Repository overview

- `gen_train_matrices.py` — generation of the training data (voxelized lattice structures)  
- `train_model.py` — 3DCNN training with k-fold cross-validation
- `DenseNet3D.py` — model architecture definitions  
- `unit_cells/` — .stl meshes for supercells generation 
- `initial_population/` — 4×4×4 matrices encoding the cell types
-  `binary_matrices/` — several examples of training structures for the smoke test

## Dataset generation

The training dataset consists of 1,000 lattice structures, each composed of a 4×4×4 arrangement of unit cells. Five unit cell topologies:

- Body-Centered Cubic (BCC)
- Face-Centered Cubic (FCC)
- Octet Truss (OT)
- Simple Cubic (SC)
- Diamond (DIA)

All unit cells have the same ligament diameter.  
Each lattice composition is defined by a corresponding `.npy` matrix specifying the unit cell type at each of the 64 lattice positions.

To generate the voxelized training data, run:

```
python gen_train_matrices.py

```

## Model training
The target values of FFT-calculated specific elastic modulus are provided in a separate array.
To train the model, run the following command, adjusting the hyperparameters as needed:

```
python train_model.py --batch_size 10 --n_epochs 100 --n_folds 5

```
The smoke test runs the full training on a small provided dataset to verify correct installation.

```
python train_model.py --smoke_test

```

## Citation

If you use part of the paper/code, please consider citing:

```
@article{ZORKALTSEV2026114332,
title = {Does high entropy improve elastic properties of 3D lattice materials?—A genetic algorithm and active learning study},
journal = {Computational Materials Science},
volume = {262},
pages = {114332},
year = {2026},
issn = {0927-0256},
doi = {https://doi.org/10.1016/j.commatsci.2025.114332},
url = {https://www.sciencedirect.com/science/article/pii/S0927025625006755},
author = {Sergei Zorkaltsev and Javier Segurado and María Teresa Pérez-Prado and Maciej Haranczyk},
keywords = {Genetic algorithm, Optimization, Structure-property relationship, Convolutional neural networks}}
```
## Requirements

The code requires the following Python packages:

- NumPy
- Pandas
- PyTorch
- scikit-learn
- MLflow
- tqdm
- joblib
- trimesh
- numpy-stl
