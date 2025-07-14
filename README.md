# 3D-CNN for Specific Elastic Modulus Prediction

This repository contains the training and testing pipeline for a 3D Convolutional Neural Network used to predict a mechanical property (e.g. specific elastic modulus) of 3D lattice structures represented as binary matrices. It is part of an active learning framework that accelerates the discovery of high-performance metamaterials.

### Nanoporous structures (training dataset) generation 
The training dataset contains 1,000 lattices. Each lattice consists of 4×4×4 unit cells and includes five possible cell types: Body Centered Cubic (BCC), Face Centered Cubic (FCC), Octet Truss (OT), Simple Cubic (SC) and Diamond (DIA). All the unit cells have the same diameter of ligaments. To define a lattice composition (i.e. to determine the cell type at each of the 64 positions in a lattice), there is a corresponding .npy matrix. Run the following to generate the training data:

```
python gen_train_matrices.py

```

### Model training
Once your training binary matrices are ready, you can start model training. The target values of FFT-calculated specific elastic modulus are already provided in a separate array.
To train the model, run the following command, adjusting the hyperparameters as needed:

```
python train_model.py --batch_size 10 --n_epochs 100 --n_folds 5

```

