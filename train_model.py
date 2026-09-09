import re
import os
import sys
import time
import tempfile
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import mlflow
import mlflow.pytorch

import warnings
warnings.filterwarnings('ignore')

from DenseNet3D import DenseNet

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--smoke_test', action='store_true', help='Run smoke test only')
    return vars(parser.parse_args())
    
def make_run_name(smoke_test, lr, fold, n_folds):
    if smoke_test:
        return "smoke_test"
    return f"lr{lr:.2e}_fold{fold+1}of{n_folds}"
    
def get_true_pred(model, dataloader, device='cuda'):
    true_vals, pred_vals = [], []
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            true_vals.extend(targets.cpu().numpy())
            pred_vals.extend(outputs.view(-1).detach().cpu().numpy())
    return true_vals, pred_vals
    
class Custom3DDataset(Dataset):
    def __init__(self, folder_path, file_indices, target_array, augment=False):
        self.folder_path = folder_path
        self.file_indices = file_indices
        self.targets = target_array
        self.augment = augment
        self.total_len = len(self.file_indices) * (8 if augment else 1)  

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        file_idx = idx // 8 if self.augment else idx  
        transformation = idx % 8  

        matrix_index = self.file_indices[file_idx]
        file_name = f'matrix_{matrix_index}.npy'

        matrix = np.load(os.path.join(self.folder_path, file_name))
        supercell = matrix

        if transformation == 0:  
            supercell = supercell.copy()  
        elif transformation == 1:  
            supercell = np.rot90(supercell, k=1, axes=(0, 1)).copy()                       
        elif transformation == 2:  
            supercell = np.rot90(supercell, k=2, axes=(0, 1)).copy()    
        elif transformation == 3: 
            supercell = np.rot90(supercell, k=3, axes=(0, 1)).copy()                         
        elif transformation == 4:  
            supercell = np.flip(supercell, axis=2).copy()
        elif transformation == 5:  
            supercell = np.flip(np.rot90(supercell, k=1, axes=(0, 1)), axis=2).copy()
        elif transformation == 6:  
            supercell = np.flip(np.rot90(supercell, k=2, axes=(0, 1)), axis=2).copy()                      
        elif transformation == 7:   
            supercell = np.flip(np.rot90(supercell, k=3, axes=(0, 1)), axis=2).copy()

        target = self.targets[file_idx]
        return torch.tensor(supercell, dtype=torch.float32).unsqueeze(0), torch.tensor(target, dtype=torch.float32)


def train_model(model,
                dataloaders,
                criterion,
                optimizer,
                num_epochs=20,
                device='cpu',
                fold=0,
                scheduler=None,
                experiment_name=None,
                run_name=None,
                patience=10,
                min_delta=0.005):
                
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    best_r2 = float('-inf')
    best_metric_for_es = float('-inf')
    best_epoch = 0
    patience_counter = 0

    if experiment_name is not None:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param('num_epochs', num_epochs)
        mlflow.log_param('fold', fold)
        mlflow.log_param('optimizer', type(optimizer).__name__)
        mlflow.log_param('criterion', type(criterion).__name__)
        mlflow.log_param('scheduler', type(scheduler).__name__ if scheduler else 'None')

        for epoch in range(num_epochs):
            ### Train
            model.train()
            train_loss = 0.0
            y_true_train, y_pred_train = [], []

            for inputs, targets in tqdm(dataloaders['train'], desc='train'):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                y_true_train.extend(targets.detach().cpu().numpy().ravel())
                y_pred_train.extend(outputs.detach().cpu().numpy().ravel())

            train_loss /= len(dataloaders['train'].dataset)
            train_r2 = r2_score(y_true_train, y_pred_train)

            ### Validation
            model.eval()
            val_loss = 0.0
            y_true_val, y_pred_val = [], []

            with torch.no_grad():
                for inputs, targets in tqdm(dataloaders['val'], desc='val'):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, targets)

                    val_loss += loss.item() * inputs.size(0)
                    y_true_val.extend(targets.detach().cpu().numpy().ravel())
                    y_pred_val.extend(outputs.detach().cpu().numpy().ravel())

            val_loss /= len(dataloaders['val'].dataset)
            val_r2 = r2_score(y_true_val, y_pred_val)

            print(f'Epoch {epoch+1}: Train loss={train_loss:.4f}, Train $R^2$={train_r2:.4f} Val loss={val_loss:.4f}, Val $R^2$={val_r2:.4f}')
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            mlflow.log_metric('train_r2', train_r2, step=epoch)
            mlflow.log_metric('val_r2', val_r2, step=epoch)

            if scheduler is not None:
                scheduler.step(val_loss)

            if val_r2 > best_r2:
                best_r2 = val_r2
                best_epoch = epoch

                state_dict = (
                    model.module.state_dict()
                    if hasattr(model, 'module')
                    else model.state_dict()
                )

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = os.path.join(tmpdir, f'best_model_fold_{fold}.pth')
                    torch.save(state_dict, tmp_path)
                    mlflow.log_artifact(tmp_path, artifact_path='models')

            if val_r2 > best_metric_for_es + min_delta:
                best_metric_for_es = val_r2
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        print(f'Fold {fold}: best model at epoch {best_epoch + 1} with Val $R^2$ = {best_r2:.4f}')
                  
args = parse_command_line_args()
batch_size = args['batch_size']
learning_rate = args['lr']
n_epochs = args['n_epochs']
n_folds = args['n_folds']
smoke_test = args['smoke_test']
augment_train = True

current_dir = os.path.dirname(os.path.abspath(__file__))
matrix_path = os.path.join(current_dir, 'binary_matrices')
target_array = np.load(os.path.join(current_dir, 'specific_E.npy'))

if smoke_test:
    batch_size = 1
    n_folds = 2
    n_epochs = 3
    augment_train = False
    n_available = len(os.listdir(matrix_path))
    target_array = target_array[:n_available]
    print(f'Running smoke test with {n_available} lattice structures')
    
if not smoke_test:
    if len(target_array) != len(os.listdir(matrix_path)):
        raise TypeError('Number of training matrices does not match the target array')

kf = KFold(n_splits=n_folds, shuffle = True, random_state = 0)

for i, (train_index, test_index) in enumerate(kf.split(target_array)):
    print(f'Cross-validation fold {i}')
    train_targets = target_array[train_index]
    test_targets = target_array[test_index]
    
    train_idx_sub, val_idx_sub = train_test_split(train_index, test_size=0.2, random_state=0, shuffle=True)
    
    train_dataset = Custom3DDataset(matrix_path, train_idx_sub, target_array[train_idx_sub], augment=augment_train)
    val_dataset   = Custom3DDataset(matrix_path, val_idx_sub,   target_array[val_idx_sub],   augment=False)
    test_dataset  = Custom3DDataset(matrix_path, test_index,    target_array[test_index],    augment=False)

    dataloaders = {'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                   'val':   DataLoader(val_dataset,   batch_size=batch_size, shuffle=False),
                   'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)}
                   
    print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')
                   
    model = DenseNet()           
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=8)
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(model, dataloaders, criterion, optimizer, num_epochs=n_epochs, device = device, fold = i)
    
