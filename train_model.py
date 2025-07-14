import re
import os
import sys
import time
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.distributed as dist
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings("ignore")

from DenseNet3D import DenseNet

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds for cross-validation")
    return vars(parser.parse_args())
    
args = parse_command_line_args()
batch_size = args['batch_size']
learning_rate = args['lr']
n_epochs = args['n_epochs']
n_folds = args['n_folds']

cur_dir = os.path.dirname(os.path.abspath(__file__))
matrix_path = os.path.join(cur_dir, 'binary_matrices')

output_results = os.path.join(cur_dir, 'train_results') 
if not os.path.exists(output_results):
    os.makedirs(output_results)


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


def train_model(model, dataloaders, criterion, optimizer, num_epochs=20, device='cpu', fold = 0, folder = os.path.dirname(os.path.abspath(__file__))):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    r2_train_storage = []
    r2_test_storage = []
    
    loss_test_storage = []
    loss_train_storage = []
    
    patience = 10  
    min_delta = 0.005  
    best_loss = float('-inf')
    patience_counter = 0

    model.to(device)
    if torch.cuda.device_count() > 1:
    	print(torch.cuda.device_count())
    	available_gpus = list(range(torch.cuda.device_count()))  
    	model = torch.nn.DataParallel(model, device_ids=available_gpus)
    	

    best_r2 = float('-inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            true_vals, pred_vals = [], []

            for inputs, targets in tqdm(dataloaders[phase], desc=f'{phase} loop'):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, targets)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                true_vals.extend(targets.cpu().numpy())
                pred_vals.extend(outputs.view(-1).detach().cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_r2 = r2_score(true_vals, pred_vals)

            print(f"{phase} Loss: {epoch_loss:.4f} | {phase} R²: {epoch_r2:.4f}")

            if phase == 'train':
                r2_train_storage.append(epoch_r2)
                loss_train_storage.append(epoch_loss)
                scheduler.step(epoch_loss)
            else:
                r2_test_storage.append(epoch_r2)
                loss_test_storage.append(epoch_loss)
                
                if epoch_r2 > best_r2:
                    best_r2 = epoch_r2
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(folder, f'best_model_{fold+1}.pth'))
                    np.save(os.path.join(folder, f'true_test_{fold+1}.npy'), np.array(true_vals))
                    np.save(os.path.join(folder, f'pred_test_{fold+1}.npy'), np.array(pred_vals))
                    
        plt.figure(figsize = (8,8))
        plt.rc('xtick', labelsize=14) 
        plt.rc('ytick', labelsize=14)  
        plt.scatter(np.array(true_vals), np.array(pred_vals))
        plt.plot([min(np.array(true_vals))*0.95, max(np.array(true_vals))*1.05], [min(np.array(true_vals))*0.95, max(np.array(true_vals))*1.05], '--', color='black')
        plt.xlabel('FFT-calculated specific E, GPa', fontsize = 14)
        plt.ylabel('Predicted specific E, GPa', fontsize = 14)
        plt.savefig(os.path.join(folder, f'fold_{fold+1}.png'))
                    
        if epoch_r2 > (best_loss + min_delta):
            best_loss = epoch_r2
            patience_counter = 0  
        else:
            patience_counter += 1  

        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break 

        np.save(os.path.join(folder, f'r2_test_{fold+1}.npy'), np.array(r2_test_storage))
        np.save(os.path.join(folder, f'r2_train_{fold+1}.npy'), np.array(r2_train_storage))
        
        np.save(os.path.join(folder, f'loss_test_{fold+1}.npy'), np.array(loss_test_storage))
        np.save(os.path.join(folder, f'loss_train_{fold+1}.npy'), np.array(loss_train_storage))

    print(f'Fold {fold+1}: best model at epoch {best_epoch} with R²: {best_r2:.4f}')

kf = KFold(n_splits=n_folds, shuffle = True, random_state = 0)

target_array = np.load(os.path.join(cur_dir, 'specific_E.npy'))

for i, (train_index, test_index) in enumerate(kf.split(target_array)):
    print(f'Cross-validation fold {i+1}')
    train_targets = target_array[train_index]
    test_targets = target_array[test_index]

    train_dataset = Custom3DDataset(matrix_path, train_index, train_targets, augment=True)
    test_dataset = Custom3DDataset(matrix_path, test_index, test_targets, augment=False)

    dataloaders = {'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                   'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)}
                   
    model = DenseNet()
                       
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=5)
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    folder = os.path.join(output_results, f'fold_{i+1}')
    train_model(model, dataloaders, criterion, optimizer, num_epochs=n_epochs, device = device, fold = i, folder = folder)
    
    true = np.load(os.path.join(folder, f'true_test_{i+1}.npy')) 
    pred = np.load(os.path.join(folder, f'pred_test_{i+1}.npy')) 

    plt.figure(figsize = (8,8))
    plt.rc('xtick', labelsize=14) 
    plt.rc('ytick', labelsize=14)  
    plt.scatter(true, pred)
    r2 = r2_score(true, pred)
    mse = mean_squared_error(true, pred) 
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, pred) 
    plt.plot([min(true)*0.95, max(true)*1.05], [min(true)*0.95, max(true)*1.05], '--', color='black')
    plt.xlabel("FFT-calculated specific E, GPa", fontsize = 14)
    plt.ylabel("Predicted specific E, GPa", fontsize = 14)
    plt.title(f'Best R²: {r2:.3f}, RMSE: {rmse:.3f} GPa, MAE: {mae:.3f} GPa')
    plt.savefig(os.path.join(folder, f'fold_{i+1}.png'))
    print(f'Best R²: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}')
                       


