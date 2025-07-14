import os
import re
import sys
import trimesh
import numpy as np
from stl import mesh
from tqdm import tqdm

from IPython.display import clear_output
from joblib import Parallel, delayed 

cur_dir = os.getcwd()

bcc = mesh.Mesh.from_file(os.path.join(cur_dir, 'unit_cells/BCC.stl'))
fcc = mesh.Mesh.from_file(os.path.join(cur_dir, 'unit_cells/FCC.stl'))
ot = mesh.Mesh.from_file(os.path.join(cur_dir, 'unit_cells/OT.stl'))
sc = mesh.Mesh.from_file(os.path.join(cur_dir, 'unit_cells/SC.stl'))
dia = mesh.Mesh.from_file(os.path.join(cur_dir, 'unit_cells/DIA.stl'))

output_path_binary = os.path.join(cur_dir, 'binary_matrices') 
if not os.path.exists(output_path_binary):
    os.makedirs(output_path_binary)
         
nx, ny, nz = 4, 4, 4
dx, dy, dz = 2.5, 2.5, 2.5  
array_size = 164

def create_unit_cell_copy(i, j, k, cell_type):
    if cell_type == 0:
        unit_cell = bcc
    elif cell_type == 1:
        unit_cell = fcc
    elif cell_type == 2:
        unit_cell = ot
    elif cell_type == 3:
        unit_cell = sc
    else:
        unit_cell = dia

    unit_cell_copy = mesh.Mesh(unit_cell.data.copy())
    unit_cell_copy.x -= i * dx
    unit_cell_copy.y -= j * dy
    unit_cell_copy.z -= k * dz
    return unit_cell_copy
    
num_matrices = len(os.listdir(os.path.join(cur_dir, 'initial_population')))
    
for num in tqdm(range(num_matrices)):    
    matrix = np.load(os.path.join(cur_dir, f'initial_population/matrix_{num}.npy'))

    unit_cells = []
    for z in range(nz):
        layer_cells = Parallel(n_jobs=-1)(delayed(create_unit_cell_copy)(x, y, z, matrix[z, x, y])
            for x in range(nx) for y in range(ny))
        unit_cells.extend(layer_cells)  
    
    supercell = mesh.Mesh(np.concatenate([cell.data for cell in unit_cells]))
    supercell.save('supercell.stl')
    lattice = trimesh.load_mesh('supercell.stl')
    ext_x, ext_y, ext_z = lattice.bounding_box.extents
    pitch = ext_x / (array_size-1) 
    volume = lattice.voxelized(pitch=pitch).fill()
    mat = volume.matrix.astype(int)
    
    mask = mat[mat != 0]    
    np.save(os.path.join(output_path_binary, f'matrix_{num}.npy'), mat)
