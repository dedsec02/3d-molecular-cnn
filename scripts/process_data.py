#!/usr/bin/env python
"""
process_data.py

Stage 2: Data Gathering.
Parses the directory structure created by run_simulation.py, reads .out and .cube files,
calculates RDKit descriptors, and compiles everything into an HDF5 file.
"""

import os
import re
import argparse
import numpy as np
import h5py
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import List, Dict, Any

def get_molecule_descriptors(smiles: str) -> Dict[str, Any]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return {"Error": "Invalid SMILES"}
        return {name: func(mol) for name, func in Descriptors.descList}
    except Exception as e:
        return {"Error": str(e)}

def read_csv_data(filepath: str, cols: List[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath, skiprows=1, header=None, usecols=cols)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return pd.DataFrame()

def process_cube_file(file_path: str) -> np.ndarray:
    """Reads a .cube file and returns volumetric data."""
    if not os.path.exists(file_path): return None
    
    try:
        with open(file_path, 'r') as f:
            f.readline(); f.readline() # Comments
            natoms = int(f.readline().split()[0])
            nx, _, _ = [int(x) for x in f.readline().split()[1:]]
            ny, _, _ = [int(x) for x in f.readline().split()[1:]]
            nz, _, _ = [int(x) for x in f.readline().split()[1:]]
            
            # Skip atoms
            for _ in range(abs(natoms)): f.readline()
            
            data = []
            for line in f:
                data.extend([float(v) for v in line.split()])
                
        return np.array(data).reshape((nx, ny, nz))
    except Exception as e:
        print(f"Error reading cube {file_path}: {e}")
        return None

def parse_orca_output(file_path: str) -> Dict[str, float]:
    data = {}
    if not os.path.exists(file_path): return data
    
    with open(file_path, 'r') as f: content = f.read()
    
    # Energies
    match_homo = re.findall(r"\s+\d+\s+2\.0000\s+-?\d+\.\d+\s+(-?\d+\.\d+)", content)
    match_lumo = re.findall(r"\s+\d+\s+0\.0000\s+-?\d+\.\d+\s+(-?\d+\.\d+)", content)
    
    if match_homo: data['homo_ev'] = float(match_homo[-1])
    if match_lumo: data['lumo_ev'] = float(match_lumo[0])
        
    match_sp = re.search(r"FINAL SINGLE POINT ENERGY\s+([\-\d\.]+)", content)
    if match_sp: data['final_energy'] = float(match_sp.group(1))
        
    match_dip = re.search(r"Magnitude \(Debye\)\s+:\s+([\d\.]+)", content)
    if match_dip: data['dipole_debye'] = float(match_dip.group(1))
        
    return data

def main(root_dir: str, csv_path: str, output_name: str):
    hdf5_path = os.path.join(root_dir, output_name)
    print(f"Scanning {root_dir}...")
    
    # Load metadata CSV (Expects: Index, SMILES, Label)
    # Adjust columns [0, 1, 2] based on your specific CSV structure
    df = read_csv_data(csv_path, [0, 1, 2])
    if df.empty: return

    with h5py.File(hdf5_path, 'w') as h5:
        # Assumes structure: root/class_folder/example_folder/frames/frame_folder
        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)
            if not os.path.isdir(class_path): continue
            
            out_data_path = os.path.join(class_path, "output_data")
            if not os.path.exists(out_data_path): continue

            for ex_dir in os.listdir(out_data_path):
                if not ex_dir.isdigit(): continue # Skip non-integer folders
                
                ex_idx = int(ex_dir)
                ex_path = os.path.join(out_data_path, ex_dir)
                frames_path = os.path.join(ex_path, "frames")
                
                # Create Group
                grp_path = f"/{class_dir}/{ex_dir}"
                ex_group = h5.create_group(grp_path)

                # RDKit Descriptors
                try:
                    smiles = df.iloc[ex_idx+1, 1] # Adjust index logic if needed
                    descs = get_molecule_descriptors(smiles)
                    vals = [v for v in descs.values() if isinstance(v, (int, float))]
                    ex_group.create_dataset("rdkit_descriptors", data=vals)
                except Exception as e:
                    print(f"Skipping RDKit for {ex_dir}: {e}")

                if not os.path.exists(frames_path): continue

                for frame in os.listdir(frames_path):
                    f_path = os.path.join(frames_path, frame)
                    if not os.path.isdir(f_path): continue
                    
                    frame_grp = h5.create_group(f"{grp_path}/{frame}")
                    
                    # 1. ORCA Scalars
                    orca_data = parse_orca_output(os.path.join(f_path, f"{frame}.out"))
                    if orca_data:
                        frame_grp.create_dataset("orca_scalars", data=list(orca_data.values()))
                    
                    # 2. Cube Tensors
                    for c_name in ["homo", "lumo", "density"]:
                        c_data = process_cube_file(os.path.join(f_path, f"{c_name}.cube"))
                        if c_data is not None:
                            frame_grp.create_dataset(f"{c_name}_cube", data=c_data, compression="gzip")
                            
    print(f"Done. Data saved to {hdf5_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root directory containing class folders")
    parser.add_argument("--csv", required=True, help="Path to properties CSV (SMILES/Labels)")
    parser.add_argument("--out", default="simulation_data.h5", help="Output HDF5 filename")
    args = parser.parse_args()
    
    main(args.root, args.csv, args.out)
