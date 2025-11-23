# 3D-CNN for Molecular Property Prediction

This repository contains a complete workflow for generating quantum chemical data, processing volumetric electronic densities, and training a 3D Convolutional Neural Network (CNN).

## Overview

The workflow consists of three stages:
1.  **Simulation:** Running QM/MD simulations and calculating electronic properties (HOMO, LUMO, Density) using ORCA.
2.  **Processing:** Extracting volumetric cube files and scalar descriptors into a compressed HDF5 dataset.
3.  **Training:** Training a custom 3D CNN on the volumetric data using TensorFlow.

## Project Structure

```text
├── scripts/
│   ├── run_simulation.py   # Automates ORCA MD and Single-Point calculations
│   └── process_data.py     # Parses output files into HDF5 format
├── src/
│   ├── generator.py        # TensorFlow Data Generator for HDF5
│   └── model.py            # 3D CNN Model definition and training loop
├── README.md
└── requirements.txt


## 🛠️ Detailed Usage Guide

### 0. Preparation
Before running the scripts, you must set up your directory and create the required ORCA template files.

#### A. Directory Setup
Create a folder for your input molecules and place your `.xyz` files there:
```bash
mkdir -p data/input_xyz
# Place your molecule.xyz files inside data/input_xyz/
```

#### B. CSV Data Format
The processing script (`scripts/process_data.py`) requires a CSV file to link molecule properties (SMILES, Labels) to the directory structure.
**Format:** The CSV **must** be headerless (or skip the first row) and follow this column order:
1.  **Column 0:** Index/ID (Integer)
2.  **Column 1:** SMILES String
3.  **Column 2:** Label/Target Value (e.g., pIC50)

*Example `data/molecules.csv`:*
```csv
No,SMILES,Activity
0,CC(=O)Oc1ccccc1C(=O)O,1
1,CN1C=NC2=C1C(=O)N(C(=O)N2C)C,0
```

#### C. ORCA Templates
Create two template files in the root directory so the script knows how to run ORCA. The python script replaces keywords inside curly braces `{}`.

**1. `orca_md_template.inp`** (For Molecular Dynamics)
```text
! {METHOD_SETTINGS} MD OPT
%maxcore 2000
%pal nprocs {NPROCS} end
* xyz {CHARGE} {MULTIPLICITY}
{COORDINATES}
*
%md
   timestep 1.0
   initvel 300
   thermostat thermostat vrescale eta 5.0 temp 300.0 end
   time 200.0 # Duration in fs
   dump step 1
   dump position xyz
end
```

**2. `orca_surface_template.inp`** (For Cube Generation)
```text
! {METHOD_SETTINGS}
%maxcore 2000
%pal nprocs {NPROCS} end
%plots
  dim1 {GRID_POINTS}
  dim2 {GRID_POINTS}
  dim3 {GRID_POINTS}
  Format Cube
  MO("homo.cube", {HOMO_O}, 0);
  MO("lumo.cube", {LUMO_O}, 0);
  ElDens("density.cube");
end
* xyz {CHARGE} {MULTIPLICITY}
{COORDINATES}
*
```

---

### 1. Run Simulation (Data Generation)
This script runs the MD simulation, extracts frames, and generates `.cube` files for electron density.

```bash
python scripts/run_simulation.py \
  --input_dir data/input_xyz \
  --output_dir output/raw_data \
  --orca_path /usr/bin/orca \
  --nprocs 4 \
  --stride 10 \
  --grid 80
```
*   `--orca_path`: Full path to your ORCA binary.
*   `--stride`: Analyzes every Nth frame of the trajectory.
*   `--grid`: Resolution of the 3D cube (80x80x80).

---

### 2. Process Data (HDF5 Creation)
This script scans the folders generated in Step 1, parses the `.cube` files, calculates RDKit descriptors, and packs everything into a single compressed HDF5 file.

```bash
python scripts/process_data.py \
  --root output/raw_data \
  --csv data/molecules.csv \
  --out output/training_data.h5
```
*   `--root`: The folder where `run_simulation.py` saved its output.
*   `--csv`: The metadata CSV file created in step 0.

---

### 3. Train Model
Train the 3D CNN using the generated HDF5 file.

```bash
python src/model.py \
  --data output/training_data.h5 \
  --epochs 50 \
  --batch 16
```
*   The model automatically detects input shapes and splits the data into batches.
*   It outputs `sparse_categorical_accuracy` and loss per epoch.
