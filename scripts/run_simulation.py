#!/usr/bin/env python
"""
run_simulation.py

Stage 1: Automates ORCA workflow.
1. Runs a QM/MD (AIMD) simulation for molecules in an input folder.
2. Extracts frames and runs single-point calculations (HOMO, LUMO, Density).
"""

import os
import glob
import subprocess
import shutil
import json
import re
import argparse
from typing import List, Tuple, Dict, Any

# --- Atomic Data ---
ATOMIC_NUMBERS = {
    'H': 1, 'HE': 2, 'LI': 3, 'BE': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'NE': 10,
    'NA': 11, 'MG': 12, 'AL': 13, 'SI': 14, 'P': 15, 'S': 16, 'CL': 17, 'AR': 18,
    'K': 19, 'CA': 20, 'SC': 21, 'TI': 22, 'V': 23, 'CR': 24, 'MN': 25, 'FE': 26,
    'CO': 27, 'NI': 28, 'CU': 29, 'ZN': 30, 'GA': 31, 'GE': 32, 'AS': 33, 'SE': 34,
    'BR': 35, 'KR': 36, 'RB': 37, 'SR': 38, 'Y': 39, 'ZR': 40, 'NB': 41, 'MO': 42,
    'TC': 43, 'RU': 44, 'RH': 45, 'PD': 46, 'AG': 47, 'CD': 48, 'IN': 49, 'SN': 50,
    'SB': 51, 'TE': 52, 'I': 53, 'XE': 54, 'CS': 55, 'BA': 56, 'LA': 57, 'CE': 58,
    'PR': 59, 'ND': 60, 'PM': 61, 'SM': 62, 'EU': 63, 'GD': 64, 'TB': 65, 'DY': 66,
    'HO': 67, 'ER': 68, 'TM': 69, 'YB': 70, 'LU': 71, 'HF': 72, 'TA': 73, 'W': 74,
    'RE': 75, 'OS': 76, 'IR': 77, 'PT': 78, 'AU': 79, 'HG': 80, 'TL': 81, 'PB': 82,
    'BI': 83, 'PO': 84, 'AT': 85, 'RN': 86,
}

def read_xyz_coords(xyz_file: str) -> str:
    with open(xyz_file, 'r') as f:
        lines = f.readlines()[2:]  # Skip atom count and comment
    return "".join(lines).strip()

def calculate_total_electrons(xyz_file_path: str, charge: int) -> int:
    total_protons = 0
    with open(xyz_file_path, 'r') as f:
        lines = f.readlines()[2:]
    
    for line in lines:
        line = line.strip()
        if not line: continue
        parts = line.split()
        symbol = parts[0].upper()
        protons = ATOMIC_NUMBERS.get(symbol)
        if protons is None:
            raise ValueError(f"Unknown element '{symbol}' in {xyz_file_path}")
        total_protons += protons
    return total_protons - charge

def parse_md_trajectory(traj_file: str) -> List[Tuple[int, str]]:
    frames = []
    if not os.path.exists(traj_file): return frames
    
    with open(traj_file, 'r') as f:
        frame_idx = 1
        while True:
            line = f.readline()
            if not line: break
            try:
                natoms = int(line.strip())
            except ValueError: continue
            f.readline() # Skip comment
            coords = [f.readline().strip() for _ in range(natoms)]
            frames.append((frame_idx, "\n".join(coords)))
            frame_idx += 1
    return frames

def run_orca(cmd_path: str, input_file: str, output_file: str, work_dir: str) -> None:
    with open(output_file, 'w') as f_out:
        subprocess.run([cmd_path, input_file], cwd=work_dir, stdout=f_out, stderr=subprocess.PIPE, check=True, text=True)

def parse_sp_output(output_file: str) -> Dict[str, Any]:
    properties = {"energy_au": None, "homo_ev": None, "lumo_ev": None, "dipole_debye": None}
    if not os.path.exists(output_file): return properties
    
    with open(output_file, 'r') as f: content = f.read()

    match = re.search(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", content)
    if match: properties["energy_au"] = float(match.group(1))

    match = re.search(r"Total Dipole Moment\s+:\s+(-?\d+\.\d+)", content)
    if match: properties["dipole_debye"] = float(match.group(1))

    match = re.search(r"ORBITAL ENERGIES", content)
    if match:
        text_block = content[match.end():]
        end_match = re.search(r"\n\n\s*\w", text_block, re.MULTILINE)
        if end_match: text_block = text_block[:end_match.start()]
        
        occupied = re.findall(r"\s+\d+\s+2\.0000\s+-?\d+\.\d+\s+(-?\d+\.\d+)", text_block)
        unoccupied = re.findall(r"\s+\d+\s+0\.0000\s+-?\d+\.\d+\s+(-?\d+\.\d+)", text_block)
        if occupied: properties["homo_ev"] = float(occupied[-1])
        if unoccupied: properties["lumo_ev"] = float(unoccupied[0])
    return properties

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Templates
    try:
        with open(args.md_template, 'r') as f: md_template = f.read()
        with open(args.surf_template, 'r') as f: surface_template = f.read()
    except FileNotFoundError:
        print("[!] Templates not found.")
        return

    input_files = glob.glob(os.path.join(args.input_dir, "*.xyz"))
    print(f"--- Starting ORCA Workflow: {len(input_files)} molecules found ---")

    for input_xyz in input_files:
        basename = os.path.splitext(os.path.basename(input_xyz))[0]
        print(f"\nProcessing: {basename}")

        mol_dir = os.path.join(args.output_dir, basename)
        md_dir, frames_dir = os.path.join(mol_dir, "md_output"), os.path.join(mol_dir, "frames")
        os.makedirs(md_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)

        try:
            electron_count = calculate_total_electrons(input_xyz, args.charge)
            
            # 1. Run QM/MD
            md_inp_content = md_template.format(
                METHOD_SETTINGS=args.method_md, NPROCS=args.nprocs, BASENAME=basename,
                CHARGE=args.charge, MULTIPLICITY=args.mult, COORDINATES=read_xyz_coords(input_xyz)
            )
            md_inp_file = os.path.join(md_dir, f"{basename}_md.inp")
            with open(md_inp_file, 'w') as f: f.write(md_inp_content)
            
            print("  Running MD...")
            run_orca(args.orca_path, md_inp_file, os.path.join(md_dir, f"{basename}_md.out"), md_dir)

            # 2. Process Frames
            frames = parse_md_trajectory(os.path.join(md_dir, f"{basename}.md.xyz"))
            all_props = []
            
            for f_idx, f_coords in frames:
                if (f_idx - 1) % args.stride != 0: continue
                
                f_str = f"frame_{f_idx:04d}"
                f_dir = os.path.join(frames_dir, f_str)
                os.makedirs(f_dir, exist_ok=True)
                
                sp_name = f"{basename}_{f_str}"
                sp_inp_content = surface_template.format(
                    METHOD_SETTINGS=args.method_sp, NPROCS=args.nprocs, BASENAME=sp_name,
                    CHARGE=args.charge, MULTIPLICITY=args.mult, COORDINATES=f_coords,
                    GRID_POINTS=args.grid, HOMO_O=int((electron_count/2)-1), LUMO_O=int(electron_count/2)
                )
                
                sp_inp_file = os.path.join(f_dir, f"{sp_name}.inp")
                with open(sp_inp_file, 'w') as f: f.write(sp_inp_content)
                
                print(f"    Processing {f_str}...")
                run_orca(args.orca_path, sp_inp_file, os.path.join(f_dir, f"{f_str}.out"), f_dir)

                # Rename cubes
                for s in ["homo", "lumo", "density"]:
                    src = os.path.join(f_dir, f"{sp_name}_{s}.cube")
                    if os.path.exists(src): shutil.move(src, os.path.join(f_dir, f"{s}.cube"))
                
                props = parse_sp_output(os.path.join(f_dir, f"{f_str}.out"))
                props['frame'] = f_str
                all_props.append(props)

            with open(os.path.join(mol_dir, "properties.json"), 'w') as f:
                json.dump(all_props, f, indent=2)

        except Exception as e:
            print(f"  [!] Error processing {basename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ORCA MD & Surface Generation Workflow")
    parser.add_argument("--input_dir", required=True, help="Directory containing .xyz files")
    parser.add_argument("--output_dir", required=True, help="Directory for output data")
    parser.add_argument("--orca_path", default="orca", help="Path to ORCA binary")
    parser.add_argument("--md_template", default="orca_md_template.inp")
    parser.add_argument("--surf_template", default="orca_surface_template.inp")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--mult", type=int, default=1)
    parser.add_argument("--nprocs", type=int, default=1)
    parser.add_argument("--stride", type=int, default=10, help="Frame analysis stride")
    parser.add_argument("--grid", type=int, default=80, help="Cube grid resolution")
    parser.add_argument("--method_md", default="GFN2-XTB")
    parser.add_argument("--method_sp", default="B3LYP def2-SVP")
    
    args = parser.parse_args()
    main(args)
