#!/usr/bin/env python3
"""eOn ext_pot wrapper for PET-MAD via metatomic ASE calculator."""
import os
import numpy as np
from ase import Atoms
from metatomic.torch.ase_calculator import MetatomicCalculator

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pet-mad-xs-v1.5.0-rc1.pt")
calc = MetatomicCalculator(MODEL, extensions_directory=None)

lines = open("from_eon_to_extpot").readlines()
box = np.array([[float(v) for v in l.split()] for l in lines[:3]])
numbers = []
positions = []
for l in lines[3:]:
    tok = l.split()
    numbers.append(int(tok[0]))
    positions.append([float(tok[1]), float(tok[2]), float(tok[3])])

system = Atoms(numbers=numbers, positions=positions, cell=box, pbc=True)
system.calc = calc

energy = system.get_potential_energy()
forces = system.get_forces()

with open("from_extpot_to_eon", "w") as f:
    f.write(f"{energy:.15f}\n")
    for fx, fy, fz in forces:
        f.write(f"{fx:.15f} {fy:.15f} {fz:.15f}\n")
