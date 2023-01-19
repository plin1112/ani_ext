import torch
import numpy as np
import math
from math import pi
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Geometry import Point3D
from ase.constraints import FixInternals
from ase.optimize import LBFGS, FIRE
from ase import Atoms
from ase.io import read, write
from ase.units import Hartree, mol, kcal
from ase.io.trajectory import Trajectory
import torchani
import string

from ani2x_ext.custom_emsemble_ani2x_ext import CustomEnsemble

def write_xyz(file_name, species, coords, title="title"):
    with open(file_name, "a") as file:
        file.write(str(len(species))+"\n")
        file.write(title+"\n")
        for symbol, xyz in zip(species, coords):
            file.write('%-4s %15.8f %15.8f %15.8f \n' % (symbol, xyz[0], xyz[1], xyz[2]))
    return 0

au2kcalmol = Hartree * mol / kcal
ev2kcalmol = mol / kcal

if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input SDF filename')
    parser.add_argument('-o', '--output', required=True, help='Output SDF filename')
    parser.add_argument('-l', '--log', default='job.log', help='Information Log filename')
    args = parser.parse_args()

    device = 'cuda'
    my_model = CustomEnsemble(model_choice=1).to(device)

    # calculator = torchani.models.ANI2x().ase()
    calculator = my_model.ase()

    f_input = args.input
    suppl = Chem.SDMolSupplier(f_input, removeHs=False)
    writer = Chem.SDWriter(args.output)

    for i, mol in enumerate(suppl):
        species = [at.GetSymbol() for at in mol.GetAtoms()]
        # numbers = [at.GetAtomicNum() for at in mol.GetAtoms()]
        coords = mol.GetConformer().GetPositions()
        formula = CalcMolFormula(mol)

        if mol.HasProp('TORSION_ATOMS_FRAGMENT') and mol.HasProp('TORSION_ANGLE'):
            tor_atoms = mol.GetProp('TORSION_ATOMS_FRAGMENT')
            tor_val = mol.GetProp('TORSION_ANGLE')
        else:
            print('torsional angle not defined!')
            exit()
        # charge = Chem.GetFormalCharge(mol)
        # valence = Chem.Descriptors.NumValenceElectrons(mol)
        system = Atoms(symbols=species, positions=coords)
        # for atom_type, xyz in zip(numbers, coords):
        #     system.append(atom_type)
        #     system.positions[-1] = np.array(xyz)
        print(system)

        # Fix this dihedral angle to whatever it was from the start
        indices = [int(x) -1 for x in tor_atoms.split()]
        dihedral1 = system.get_dihedral(*indices)
        constraint = FixInternals(
                          dihedrals=[(dihedral1 * pi / 180, indices)],
                          epsilon=1e-10)

        system.set_calculator(calculator)
        system.set_constraint(constraint)
        opt = LBFGS(system, logfile=args.log) # , trajectory='opt.traj')
        previous_dihedral = system.get_dihedral(*indices)
        print('dihedral before', previous_dihedral)
        print('-----Optimization-----')
        try:
            opt.run(fmax=0.01)
            info_txt = f'Final Energy: {system.get_potential_energy() * ev2kcalmol} kcal/mol'
            print(info_txt)
            # fname = formula + '_' + str(int(i / 24)) + '.xyz'
            # write_xyz(fname, species, system.positions, title=info_txt)
            new_dihedral = system.get_dihedral(*indices)
            print('dihedral after', new_dihedral)
            err = new_dihedral - previous_dihedral
            print('error in dihedral', repr(err))
            conf = mol.GetConformer()
            for j in range(mol.GetNumAtoms()):
                x,y,z = system.positions[j]
                conf.SetAtomPosition(j, Point3D(x,y,z))
            mol.SetProp('ANI2x_Ext Energy', str(system.get_potential_energy() * ev2kcalmol))
        except:
            print('optimization failed')
            mol.SetProp('ANI2x_Ext Energy', 'opt failed: ' + str(system.get_potential_energy() * ev2kcalmol))

        writer.write(mol)

        # if i > 24:
        #     exit()

