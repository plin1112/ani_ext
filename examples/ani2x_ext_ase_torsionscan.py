import torch
from math import pi
from rdkit import Chem
from rdkit.Geometry import Point3D
from ase.constraints import FixInternals
from ase.optimize import BFGS, LBFGS, FIRE
from ase import Atoms
from ase.io import read, write
from ase.units import Hartree, mol, kcal
from ase.io.trajectory import Trajectory
import torchani

from ani2x_ext.custom_emsemble_ani2x_ext import CustomEnsemble

au2kcalmol = Hartree * mol / kcal
ev2kcalmol = mol / kcal

if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input Initial SDF filename')
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
        # species = [at.GetSymbol() for at in mol.GetAtoms()]
        numbers = [at.GetAtomicNum() for at in mol.GetAtoms()]
        coords = mol.GetConformer().GetPositions()
        has_init_tor_val = False

        if mol.HasProp('TORSION_ATOMS'):
            tor_atoms = mol.GetProp('TORSION_ATOMS')
        else:
            sys.exit('torsional angle not defined!')
        if mol.HasProp('TORSION_ANGLE_SCAN'):
            tor_val_info = mol.GetProp('TORSION_ANGLE_SCAN')
            cols = tor_val_info.split()
            if cols[0] == 'S' and len(cols) == 3:
                n_step = int(cols[1])
                step_size = float(cols[2])
            elif cols[1] == 'S' and len(cols) == 4:
                init_tor_val = float(cols[0])
                n_step = int(cols[2])
                step_size = float(cols[3])
                has_init_tor_val = True

        # system = Atoms(symbols=species, positions=coords)
        system = Atoms(numbers=numbers, positions=coords)
        system.set_calculator(calculator)

        # Fix this dihedral angle to whatever it was from the start
        indices = [int(x) -1 for x in tor_atoms.split()]
        if has_init_tor_val:
            dihedral0 = init_tor_val
        else:
            dihedral0 = system.get_dihedral(*indices)

        for i in range(n_step):
            print('dihedral before', dihedral0)
            constraint = FixInternals(
                          dihedrals_deg=[(dihedral0, indices)],
                          epsilon=1e-10)

            system.set_constraint(constraint)
            print('-----Optimization-----')
            opt = LBFGS(system, logfile=args.log) # , trajectory='opt.traj')
            try:
                opt.run(fmax=0.01)
                print(f'Final Energy: {system.get_potential_energy() * ev2kcalmol} kcal/mol')
                new_dihedral = system.get_dihedral(*indices)
                print(f'dihedral after: {new_dihedral}, error in dihedral: {new_dihedral-dihedral0}')
                conf = mol.GetConformer()
                for j in range(mol.GetNumAtoms()):
                    x,y,z = system.positions[j]
                    conf.SetAtomPosition(j, Point3D(x,y,z))
                mol.setProp('TORSION_ANGLE_SCAN', str(new_dihedral))
                mol.SetProp('ANI2x_Ext Energy', str(system.get_potential_energy() * ev2kcalmol))
            except:
                print('optimization failed')
                new_dihedral = system.get_dihedral(*indices)
                mol.setProp('TORSION_ANGLE_SCAN', str(new_dihedral))
                mol.SetProp('ANI2x_Ext Energy', 'opt failed: ' + str(system.get_potential_energy() * ev2kcalmol))

            system.set_constraint()
            dihedral0 += step_size
            system.set_dihedral(*indices, dihedral0)

            writer.write(mol)


