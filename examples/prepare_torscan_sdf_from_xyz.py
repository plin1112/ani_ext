# requires xyz2mol https://github.com/jensengroup/xyz2mol
# needs rdkit and networkx
import os
import xyz2mol
from rdkit import Chem

if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input XYZ filename')
    parser.add_argument('-o', '--output', required=True, help='Output SDF filename')
    parser.add_argument('-a', '--atoms', default='1 2 3 4', help='Atomic indexes for Torsional Angle')
    parser.add_argument('-t', '--torsion', default='S 12 30.0', help='Define Torsional Scan steps')
    args = parser.parse_args()

    f_xyz = args.input
    writer = Chem.SDWriter(args.output)

    atoms, charge, xyz_coordinates = xyz2mol.read_xyz_file(f_xyz)
    mols = xyz2mol.xyz2mol(atoms, xyz_coordinates, charge=charge)

    for mol in mols:
        mol.SetProp('TORSION_ATOMS', args.atoms)
        mol.SetProp('TORSION_ANGLE_SCAN', args.torsion)
        writer.write(mol)

