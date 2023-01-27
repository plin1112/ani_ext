import torch
import numpy as np
import math
import torchani
from torchani.units import hartree2kcalmol
from ani2x_ext.custom_emsemble_ani2x_ext import CustomEnsemble

class xyzfile(object):
    def __init__(self, filename):
        self.fhandle = open(filename, 'r')
        self.length = int(self.fhandle.readline())
        self.atoms = []
        self.coordinates = []
        self.load_next_frame(is_first = True)

    def load_next_frame(self, is_first = False):
        exists = self.fhandle.readline()
        if len(exists) == 0:
            return False
        if not is_first:
            self.fhandle.readline()

        self.atoms = []
        self.coordinates = []
        for i in range(self.length):
            atom,x,y,z = self.fhandle.readline().strip().split()
            self.atoms.append(atom)
            self.coordinates.append([float(x), float(y), float(z)])
        return True

if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input XYZ filename')
    parser.add_argument('-o', '--output', required=True, help='Output logfile filename')
    args = parser.parse_args()

    f_xyz = args.input
    f_log = open(args.output, 'w+')
    f_log.write('id, Model_Energy, Model_Stds\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    my_model = CustomEnsemble(model_choice=0, periodic_table_index=False, return_option=1).to(device)

    max_conf = 2000

    x = xyzfile(f_xyz)
    coordinates = []
    coordinates.append(x.coordinates)
    species = x.atoms
    while x.load_next_frame():
        coordinates.append(x.coordinates)

    print(species)
    print(len(coordinates))

    natom = len(species)
    species_tensor = my_model.species_to_tensor(species).to(device).unsqueeze(0)

    total_coordinates = len(coordinates)

    energies = []
    ene_stds = []

    if total_coordinates > 0:
        max_round = total_coordinates // max_conf
        left_conf = total_coordinates % max_conf
        for i_round in range(max_round):
            idx_start = i_round * max_conf 
            idx_end = (i_round + 1) * max_conf
            i_species_tensor = species_tensor.expand(max_conf, -1)
            i_coords = torch.tensor(coordinates[idx_start:idx_end], requires_grad=True, device=device)
            _, enes, ene_std = my_model((i_species_tensor, i_coords))
            # forces = -torch.autograd.grad(ani2x_enes.sum(), i_coords, create_graph=True, retain_graph=True)[0]
            # hessian = torchani.utils.hessian(i_coords, forces=forces)
            enes = hartree2kcalmol(enes) 
            ene_std = hartree2kcalmol(ene_std)
            energies.extend(enes.tolist())
            ene_stds.extend(ene_std.tolist())
            # all_forces.extend(torch.norm(forces, dim=-1).tolist())
            # all_hessian.extend(torch.norm(hessian, dim=-1).tolist())
        # doing the left_round
        if left_conf != 0:
            idx_start = max_round * max_conf
            idx_end = total_coordinates
            species_tensor = my_model.species_to_tensor(species).to(device).unsqueeze(0)
            i_species_tensor = species_tensor.expand(left_conf, -1)
            i_coords = torch.tensor(coordinates[idx_start:idx_end], requires_grad=True, device=device)
            _, enes, ene_std = my_model((i_species_tensor, i_coords))
            # forces = -torch.autograd.grad(ani2x_enes.sum(), i_coords, create_graph=True, retain_graph=True)[0]
            # hessian = torchani.utils.hessian(i_coords, forces=forces)
            enes = hartree2kcalmol(enes)
            ene_std = hartree2kcalmol(ene_std)
            energies.extend(enes.tolist())
            ene_stds.extend(ene_std.tolist())
            # all_forces.extend(torch.norm(forces, dim=-1).tolist())
            # all_hessian.extend(torch.norm(hessian, dim=-1).tolist())

    for i in range(total_coordinates): 
        f_log.write('%d, %f, %f\n' % (i, energies[i], ene_stds[i])) # all_forces[i], all_hessian[i]))

    f_log.close()


