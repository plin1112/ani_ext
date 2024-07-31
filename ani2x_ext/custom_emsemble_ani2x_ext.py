import torch
import torchani
from torch import Tensor
from typing import Tuple, Optional
from torchani.aev import AEVComputer
from torchani.nn import SpeciesConverter, SpeciesEnergies, ANIModel
from torchani.utils import EnergyShifter
import copy
import os, sys
from torchani import ase

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Borh length adopted from 2014 CODATA
Bohr2Ang = 0.52917721067
Ang2Bohr = 1. / Bohr2Ang
# Hartree to kcal/mol is taken from ASE
Au2KcalMol = 627.5094738898777

# repulsion parameters for H,C,N,O,F,S,Cl taken from GFN2-xTB paper DOI: 10.1021/acs.jctc.8b01176
v_yeff = {'H':1.105388, 'C':4.231078, 'N':5.242592, 'O':5.784415, 'F':7.021486, 'S':14.995090, 'Cl':17.353134}
v_alpha = {'H':2.213717, 'C':1.247655, 'N':1.682689, 'O':2.165712, 'F':2.421394, 'S':1.214553, 'Cl':1.577144}

def atomic_net(layers_dims):
    # for constructing atomic network
    assert len(layers_dims) == 5
    return torch.nn.Sequential(
        torch.nn.Linear(layers_dims[0], layers_dims[1], bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(layers_dims[1], layers_dims[2], bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(layers_dims[2], layers_dims[3], bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(layers_dims[3], layers_dims[4], bias=False)
    )

# parameters for ANI2x
aev_params = {
        'Rcr' : 5.1000e+00,
        'Rca' : 3.5000e+00,
        'EtaR' : [1.9700000e+01],
        'ShfR' : [8.0000000e-01, 1.0687500e+00, 1.3375000e+00, 1.6062500e+00, 1.8750000e+00,
                  2.1437500e+00, 2.4125000e+00, 2.6812500e+00, 2.9500000e+00, 3.2187500e+00,
                  3.4875000e+00, 3.7562500e+00, 4.0250000e+00, 4.2937500e+00, 4.5625000e+00,
                  4.8312500e+00],
        'Zeta' : [1.4100000e+01],
        'ShfZ' : [3.9269908e-01, 1.1780972e+00, 1.9634954e+00, 2.7488936e+00],
        'EtaA' : [1.2500000e+01],
        'ShfA' : [8.0000000e-01, 1.1375000e+00, 1.4750000e+00, 1.8125000e+00, 2.1500000e+00,
                  2.4875000e+00, 2.8250000e+00, 3.1625000e+00]
        }

def exp_cutoff(d: Tensor, rc: Tensor) -> Tensor:
    # replcing in aev def cutoff_cosine(distances: Tensor, cutoff: float) -> Tensor:
    fc = torch.exp(-1.0 / (1.0 - (d/rc).clamp(0, 1.0-1e-4).pow(2))) / 0.36787944117144233
    return fc

torchani.aev.cutoff_cosine = exp_cutoff

def distance_matrix(coordinates, pair_padding_mask, eps=1e-16):
    """
    Calculate distance matrix and replace padding by eps.
    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3

    Arguments:
        coordinates (torch.Tensor): (M, N, 3)
        eps (float): to avoid devision by zeros
    Returns:
        distances (torch.Tensor): (M, N, N)
    """
    assert coordinates.dim() == 3
    M, N, _ = coordinates.shape

    eye = torch.eye(N).to(coordinates.device)
    eps_shift = eps * (~ pair_padding_mask.to(torch.bool) + eye)
    # (M, N, N) square tensor of (xi - xj)
    diff = coordinates.unsqueeze(1) - coordinates.unsqueeze(2)
    # eps_shift added to avoid nans when distances used in denomerator
    distances = torch.sqrt(torch.sum(diff * diff, -1) + eps_shift)
    return distances

class CustomEnsemble(torch.nn.Module):
    """Use the picked model to Compute the average output of an ensemble of modules."""
    def __init__(self, periodic_table_index=True, return_option=0, model_choice=0, device=None):
        super(CustomEnsemble, self).__init__()
        self.periodic_table_index = periodic_table_index
        self.model_choice = model_choice
        self.return_option = return_option
        if return_option > 2:
            sys.exit("unknown return option for CustomEnsemble.")
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # get model and ensemble information
        # including pt_file_list and energy_shifter
        self.get_model_info()
        self.size = len(self.pt_file_list)

        # Common to ANI2x
        self.species=['H','C','N','O','F','S','Cl']
        self.num_species = len(self.species)
        # AEV parameters
        self.Rcr = aev_params['Rcr']
        self.Rca = aev_params['Rca']
        EtaR = torch.tensor(aev_params['EtaR'], device=self.device)
        ShfR = torch.tensor(aev_params['ShfR'], device=self.device)
        Zeta = torch.tensor(aev_params['Zeta'], device=self.device)
        ShfZ = torch.tensor(aev_params['ShfZ'], device=self.device)
        EtaA = torch.tensor(aev_params['EtaA'], device=self.device)
        ShfA = torch.tensor(aev_params['ShfA'], device=self.device)

        try:
            self.aev_computer = AEVComputer(self.Rcr, self.Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, self.num_species, cutoff_fn='smooth').to(self.device)
        except:
            self.aev_computer = AEVComputer(self.Rcr, self.Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, self.num_species).to(self.device)

        self.number2tensor = torchani.nn.SpeciesConverter(self.species)
        self.species_to_tensor = torchani.utils.ChemicalSymbolsToInts(self.species)

        # Set up NN models 
        aev_dim = self.aev_computer.aev_length
        if model_choice == 5:
            anets = ANIModel(
                [atomic_net([aev_dim, 256, 192, 160, 1]),  # H network
                 atomic_net([aev_dim, 256, 192, 160, 1]),  # C network
                 atomic_net([aev_dim, 192, 160, 128, 1]),  # N network
                 atomic_net([aev_dim, 192, 160, 128, 1]),  # O network
                 atomic_net([aev_dim, 160, 128, 96, 1]),  # F network
                 atomic_net([aev_dim, 160, 128, 96, 1]),  # S network
                 atomic_net([aev_dim, 160, 128, 96, 1])]  # Cl network
            )
        else:
            anets = ANIModel(
                [atomic_net([aev_dim, 256, 192, 160, 1]),  # H network
                 atomic_net([aev_dim, 224, 192, 160, 1]),  # C network
                 atomic_net([aev_dim, 192, 160, 128, 1]),  # N network
                 atomic_net([aev_dim, 192, 160, 128, 1]),  # O network
                 atomic_net([aev_dim, 160, 128, 96, 1]),  # F network
                 atomic_net([aev_dim, 160, 128, 96, 1]),  # S network
                 atomic_net([aev_dim, 160, 128, 96, 1])]  # Cl network
            )

        # Set up model ensemble
        self.models = torch.nn.ModuleList()
        self.models.append(anets)
        for i in range(1, self.size):
            self.models.append(copy.deepcopy(anets))

        for i, pt_file in enumerate(self.pt_file_list):
            if os.path.isfile(pt_file):
                checkpoint = torch.load(pt_file, map_location=self.device)
                self.models[i].load_state_dict(checkpoint['nn'])
                self.models[i].eval()

        for p in self.models.parameters():
            p.requires_grad_(False)

        # parameters for short-range repulsive potential calculations, common to all models
        Vyeff = torch.tensor([v_yeff[x] for x in self.species], dtype=torch.float, device=self.device)
        Valpha = torch.tensor([v_alpha[x] for x in self.species], dtype=torch.float, device=self.device)
        ## matrix of combinations of parameters (pre-calculates constants)
        Myeff = torch.outer(Vyeff, Vyeff)
        Malpha = -torch.sqrt(torch.outer(Valpha, Valpha))
        # parameters k_rep is 1 for H-H and 1.5 for everything else
        Mkrep = torch.ones(self.num_species, self.num_species).to(self.device)
        Mkrep[:,:] = 1.5
        Mkrep[0, 0] = 1.0

        self.register_buffer('Myeff', Myeff)
        self.register_buffer('Malpha', Malpha)
        self.register_buffer('Mkrep', Mkrep)

    def get_model_info(self):
        # set up models based on the model_choice
        nn_root = os.path.dirname(os.path.abspath(__file__))
        if self.model_choice == 0 or self.model_choice == "ani2x_n1c":
            nnpath = os.path.join(nn_root, '/ANI2x_n1c_models/')
            if not os.path.exists(nnpath): sys.exit('nnpath does not exist!')
            self.pt_file_list = [nnpath+'ani2x_gelu_expc_nc1e1_best_nn.pt',
                                 nnpath+'ani2x_gelu_expc_nc1e2_best_nn.pt',
                                 nnpath+'ani2x_gelu_expc_nc1e3_best_nn.pt',
                                 nnpath+'ani2x_gelu_expc_nc1e4_best_nn.pt',
                                 nnpath+'ani2x_gelu_expc_nc1e5_best_nn.pt',
                                 nnpath+'ani2x_gelu_expc_nc1e6_best_nn.pt',
                                 nnpath+'ani2x_gelu_expc_nc1e7_best_nn.pt',
                                 nnpath+'ani2x_gelu_expc_nc1e8_best_nn.pt']
            self.energy_shifter = EnergyShifter(
                    [-0.49932123271, -37.8338334397, -54.5732824628, -75.0424519384, \
                     -99.6949007172, -398.081416857, -460.116700576]).to(self.device)

        elif self.model_choice == 1 or self.model_choice == "r2s_o0":
            nnpath = os.path.join(nn_root, '/r2s_o0_models/')
            if not os.path.exists(nnpath): sys.exit('nnpath does not exist!')
            self.pt_file_list = [nnpath+'ani2x_gelu_expc_r2s_o0_1_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0_2_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0_3_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0_4_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0_5_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0_6_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0_7_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0_8_best.pt']
            self.energy_shifter = EnergyShifter(
                    [-0.497271685670, -37.832225901872, -54.581004402346, -75.057311846055, \
                     -99.726350798961, -398.080971275720, -460.113993263966])

        elif self.model_choice == 2 or self.model_choice == "r2s_o0water":
            nnpath = os.path.join(nn_root, '/r2s_o0water_models/')
            if not os.path.exists(nnpath): sys.exit('nnpath does not exist!')
            self.pt_file_list = [nnpath+'ani2x_gelu_expc_r2s_o0water_1_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0water_2_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0water_3_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0water_4_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0water_5_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0water_6_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0water_7_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0water_8_best.pt']
            self.energy_shifter = EnergyShifter(
                    [-0.494931329259, -37.822388062823, -54.581010824825, -75.059169500763, \
                     -99.724273365141, -398.082828534447, -460.113806300624])

        elif self.model_choice == 3 or self.model_choice == "r2s_o0chcl3":
            nnpath = os.path.join(nn_root, '/r2s_o0chcl3_models/')
            if not os.path.exists(nnpath): sys.exit('nnpath does not exist!')
            self.pt_file_list = [nnpath+'ani2x_gelu_expc_r2s_o0chcl3_1_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0chcl3_2_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0chcl3_3_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0chcl3_4_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0chcl3_5_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0chcl3_6_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0chcl3_7_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0chcl3_8_best.pt']
            self.energy_shifter = EnergyShifter(
                    [-0.496899744403, -37.824548433511, -54.576681029080, -75.056821997619, \
                     -99.726146486046, -398.085456915563, -460.116926115444])

        elif self.model_choice == 4 or self.model_choice == "r2s_o0ch3cn":
            nnpath = os.path.join(nn_root, '/r2s_o0ch3cn_models/')
            if not os.path.exists(nnpath): sys.exit('nnpath does not exist!')
            self.pt_file_list = [nnpath+'ani2x_gelu_expc_r2s_o0ch3cn_1_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0ch3cn_2_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0ch3cn_3_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0ch3cn_4_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0ch3cn_5_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0ch3cn_6_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0ch3cn_7_best.pt',
                                 nnpath+'ani2x_gelu_expc_r2s_o0ch3cn_8_best.pt']
            self.energy_shifter = EnergyShifter(
                    [-0.496684906369, -37.824424218755, -54.576572487630, -75.058406925318, \
                     -99.725926489187, -398.084853327694, -460.116392553071])

        elif self.model_choice == 5 or self.model_choice == "b973c_anid":
            nnpath = os.path.join(nn_root, '/b973c_Ignacio/')
            if not os.path.exists(nnpath): sys.exit('nnpath does not exist!')
            self.pt_file_list = [nnpath+'anid_model_0.pt',
                                 nnpath+'anid_model_1.pt',
                                 nnpath+'anid_model_2.pt',
                                 nnpath+'anid_model_3.pt',
                                 nnpath+'anid_model_4.pt',
                                 nnpath+'anid_model_5.pt',
                                 nnpath+'anid_model_6.pt']
            self.energy_shifter = EnergyShifter(
                    [-0.5069, -37.8144, -54.5565, -75.0292, -398.0432, -99.6886, -460.0822])

        else:
            sys.exit("unknown model_choice for the CustomEnsemble.")

    def ase(self, **kwargs):
        """Get an ASE Calculator using this ANI model

        Arguments:
            kwargs: ase.Calculator kwargs

        Returns:
            calculator (:class:`int`): A calculator to be used with ASE
        """
        from torchani import ase
        return ase.Calculator(self.species, self, **kwargs)

    def forward(self, species_coordinates: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None): 
        """Calculates predicted properties for minibatch of configurations

        Args:
            species_coordinates: minibatch of configurations; species can be in the form of atomic number or converted intergers
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled

        Returns:
            depends on the output option:
            0: species_energies: energies for the given configurations
            1: detached species, energies, QBC
            2: species, energies for all models in the ensemble

        .. note:: The coordinates, and cell are in Angstrom, and the energies
            will be in Hartree.
        """
        if self.periodic_table_index:
            species_coordinates = self.number2tensor(species_coordinates)

        # check if unknown species are included
        if species_coordinates[0].ge(self.aev_computer.num_species).any():
            raise ValueError(f'Unknown species found in {species_coordinates[0]}')

        # calculate short-range repulsive energy
        if cell is None and pbc is None:
            pass
        else:
            sys.exit('PBC system not tested yet')

        species, coordinates = species_coordinates
        n_atoms = species.shape[-1]
        # calculate short-range repulsive energy
        mask = species.ne(-1)
        padding_mask = mask.float()
        pair_padding_mask = padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2)
        dmat = distance_matrix(coordinates, pair_padding_mask) # / a0  # Convert to A.U.
        # dmat = torch.cdist(coordinates, coordinates)
        # for older version of pytorch cdist may not work
        # search for fast_cdist code online in this case
        # dmat = fast_cdist(coordinates, coordinates)
        n_mols = species.shape[0]
        pairs_all = torch.triu_indices(n_atoms, n_atoms, 1)
        dist12_pair_mask = pair_padding_mask[:, pairs_all[0], pairs_all[1]] 
        dist12 = dmat[:, pairs_all[0], pairs_all[1]] * Ang2Bohr
        species12 = species[:, pairs_all]
        krep = self.Mkrep[species12[:,0], species12[:,1]]
        yeff = self.Myeff[species12[:,0], species12[:,1]]
        alpha = self.Malpha[species12[:,0], species12[:,1]]
        rep_eng12 = (yeff / dist12) * torch.exp(alpha * (dist12 ** krep))
        rep_eng12 = rep_eng12 * dist12_pair_mask
        rep_eng12 = rep_eng12.sum(-1)

        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)

        energies = torch.stack([x(species_aevs)[1] for x in self.models])
        species, energies = self.energy_shifter(SpeciesEnergies(species, energies))
        energies += rep_eng12

        if self.return_option == 0:
            return SpeciesEnergies(species, energies.mean(0))
        elif self.return_option == 1:
            e = energies.detach()
            return species, energies.mean(0), e.std(0)
        elif self.return_option == 2:
            return species, energies




