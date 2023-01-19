# -*- coding: utf-8 -*-
"""
Structure minimization and constant temperature MD using ASE interface
======================================================================

This example is modified from the official `home page` and
`Constant temperature MD`_ to use the ASE interface of TorchANI as energy
calculator.

.. _home page:
    https://wiki.fysik.dtu.dk/ase/
.. _Constant temperature MD:
    https://wiki.fysik.dtu.dk/ase/tutorials/md/md.html#constant-temperature-md
"""


###############################################################################
# To begin with, let's first import the modules we will use:
# from ase.lattice.cubic import Diamond
from ase.build import molecule
from ase.md.langevin import Langevin
from ase.optimize import BFGS
from ase import units
import torchani
import torch
from ani2x_ext.custom_emsemble_ani2x_ext import CustomEnsemble

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
my_model = CustomEnsemble(model_choice=0).to(device)

###############################################################################
# Now let's set up a crystal
# atoms = Diamond(symbol="C", pbc=True)
# print(len(atoms), "atoms with periodic boundary")
atoms = molecule('CH3CHO')

###############################################################################
# Now let's create a calculator from builtin models:
# calculator = torchani.models.ANI1ccx().ase()
calculator = my_model.ase()

###############################################################################
# .. note::
#     Regardless of the dtype you use in your model, when converting it to ASE
#     calculator, it always automatically the dtype to ``torch.float64``. The
#     reason for this behavior is, at many cases, the rounding error is too
#     large for structure minimization. If you insist on using
#     ``torch.float32``, do the following instead:
#
#     .. code-block:: python
#
#         calculator = torchani.models.ANI1ccx().ase(dtype=torch.float32)

###############################################################################
# Now let's set the calculator for ``atoms``:
atoms.set_calculator(calculator)

###############################################################################
# Now let's minimize the structure:
print(atoms.get_potential_energy())
print("Begin minimizing...")
opt = BFGS(atoms, trajectory='opt1.traj')
opt.run(fmax=0.001)
print()


###############################################################################
# Now create a callback function that print interesting physical quantities:
def printenergy(a=atoms):
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))


###############################################################################
# We want to run MD with constant energy using the Langevin algorithm
# with a time step of 1 fs, the temperature 300K and the friction
# coefficient to 0.02 atomic units.
dyn = Langevin(atoms, 1 * units.fs, temperature_K=300., friction=0.2, trajectory='moldyn1.traj')
dyn.attach(printenergy, interval=50)

###############################################################################
# Now run the dynamics:
print("Beginning dynamics...")
printenergy()
dyn.run(500)
