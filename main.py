#  -Defining the molecular structure

# Eliminate noisy python warnings
import warnings

warnings.filterwarnings("ignore")

# in Angstrom
h2o_structure_direct = [["H", [-0.02111417,0.8350417,1.47688078]],  # H stands for hydrogen element in water
                        ["O", [0.0, 0.0, 0.0]],                     # O stands for oxygen element in water
                        ["H", [-0.00201087,0.45191737,-0.27300254]]]

from paddle_quantum.qchem import geometry

h2o_structure_xyz = geometry(file="h2o.xyz")
assert h2o_structure_xyz == h2o_structure_direct

#Calculate Hartree Fock orbitals

from paddle_quantum.qchem import get_molecular_data

h2o_moledata = get_molecular_data(
    h2o_structure_direct,
    charge=0,                # Water molecule is charge neutral
    multiplicity=1,          # In the ground state, the lowest 5 molecular orbitals of water molecular will be occupied by a pair of electrons with opposite spin
    basis="sto-3g",
    method="scf",
    if_save=True,            # Whether to save information contained in MolecularData object to a hdf5 file
    if_print=True,           # Wheter to print the ground state energy of water molecule
    name="",                 # Specifies the name of the hdf5 file
    file_path="."            # Specifies where to store the hdf5 file          
)

from openfermion.chem import MolecularData

assert isinstance(h2o_moledata, MolecularData

#Molecular Hamiltonian in second quantization form
import numpy as np 
np.set_printoptions(precision=4, linewidth=150)

hpq, vpqrs = h2o_moledata.get_integrals()
assert np.shape(hpq)==(7, 7)             # When use sto3g basis, the total number of molecular orbitals used in water calculation is 7
assert np.shape(vpqrs)==(7, 7, 7, 7)

print(hpq)
# print(vpqrs)

from paddle_quantum.qchem import fermionic_hamiltonian

H_of_water = fermionic_hamiltonian(
    h2o_moledata,
    multiplicity=1,
    active_electrons=4,
    active_orbitals=4
)

from openfermion.ops import FermionOperator

assert isinstance(H_of_water, FermionOperator)

from paddle_quantum.qchem import active_space

core_orbits_list, act_orbits_list = active_space(
    10,                        # number of electrons in water molecule
    7,                         # number of molecular orbitals in water molecule
    active_electrons=4,
    active_orbitals=4
)

print("List of core orbitals: {:}".format(core_orbits_list))
print("List of active orbitals: {:}".format(act_orbits_list))

#From Fermionic Hamiltonian to spin Hamiltonian

from paddle_quantum.qchem import spin_hamiltonian

pauli_H_of_water_ = spin_hamiltonian(
    h2o_moledata,
    multiplicity=1,
    active_electrons=4,
    active_orbitals=4,
    mapping_method='jordan_wigner'
)

print('There are ', pauli_H_of_water_.n_terms, 'terms in H2O Hamiltonian in total.')
print('The first 10 terms are \n', pauli_H_of_water_[:10])