from hf_backbone import Molecule
from cuhf import CUHFMolecule
import psi4
psi4.set_options({"basis":"sto-3g", "scf_type":"pk", "reference":"uhf", "e_convergence":"1e-6", "d_convergence":"1e-12"})
h2o = CUHFMolecule("""
0 1
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")
h2o.setConvergence(1e-12)
print(h2o.iterator(mute=True, criterion="density"))
print(psi4.energy("scf"))