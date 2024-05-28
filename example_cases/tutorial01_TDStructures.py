# # TDStructure
# Our main structure object is the **TDStructure** (Three D Structure). It is a powerful wrapper for an Openbabel OBMol object that also provides a connection to our in-house molecular 'graph' object and our electronic structure calculations

# Let's instantiate a molecule from a SMILES string.

from neb_dynamics.tdstructure import TDStructure

smi = "CC=CC"

td = TDStructure.from_smiles(smi)

# You can access by the OBMol object by calling `td.molecule_obmol`, and the graph object by caling `td.molecule_rp`

td.molecule_rp.draw()

td

# Another way to instantiate is from an XYZ file with `TDStructure.from_xyz(...)`, or from an array of coordiantes and symbols with `TDStructure.from_coords_symbols(...)`

# But the best thing we can do is compute energies and gradients using this object. 
# If we want to use **GFN2-XTB**, we can use `td.energy_xtb()` and `td.gradient_xtb()`

td.energy_xtb()

td.gradient_xtb()

# If we want to use **TeraChem** we need to specify the input arguments. The TDStructure object has attributes `.tc_model_method`, `.tc_model_basis`, and `.tc_kwds` that can be adjusted to the level of theory desired. By default it is populated with *b3lyp* and *6-31g* for no particular reason. 

# +
ene_default = td.energy_tc()

td_new = td.copy()
td_new.tc_model_method = 'b3lyp'
td_new.tc_model_basis = 'sto-3g'
ene_new = td_new.energy_tc()
# -

ene_default, ene_new

gradient_default = td.gradient_tc()
gradient_new = td_new.gradient_tc()

gradient_default, gradient_new

# If we want to run geometry optimizations, we can use the `.xtb_geom_optimization()` and `.tc_geom_optimization()` functions

td.xtb_geom_optimization()

td_new.tc_kwds = {'guess':'sad'}

td_new.tc_local_geom_optimization()

# Now you are perfectly prepared to do basic operations using a single TDStructure


