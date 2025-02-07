from qcio import Structure, view
from pathlib import Path
from neb_dynamics import StructureNode, RunInputs
from neb_dynamics.inputs import NetworkInputs, ChainInputs
from neb_dynamics.TreeNode import TreeNode
import neb_dynamics.chainhelpers as ch
from qcio import ProgramOutput
from neb_dynamics.pot import Pot
from neb_dynamics.molecule import Molecule
from IPython.core.display import HTML
import json
HTML('<script src="//d3js.org/d3.v3.min.js"></script>')

from neb_dynamics.NetworkBuilder import NetworkBuilder

node = StructureNode(structure=Structure.from_smiles("C"))
# dd = Path("/home/jdep/T3D_data/msmep_draft/comparisons_dft/results_manual/Beckmann-Rearrangement/network/")
dd = Path("/u/jdep/T3D/")
nb = NetworkBuilder(data_dir=dd, 
                   start=None, end=None, 
                    network_inputs=NetworkInputs(verbose=True), 
                    chain_inputs=ChainInputs(node_ene_thre=5, node_rms_thre=5))

nb.msmep_data_dir = dd

pot = nb.create_rxn_network(file_pattern="*")

dump = pot.model_dump()

# Save the dictionary to a JSON file
with open("data.json", "w+") as f:
    json.dump(dump, f)

loaded = json.load(open("data.json"))
pot2 = Pot.from_dict(loaded)
# print(pot2.graph.edges[(0,1)]['list_of_nebs'][0]['nodes'][0])
for node in pot2.graph.nodes:
    assert isinstance(pot2.graph.nodes[node]['molecule'], Molecule)

