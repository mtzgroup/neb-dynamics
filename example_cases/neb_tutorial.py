# # Using MSMEP

from neb_dynamics import QCOPEngine, ASEEngine, MSMEP, StructureNode, NEBInputs
import neb_dynamics.chainhelpers as ch
from xtb.ase.calculator import XTB
from neb_dynamics.nodes.nodehelpers import create_pairs_from_smiles

# +

# this is a very challenging pair. will take a long time. 
# smi1 = 'C=O.C=[P+](c1ccccc1)(c2ccccc2)c3ccccc3'
# smi2 = 'C=C.O=P(c1ccccc1)(c2ccccc2)c3ccccc3'

# this is an SN2 reaction
smi1 = 'CBr.[OH-]'
smi2 = 'CO.[Br-]'
start, end = create_pairs_from_smiles(smi1, smi2) # uses IBM RXNMapper to get an atom-atom index map.
# -

calc = XTB(method="GFN2-xTB", solvent="water")
eng = ASEEngine(calculator=calc)
nbi = NEBInputs(v=True)
m = MSMEP(engine=eng, path_min_method='neb')

start_opt_traj = eng.compute_geometry_optimization(StructureNode(structure=start))
start_node = start_opt_traj[-1]
end_opt_traj = eng.compute_geometry_optimization(StructureNode(structure=end))
end_node = end_opt_traj[-1]

from qcio import view

# initial_chain = ch.run_geodesic([start_node, end_node], nimages=15)
initial_chain = ch.run_geodesic([start_opt_traj[0], end_opt_traj[0]], nimages=20)

eng.compute_energies(initial_chain)

path_min_obj, elem_step_result = m.run_minimize_chain(initial_chain)

# +
# ch.visualize_chain(path_min_obj.optimized)
# -

history = m.run_recursive_minimize(initial_chain)

ch.visualize_chain(history.output_chain)


