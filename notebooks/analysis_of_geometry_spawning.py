# +
from neb_dynamics.Chain import Chain
from neb_dynamics.NEB import NEB

from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.Inputs import NEBInputs, GIInputs, ChainInputs
from neb_dynamics.constants import BOHR_TO_ANGSTROMS
from neb_dynamics.Node3D import Node3D

from pathlib import Path

from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.TreeNode import TreeNode
from retropaths.abinitio.trajectory import Trajectory
from retropaths.molecules.smiles_tools import (
    bond_ord_number_to_string,
    from_number_to_element,
)
from IPython.core.display import HTML

import multiprocessing as mp
import numpy as np
from tqdm.notebook import tqdm

from ase.optimize import BFGS, LBFGS, FIRE
from ase import Atoms
from xtb.ase.calculator import XTB
import retropaths.helper_functions as hf

from retropaths.abinitio.tdstructure import TDStructure
from pathlib import Path
HTML('<script src="//d3js.org/d3.v3.min.js"></script>')

# -

reactions = hf.pload("/home/jdep/retropaths/data/reactions.p")

# # Do NM spawning

c = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Claisen-Rearrangement/initial_guess_msmep/")
# c = TreeNode.read_from_disk("/home/jdep/T3D_data/geometry_spawning/click_chem_results/click_chem_reaction_0-1")

leaves = c.ordered_leaves

opt_history = leaves[0].data.chain_trajectory

chain = opt_history[-1]
# chain = out

def get_input_tuple(td):
    return (td.atomn, td.charge, td.spinmult, td.symbols, td.coords)


def relax_geom(input_tuple):
    try:

        atomn, charge, spinmult, symbols, coords = input_tuple


        charges = np.zeros(atomn)
        charges[0] = charge

        spins = np.zeros(atomn)
        spins[0] = spinmult - 1

        atoms = Atoms(
            symbols=symbols.tolist(),
            positions=coords,
            charges=charges,
            magmoms=spins,
        )

        atoms.calc = XTB(method="GFN2-xTB", accuracy=0.001)
        opt = FIRE(atoms, logfile="/tmp/hi_dip.txt")
        # opt = /(atoms, logfile=None)
        opt.run(fmax=0.005)

        coords = atoms.get_positions()
        symbols = np.asarray([from_number_to_element(x) for x in atoms.numbers])

        output_tuple = coords,symbols, charge, spinmult
        return output_tuple

    except:
        return None


def spawn_geometry_xtb(chain=None, ts_guess=None, dr=3):
    assert chain or ts_guess, 'Need to give either a chain or a TS guess'
    if not ts_guess:
        ts_guess = chain.get_ts_guess()

    ts_guess.tc_model_basis = 'gfn2xtb'
    ts_guess.tc_model_method = 'gfn2xtb'
    ts_guess.tc_kwds = {}

    nmas = ts_guess.tc_nma_calculation()
    freqs = ts_guess.tc_freq_calculation()

    output_data = {
        'all_displaced': Trajectory([]),
        'nm_index': [],
        'freq':[],
        'ts_guess': ts_guess,
    }
    all_geoms = []
    for ind, _ in enumerate(nmas):
        
        nm = np.array(nmas[ind]).reshape(ts_guess.coords.shape)

        ts_displaced = ts_guess.copy()
        ts_displaced_plus = ts_displaced.update_coords(ts_displaced.coords + dr*nm)
        ts_displaced_minus = ts_displaced.update_coords(ts_displaced.coords - dr*nm)
            
        output_data['all_displaced'].traj.append(ts_displaced_plus)
        output_data['all_displaced'].traj.append(ts_displaced_minus)
        
    output_data['all_displaced'].update_tc_parameters(ts_guess)
    
    all_inputs = [get_input_tuple(td) for td in output_data['all_displaced']]
    with mp.Pool() as pool:
        geom_opt_output_tuples = list(tqdm(pool.imap(relax_geom, all_inputs), total=len(all_inputs)))
    
    output_tds = [td_from_output_tuple(ot) if ot else None for ot in geom_opt_output_tuples  ]
    
    converged_results = []
    for i, r in enumerate(output_tds):
        if r:
            converged_results.append(r)
            # output_data['nm_index'].append(i)
            output_data['nm_index'].append(int(i/2))
            output_data['freq'].append(freqs[int(i/2)])
            

    output_data['all_displaced_opt'] = Trajectory(converged_results)
            
    return output_data


def td_from_output_tuple(output_tuple):
    coords,symbols, charge, spinmult = output_tuple 
    td = TDStructure.from_coords_symbols(
            coords=coords,
            symbols=symbols,
            tot_charge=charge,
            tot_spinmult=spinmult,
        )
    return td


# +
# output_data = spawn_geometry_opts(chain, dr=3)
# output_data = spawn_geometry_xtb(chain, dr=.5)
# -

drs = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
# all_output_data = [output_data]
all_output_data = []
# for dr in drs:
for dr in drs:
    print(f"doing {dr}")
    new_output_data = spawn_geometry_xtb(chain, dr=dr)
    all_output_data.append(new_output_data)

odb.keys()


def extract_new_molecules(chain, output_data_object):
    t = output_data_object['all_displaced_opt']
    ref_traj = chain.to_trajectory()
    new_molecules = []
    new_molecule_freq = []
    displaced_mol = []
    for i, td in enumerate(t):
        if td.molecule_rp.is_bond_isomorphic_to(ref_traj[0].molecule_rp) or\
            td.molecule_rp.is_bond_isomorphic_to(ref_traj[-1].molecule_rp) or \
            any([td.molecule_rp.is_bond_isomorphic_to(m.molecule_rp) for m in new_molecules]):
            continue
        else:
            new_molecules.append(td)
            new_molecule_freq.append(output_data_object['freq'][i])
            # nma = output_data_object['ts_guess'].tc_nma_calculation()
            # new_molecule_nm.append(nma[int(i/2)])
            
            displaced_mol.append(output_data_object['all_displaced'][i])
            
    return new_molecules, new_molecule_freq, displaced_mol


n_of_new_molecules = [len(extract_new_molecules(chain, odb)[0]) for odb in all_output_data]

import matplotlib.pyplot as plt

plt.plot(drs, n_of_new_molecules, 'o-')

odb = all_output_data[4]
new_molecules, new_molecule_freqs, displaced_mol = extract_new_molecules(chain, odb)

all_molecules = [chain[0].tdstructure, chain[-1].tdstructure]+new_molecules

# t_out = Trajectory(new_molecules)
t_out = Trajectory(all_molecules)

plt.hist(t_out.energies_xtb())

new_molecule_freqs

new_molecules[0]

displaced_mol[0]

# +
min_energy_rel_to_start = []
min_td = []

for odb in all_output_data:
    new_molecules, new_molecule_freqs = extract_new_molecules(chain, odb)
    all_molecules = Trajectory([chain[0].tdstructure, chain[-1].tdstructure]+new_molecules)
    
    
    ens = all_molecules.energies_xtb()
    
    min_energy_rel_to_start.append(min(ens))
    min_td.append(all_molecules[np.argmin(ens)])

# -

min_energy_rel_to_start

exploring_more = spawn_geometry_xtb(ts_guess=min_td[-1])

stupid_chain = Chain([chain[0], Node3D(min_td[-1])], ChainInputs())
new_mols, new_mols_freqs = extract_new_molecules(stupid_chain, exploring_more)

foobar_t = Trajectory([min_td[-1]]+new_mols)

foobar_t.energies_xtb()

foobar_t[11]


