# +
from pathlib import Path
from neb_dynamics.CompetitorAnalyzer import CompetitorAnalyzer
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs, GIInputs, NEBInputs
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.NEB import NEB
from neb_dynamics.Node3D_TC import Node3D_TC
from neb_dynamics.TreeNode import TreeNode
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.abinitio.trajectory import Trajectory
import numpy as np

from chemcloud import CCClient
# import os
# del os.environ['OE_LICENSE']
# -

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import warnings

warnings.filterwarnings('ignore')

# +
import retropaths.helper_functions as hf

reactions = hf.pload("../../retropaths/data/reactions.p")
# -

# # subselect RP database

all_rns = [r for r in reactions]


def n_fragments(reaction_object):
    return len(reaction_object.reactants.separate_graph_in_pieces())


uni_molecular_rns = [r for r in all_rns if n_fragments(reactions[r]) == 1]

# # make structures

directory = Path("/home/jdep/T3D_data/msmep_draft/comparisons_dft/")
ca = CompetitorAnalyzer(comparisons_dir=directory,method='dlfind')

# rns = ca.available_reaction_names
rns = uni_molecular_rns

m = MSMEP(neb_inputs=NEBInputs(), gi_inputs=GIInputs(), chain_inputs=ChainInputs())

frics = [0.001, 0.01, 0.1, 1]

# +
all_structures = []
for rn in rns:
    print(rn)

    rn_dir = ca.structures_dir / rn
    rn_dir.mkdir(exist_ok=True)
    
    start_fp = rn_dir  / "start.xyz"
    end_fp = rn_dir / "end.xyz"
    
    
#     rxn = reactions[rn]
#     try:
#         r, p = m.create_endpoints_from_rxn_name(rn, reactions)
#         if r.molecule_rp.is_bond_isomorphic_to(p.molecule_rp):
#             print(f"Redo: {rn}")
#         else:
#             r_opt = r.xtb_geom_optimization()
#             p_opt = p.xtb_geom_optimization()

#             r_opt.to_xyz(start_fp)
#             p_opt.to_xyz(end_fp)
#     except:
#         print(f"Redo: {rn}")
    try:
        r = TDStructure.from_xyz(str(start_fp.resolve()))
        p = TDStructure.from_xyz(str(end_fp.resolve()))
        if len(r.coords) > 0 and len(p.coords) > 0:
            all_structures.append(r)
            all_structures.append(p)
    except:
        print(f"\tRedo: {rn}")
# -


basis = 'def2-svp'
method = 'wb97xd3'
kwds = {'maxit':500}

client = CCClient()

batch1 = all_structures[:100]
for td in batch1:
    td.tc_model_basis = basis
    td.tc_model_method = method
    td.tc_kwds = kwds

batch1_inputs = [td._prepare_input('opt') for td in batch1]

batch1_opts = client.compute_procedure(batch1_inputs, procedure='geometric')

batch1_results = batch1_opts.get()

failed_inds = []
for i, result in enumerate(batch1_results):
    if not result.success:
        failed_inds.append(i)    
    

batch1_results[6].error.extras

batch1_results[failed_inds[0]].error.extras

len(failed_inds)

r = client.compute_procedure(batch1_inputs[6], procedure='geometric', program='terachem_pbs')
r_out = r.get()

print(r_out.error.error_message) 

print(r_out.error.extras)

batch1[failed_inds[-1]].symbols

batch1_inputs[6].keywords

r = client.compute_procedure(batch1_inputs[6], procedure='geometric')
r_out = r.get()

print(r_out.error.error_message)

batch1_inputs[6].keywords["program"] = 'terachem_pbs'

# +
# for rn in rns:
#     # try:
#     if rn == "Wittig": continue
#     out = ca.structures_dir / rn / "initial_guess.xyz"
#     # if out.exists():
#     #     out.unlink()
#     if not out.exists():
#         print(rn)

#         start_fp = ca.structures_dir / rn / "start.xyz"
#         end_fp = ca.structures_dir / rn / "end.xyz"

#         r = TDStructure.from_xyz(str(start_fp.resolve()))
#         p = TDStructure.from_xyz(str(end_fp.resolve()))


#         # r.tc_model_basis = 'gfn2xtb'
#         # r.tc_model_method = 'gfn2xtb'

# #         r_opt = r.tc_geom_optimization()

# #         p.tc_model_basis = 'gfn2xtb'
# #         p.tc_model_method = 'gfn2xtb'

# #         p_opt = p.tc_geom_optimization()

#         gis = []
#         eAs = []
#         for i, fric in enumerate(frics):
#             tr = Trajectory([r,p]).run_geodesic(nimages=15, friction=fric)
#             gi_out = ca.structures_dir / rn / f'gi_fric{fric}.xyz'
#             tr.write_trajectory(str(gi_out.resolve()))
#             try:
#                 eAs.append(max(tr.energies_xtb()))
#                 gis.append(tr)

#             except:
#                 continue

#         best_gi = np.argmin(eAs)
#         gis[best_gi].write_trajectory(str(out.resolve()))





#     else:
#         print(f"\tInterpolation for {rn} already exists.")
#     # except:
#     #     print(f"\t{rn} did not work")

# -

# # submit jobs

comparisons_dir = Path("/home/jdep/T3D_data/msmep_draft/comparisons") 
# ca = CompetitorAnalyzer(comparisons_dir=comparisons_dir,method='dlfind')
# ca2 = CompetitorAnalyzer(comparisons_dir=comparisons_dir,method='pygsm')
ca3 = CompetitorAnalyzer(comparisons_dir=comparisons_dir,method='nebd')

ca3.create_all_files_and_folders()

ca3.submit_all_jobs()

ca4 = CompetitorAnalyzer(comparisons_dir=comparisons_dir,method='asneb')

ca4.create_all_files_and_folders()

ca4.submit_all_jobs()

# +
# ca3.submit_all_jobs()
# -

ca4 = CompetitorAnalyzer(comparisons_dir=comparisons_dir,method='asneb')

# +
# ca4.create_all_files_and_folders()

# +
# ca4.submit_all_jobs()
# -

ca3.conv_file_name = "initial_guess_neb.xyz"
ca4.conv_file_name = "initial_guess_msmep.xyz"

# +
# obj = ca4
obj = ca3
ind = 1


# for rn in ca4.available_reaction_names:
rn = obj.available_reaction_names[ind]
print(rn)
data_dir = obj.out_folder / rn 
# data_fp = data_dir / "initial_guess_msmep.xyz"
data_fp = data_dir / obj.conv_file_name

cni = ChainInputs()
c = Chain.from_xyz(data_fp, cni)
c.plot_chain()

# -

n = NEB.read_from_disk(data_dir / obj.conv_file_name[:-4])

history_fp = data_dir / obj.conv_file_name[:-4]
history = TreeNode.read_from_disk(history_fp)


history.data.plot_chain_distances()
