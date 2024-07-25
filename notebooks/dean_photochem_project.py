from neb_dynamics.tdstructure import TDStructure
from neb_dynamics.trajectory import Trajectory
import numpy as np

td_b3lyp = TDStructure.from_xyz("/home/jdep/T3D_data/dean_photochem/CH2I2/b3lyp_def2svp.xyz")
td_wb97xd3 = TDStructure.from_xyz("/home/jdep/T3D_data/dean_photochem/CH2I2/wb97xd3_def2svp.xyz")
td_wb97xd3_lanl = TDStructure.from_xyz("/home/jdep/T3D_data/dean_photochem/CH2I2/wb97xd3_lanl2dzecp.xyz")
td_wpbe = TDStructure.from_xyz("/home/jdep/T3D_data/dean_photochem/CH2I2/wpbe_def2svp.xyz")
td_wpbe_lanl = TDStructure.from_xyz("/home/jdep/T3D_data/dean_photochem/CH2I2/wpbe_lanl2dzecp.xyz")

td = TDStructure.from_smiles("C(I)I")

# +
td.tc_model_method = 'wb97xd3'
td.tc_model_basis = 'lanl2dz_ecp'
# td.tc_kwds = {}
# td.tc_kwds = {'rc_w':0.3}
td.tc_kwds = {
    'cis':'yes',
    'cisnumstates':20,
    'cisguessvecs':50,
    'cismax':100,
    'ciscouplings':'yes',
    
#    'd3':'bj'
}


# -

def get_CIC_angle(td):
    opp = td.coords[1] - td.coords[0]
    adj = td.coords[2] - td.coords[0]
    angle = np.arccos(
    np.dot(opp, adj) / (np.linalg.norm(opp)*np.linalg.norm(adj))
         )*180/np.pi
    return angle


td_wb97xd3_lanl.update_tc_parameters(td)

# +
# from qcio import DualProgramInput, Molecule, SinglePointOutput

# from chemcloud import CCClient

# water = Molecule(
#     symbols=["O", "H", "H"],
#     geometry=[
#         [0.0000, 0.00000, 0.0000],
#         [0.2774, 0.89290, 0.2544],
#         [0.6067, -0.23830, -0.7169],
#     ],
# )

# client = CCClient()

# prog_inp = DualProgramInput(
#     molecule=water,
#     calctype="hessian",
#     subprogram="terachem",
#     subprogram_args={"model": {"method": "b3lyp", "basis": "6-31g"}},
# )


# # Submit calculation
# future_result = client.compute("bigchem", prog_inp, collect_files=True)
# output: SinglePointOutput = future_result.get()

# # SinglePointOutput object containing all returned data
# print(output)
# print(output.results.hessian)
# # Frequency data always included too
# print(f"Wavenumbers: {output.results.freqs_wavenumber}")
# print(output.results.normal_modes_cartesian)
# print(output.results.gibbs_free_energy)
# -

from pathlib import Path

xyz_dir = Path('/home/jdep/T3D_data/dean_photochem/CH2I2/wigner/0K/wigner_ICs/')

all_xyz_paths = list(xyz_dir.glob("x*.xyz"))

tr = Trajectory([TDStructure.from_xyz(fp) for fp in all_xyz_paths])

tr

# +
# submit tddft energies and collect the excittaiton energies
# -

td_wb97xd3_lanl.compute_tc('terachem','hessian')

hess = td_wb97xd3_lanl.compute_tc('terachem','hessian')

hess

obj2 = td_wb97xd3_lanl.compute_tc_local('terachem', 'energy', return_object=True)

obj2.pstdout

obj = td_wb97xd3_lanl.compute_tc_local('terachem', 'energy',return_object=True)

obj.pstdout

# !pwd

from neb_dynamics.Refiner import Refiner
from pathlib import Path

ref = Refiner()


leaves = ref.read_leaves_from_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Benzimidazolone-Synthesis-1-X-Iodine/refined_results/"))

out = ref.join_output_leaves(leaves)

from neb_dynamics.treenode import TreeNode

h = TreeNode.read_from_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Benzimidazolone-Synthesis-1-X-Iodine/production_vpo_tjm_msmep/"))

h.output_chain.plot_chain()

import numpy as np

np.argmax(out.energies)

td = out[6].tdstructure

td.tc_model_method = 'wb97xd3'
td.tc_model_basis = 'def2-svp'

obj = td.compute_tc_local('terachem','energy', return_object=True)

obj.pstdout

len([line for line in mold_ts.split("\n") if 'Occup= 2' in line])



mold = obj.files['scr.geometry/geometry.molden']


def write_molden_to_fp(fp, text):
    with open(fp,'w+') as f:
        f.write(obj.files['scr.geometry/geometry.molden'])


write_molden_to_fp('/home/jdep/T3D_data/mold_ind6.molden', mold)

len([l for l in obj.files['scr.geometry/geometry.molden'].split("\n") if 'Ene=' in l ])

obj.files.keys()


def get_CI_lengths(td):
    opp = td.coords[1] - td.coords[0]
    adj = td.coords[2] - td.coords[0]
    lengths = np.linalg.norm(opp), np.linalg.norm(adj)
    return lengths


get_CIC_angle(td_b3lyp), get_CIC_angle(td_wb97xd3), get_CIC_angle(td_wpbe),  get_CIC_angle(td_xtb), get_CIC_angle(td_wpbe_lanl),get_CIC_angle(td_wb97xd3_lanl)

get_CI_lengths(td_b3lyp), get_CI_lengths(td_wb97xd3), get_CI_lengths(td_wpbe),  get_CI_lengths(td_xtb), get_CI_lengths(td_wpbe_lanl),get_CI_lengths(td_wb97xd3_lanl)

td_b3_2

get_CIC_angle()

get_CI_lengths(td_b3lyp), get_CI_lengths(td_wb97xd3), get_CI_lengths(td_wpbe), get_CI_lengths(td_xtb), get_CI_lengths(td)

td_xtb

# +
# td_opt.to_xyz("/home/jdep/T3D_data/dean_photochem/CH2I2/wpbe_def2svp.xyz")

# +
# td_opt.to_xyz("/home/jdep/T3D_data/dean_photochem/CH2I2/b3lyp_def2svp.xyz")

# +
# td_opt.to_xyz("/home/jdep/T3D_data/dean_photochem/CH2I2/wb97xd3_def2svp.xyz")
# -


