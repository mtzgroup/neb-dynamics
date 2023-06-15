from retropaths.abinitio.trajectory import Trajectory
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.molecules.molecule import Molecule
import numpy as np
from IPython.core.display import HTML, display
from pathlib import Path
import warnings 
from subprocess import Popen, PIPE
warnings.filterwarnings('ignore')
from rdkit import RDLogger
import shutil
RDLogger.DisableLog('rdApp.*')
HTML('<script src="//d3js.org/d3.v3.min.js"></script>')

# # GAM

# +
# import os
# del os.environ['OE_LICENSE']
# -

from GeodesicAtomMapper import AlessioError, GIAtomMapper, ReactionHandler


def get_eA(arr: np.array):
    return max(arr) - arr[0]


smi_r, smi_p = 'C=C.C=CC=C','C1C=CCCC1'
smi_correct = '[CH2:1]=[CH2:2].[CH2:3]=[CH:4][CH:5]=[CH2:6]>>[CH:1]1[CH:2][CH2:3][CH2:4]=[CH2:5][CH2:6]1'
smi_r_correct, smi_p_correct = smi_correct.split(">>")

mol_r_correct = Molecule.from_mapped_smiles(smi_r_correct)

gam = GIAtomMapper.from_smiles(smi1=smi_r, smi2=smi_p, run_in_parallel=True, do_combinatorics=True)
# try:
#     gam.correctly_map_two_smiles()
# except AlessioError as e:
#     print(e.message)

p = TDStructure.from_smiles(smi_p)

r = TDStructure.from_smiles(smi_r)

import time

# +
input_pair = r,p

def estimate_gi_barrier(input_pair, conv_thre=5/627.5, prev_en=0,nimg=25):
    td0, td1 = input_pair
    t_start = time.time()
    tr = Trajectory([td0, td1]).run_geodesic(nimages=nimg, sweep=False)
    t_end = time.time()
    len_tr = len(tr)
    img_start, img_mid, img_end = tr[0], tr[int(len_tr/2 - 1)], tr[len_tr-1]

    pair1_en = img_start.energy_xtb() + img_mid.energy_xtb()
    pair2_en = img_end.energy_xtb() + img_mid.energy_xtb()

    all_ens = [pair1_en, pair2_en]
    all_struct_pairs = [(img_start, img_mid), (img_mid, img_end)]



    ind_max_pair = np.argmax(all_ens)
    delta_en = np.abs(all_ens[ind_max_pair] - prev_en)
    converged = delta_en <= conv_thre
    print(delta_en, converged, f'time_gi: {round(t_end - t_start, 3)} s')

    if converged:
        return all_ens[ind_max_pair]

    else:
        
        inp_pair = all_struct_pairs[ind_max_pair]
        return estimate_gi_barrier(input_pair=inp_pair, conv_thre=conv_thre, prev_en=all_ens[ind_max_pair],nimg=nimg)

# -

start = time.time()
tr = Trajectory([r, p]).run_geodesic(nimages=5, sweep=False)
end = time.time()
print(end-start)

start = time.time()
tr2 = tr.run_geodesic(nimages=10)
end = time.time()
print(end-start)

start = time.time()
tr10 = Trajectory([r, p]).run_geodesic(nimages=10, sweep=False)
end = time.time()
print(end-start)

start = time.time()
tr = Trajectory([r, p]).run_geodesic(nimages=20, sweep=False)
end = time.time()
print(end-start)

start = time.time()
tr = tr.run_geodesic(nimages=50, sweep=False)
end = time.time()
print(end-start)

start = time.time()
tr = Trajectory([r, p]).run_geodesic(nimages=50, sweep=False)
end = time.time()
print(end-start)

# %%time
estimate_gi_barrier(input_pair,nimg=5)

# %%time
estimate_gi_barrier(input_pair,nimg=10)

# %%time
estimate_gi_barrier(input_pair, nimg=20)

# %%time
estimate_gi_barrier(input_pair,nimg=30)

# %%time
estimate_gi_barrier(input_pair,nimg=40)

estimated_barrier = [-35.12520693773743, -34.57891646078902, -34.538933206548734, -34.477837502707196, -34.43539201542643]
nims = [5,10,20,30,  40]
time = [2.69, 7.76, 42.2, 2*60 + 12, 5*60+14] # seconds

import matplotlib.pyplot as plt

plt.plot(nims, estimated_barrier,'o-')

plt.plot(nims, time,'o-')

# # Arbalign

# +
from contextlib import contextmanager
import os

@contextmanager
def cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)



# -

def align_two_tds(start, end):
    with cwd(Path("/tmp")):
        start.to_xyz("/tmp/start.xyz")
        end.to_xyz("/tmp/int.xyz")
        process = Popen(['ArbAlign-scipy.py' , 'start.xyz','int.xyz'], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        td_ref = TDStructure.from_xyz("start.xyz")
        td_output = TDStructure.from_xyz("/tmp/int-aligned_to-start.xyz")
        # td_output = TDStructure.from_xyz("/tmp/int-aligned_to-start-noHydrogens.xyz")
        Path("/tmp/start.xyz").unlink()
        Path("/tmp/int.xyz").unlink()
        Path("/tmp/int-aligned_to-start.xyz").unlink()
    return td_ref, td_output


def align_two_tds_noH(start, end):
    with cwd(Path("/tmp")):
        start.to_xyz("/tmp/start.xyz")
        end.to_xyz("/tmp/int.xyz")
        process = Popen(['ArbAlign-scipy.py' , '-n', 'start.xyz','int.xyz'], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        td_ref = TDStructure.from_xyz("start.xyz")
        # td_output = TDStructure.from_xyz("/tmp/int-aligned_to-start.xyz")
        td_output = TDStructure.from_xyz("/tmp/int-aligned_to-start-noHydrogens.xyz")
        Path("/tmp/start.xyz").unlink()
        Path("/tmp/int.xyz").unlink()
        # Path("/tmp/int-aligned_to-start.xyz").unlink()
    return td_ref, td_output


# +
# a = TDStructure.from_smiles("C1=CC=CC=C1C(=O)C.C(=O)(C)C.[O-][H]")
# mol_a = Molecule.from_mapped_smiles("[CH2:6]=[CH2:3].[CH2:2]=[CH:4][CH:5]=[CH2:1]")

a = TDStructure.from_smiles("C=C.C=CC=C")
# a = TDStructure.from_RP(mol_a)
a.gum_mm_optimization()
# -

import numpy as np

pairs = list(zip(a.coords, a.symbols))

np.random.shuffle(pairs)

# +
coords = []
symbs = []
for c, s in pairs:
    if s == "C":
        coords.append(c)
        symbs.append(s)
        
for c, s in pairs:
    if s == "H":
        coords.append(c)
        symbs.append(s)
# -

coords = np.array(coords)

a = TDStructure.from_coords_symbols(coords, symbs)

# +
# a_opt = a.tc_geom_optimization()
# -

# b = TDStructure.from_smiles("[O-][H].C1=CC=CC=C1C(=O)C=C(C)C.O")
b = TDStructure.from_smiles("C1=CCCCC1")
b.gum_mm_optimization()

# +
# b_opt = b.tc_geom_optimization()

# +
# a_opt.to_xyz("./dip/start_b3lyp.xyz")

# +
# b_opt.to_xyz("./dip/end_b3lyp.xyz")
# -

# s, out = align_two_tds(a_opt,b_opt)
s, out = align_two_tds(a,b)
# s, out = align_two_tds_noH(a,b)

out

Molecule.draw_list([a.molecule_rp, b.molecule_rp], mode='d3')

Molecule.draw_list([s.molecule_rp, out.molecule_rp], mode='d3')

tr_orig = Trajectory([a,b]).run_geodesic(nimages=15)
tr = Trajectory([s,out]).run_geodesic(nimages=15)

import matplotlib.pyplot as plt

out.energy_xtb()

b.energy_xtb()

plt.plot(tr_orig.energies_xtb(),"o-",label='orig')
plt.plot(tr.energies_xtb(),"o-",label="arbd")
plt.legend()

Molecule.draw_list([s.molecule_rp, out.molecule_rp], mode='d3')

tr = Trajectory([start, td_output]).run_geodesic(nimages=10)

tr.draw();
