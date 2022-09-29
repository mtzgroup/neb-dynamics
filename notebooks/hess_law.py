from retropaths.abinitio.tdstructure import TDStructure
from neb_dynamics.NEB import Node3D
import numpy as np
import matplotlib.pyplot as plt


# +
def change_to_bohr(td):
    coords_b = td.coords_bohr
    td.update_coords(coords_b)
    return td


def change_to_ang(td):
    coords_a = td.coords*BOHR_TO_ANGSTROMS
    td.update_coords(coords_a)
    return td


# -

td = TDStructure.from_smiles("[C]")
no = Node3D(td)
td.molecule_rp.draw()
c_en = no.energy

td = TDStructure.from_smiles("[H]")
no = Node3D(td)
td.molecule_rp.draw()
h_en = no.energy


td = TDStructure.from_smiles("C")
no = Node3D(td)
td.molecule_rp.draw()
ch4_en = no.energy


c_en + h_en

(ch4_en - (c_en + h_en))*627.5

print("C-H bond en: ", (no.energy/4)*627.5)

smi = "[H][H]"
def get_bde(smi, r_min=-1,r_max=3.5, charge=0, spinmult=1):
    td = change_to_bohr(TDStructure.from_smiles(smi, tot_charge=charge, tot_spinmult=spinmult))
    coords = td.coords
    coords -= coords[0] # center around one atom
    td.update_coords(coords)
    vec = (coords[1] - coords[0])/np.linalg.norm(coords[1] - coords[0])
    # print(vec)
    
    # n = Node3D(td)
    # td = n.opt_func()
    
    ens = []
    for delta in np.arange(r_min, r_max, 0.01):
        td_c = td.copy()
        cor = td_c.coords
        # print(cor)
        cor[1] += delta*vec
        # print(cor)
        td_c.update_coords(cor)
        no = Node3D(td_c)
        # td.molecule_rp.draw()
        ens.append(no.energy)


    plt.plot(np.arange(r_min, r_max, 0.01), ens, 'o')
    print(f"{ens[-1]=} // {min(ens)}")
    print(f"{smi=} ∆E: {round((ens[-1] - min(ens))*627.5*4.184, 3)} kJ/mol")
get_bde(smi)

get_bde("[C][C]",charge=0, r_min=-1, r_max=3.6)


