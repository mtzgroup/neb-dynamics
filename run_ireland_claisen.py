# +
from neb_dynamics.MSMEP import MSMEP
import retropaths.helper_functions as hf
from retropaths.abinitio.trajectory import Trajectory
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.molecules.molecule import Molecule

from IPython.core.display import HTML


from retropaths.reactions.changes import Changes3DList, Changes3D
from retropaths.reactions.template import ReactionTemplate
from retropaths.reactions.conditions import Conditions
from retropaths.reactions.rules import Rules
from pathlib import Path
import time

n_waters = 0
smi = "C12(C(=C)O[Si](C)(C)C)C(=O)OC3CCC1C23C4=CCCCC4"
smi += ".O" * n_waters
smi_ref = "O" * n_waters


mol = Molecule.from_smiles(smi)

# -

mol.draw(mode="d3", size=(400, 400))

# + endofcell="--"
ind = 0

single_list = [(1, 2), (2, 17), (17, 16), (16, 0)]
double_list = [(15, 14), (0, 1)]
delete_list = [(0, 15), (0, 14)]

forming_list = [Changes3D(start=s, end=e, bond_order=1) for s, e in single_list]
forming_list += [Changes3D(start=s, end=e, bond_order=2) for s, e in double_list]

settings = [
    (
        mol,
        {
            "charges": [],
            "delete": delete_list,
            "single": single_list,
            "double": double_list,
        },
        [],
        [
            Changes3D(start=s, end=e, bond_order=1) for s, e in delete_list
        ],  # deleting list
        forming_list,
    )
]

mol, d, cg, deleting_list, forming_list = settings[ind]


conds = Conditions()
rules = Rules()
temp = ReactionTemplate.from_components(
    name="Ireland-Claisen",
    reactants=mol,
    changes_react_to_prod_dict=d,
    conditions=conds,
    rules=rules,
    collapse_groups=cg,
)

c3d_list = Changes3DList(deleted=deleting_list, forming=forming_list, charges=[])

root = TDStructure.from_smiles(smi, tot_spinmult=1)

root.molecule_rp.draw(mode="d3")

temp.reactants.draw()

temp.products.draw(mode="rdkit")

target = root.copy()
target.add_bonds(c3d_list.forming)
target.delete_bonds(c3d_list.deleted)
target.mm_optimization("gaff")
target.mm_optimization("uff")
target.mm_optimization("mmff94")

target = target.xtb_geom_optimization()


start = time.time()

tol = 0.005
m = MSMEP(v=True, tol=tol, max_steps=5000, step_size=0.33)

n_obj, out_chain = m.find_mep_multistep((root, target), do_alignment=False)

end = time.time()

print(f"Time (s): {end - start}")

t = Trajectory([node.tdstructure for node in out_chain])
t.write_trajectory(f"./ireland_claisen_mep_tol_{tol}.xyz")
