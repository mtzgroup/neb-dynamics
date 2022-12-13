from retropaths.molecules.molecule import Molecule
from retropaths.reactions.changes import Changes3DList, Changes3D
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.abinitio.trajectory import Trajectory


from neb_dynamics.MSMEP import MSMEP

from retropaths.reactions.template import ReactionTemplate
from retropaths.reactions.conditions import Conditions
from retropaths.reactions.rules import Rules

import random

random.seed(1)
ind = 1

settings = [

    (
        Molecule.from_smiles("CC(=O)C.CP(=C)(C)C"),
        {'charges': [], 'delete':[(5, 6), (1, 2)], 'double':[(1, 6), (2, 5)]},
        [
            (7, 5, 'Me'),
            (8, 5, 'Me'),
            (4, 5, 'Me'),
            (0, 1, 'Me'),
            (3, 1, 'Me'),
        ],
        [Changes3D(start=s, end=e, bond_order=1) for s, e in [(5, 6), (2, 1)]],
        [Changes3D(start=s, end=e, bond_order=2) for s, e in [(1, 6), (2, 5)]]

    ),
    (
        Molecule.from_smiles("C(=O)(C)C.C=P(c1ccccc1)(c2ccccc2)c3ccccc3"),
        {'charges': [], 'delete':[(5, 4), (1, 0)], 'double':[(0, 4), (1, 5)]},
        [
            (18, 5, 'Me'),
            (6, 5, 'Me'),
            (12, 5, 'Me'),
            (2, 0, 'Me'),
            (3, 0, 'Me'),
        ],
        [Changes3D(start=s, end=e, bond_order=1) for s, e in [(5, 4), (1, 0)]],
        [Changes3D(start=s, end=e, bond_order=2) for s, e in [(0, 4), (1, 5)]]

    )


]

mol, d, cg, deleting_list, forming_list = settings[ind]


conds = Conditions()
rules = Rules()
temp = ReactionTemplate.from_components(name='Wittig', reactants=mol, changes_react_to_prod_dict=d, conditions=conds, rules=rules, collapse_groups=cg)

c3d_list = Changes3DList(deleted=deleting_list, forming=forming_list, charges=[])
# -

root = TDStructure.from_RP(temp.reactants)
root = root.pseudoalign(c3d_list)
root = root.xtb_geom_optimization()

target = root.copy()
target.add_bonds(c3d_list.forming)
target.delete_bonds(c3d_list.deleted)
target.mm_optimization("gaff", steps=5000)
target.mm_optimization("uff", steps=5000)
target.mm_optimization("mmff94", steps=5000)
target = target.xtb_geom_optimization()


m = MSMEP(max_steps=2000, v=True, tol=0.01)

# n_obj, out_chain = m.get_neb_chain(root, target,do_alignment=False)
# t = Trajectory([node.tdstructure for node in out_chain])
# t.write_trajectory("./example_chain.xyz")
n_obj2, out_chain2 = m.find_mep_multistep((root, target), do_alignment=True)


t = Trajectory([node.tdstructure for node in out_chain2])
# t.write_trajectory("./example_mep.xyz")
t.write_trajectory("./example_mep_phenyl.xyz")
