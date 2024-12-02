
"""
this module contains helper general functions
"""
from __future__ import annotations

import math

# from openeye import oechem
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse.linalg
from openbabel import openbabel
from pysmiles import write_smiles
from rdkit import Chem
from typing import List

from neb_dynamics.elements import ElementData


warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", r"RuntimeWarning: invalid value encountered in divide"
    )


def pairwise(iterable):
    """
    from a list [a,b,c,d] to [(a,b),(b,c),(c,d)]
    """
    it = iter(iterable)
    a = next(it, None)

    for b in it:
        yield (a, b)
        a = b


def get_mass(s: str):
    """
    return atomic mass from symbol
    """
    ed = ElementData()
    return ed.from_symbol(s).mass_amu


def qRMSD_distance(structure, reference):
    return RMSD(structure, reference)[0]


def RMSD(structure, reference):
    c1 = np.array(structure)
    c2 = np.array(reference)
    bary1 = np.mean(c1, axis=0)  # barycenters
    bary2 = np.mean(c2, axis=0)

    c1 = c1 - bary1  # shift origins to barycenter
    c2 = c2 - bary2

    N = len(c1)
    R = np.dot(np.transpose(c1), c2)  # correlation matrix

    F = np.array(
        [
            [
                (R[0, 0] + R[1, 1] + R[2, 2]),
                (R[1, 2] - R[2, 1]),
                (R[2, 0] - R[0, 2]),
                (R[0, 1] - R[1, 0]),
            ],
            [
                (R[1, 2] - R[2, 1]),
                (R[0, 0] - R[1, 1] - R[2, 2]),
                (R[1, 0] + R[0, 1]),
                (R[2, 0] + R[0, 2]),
            ],
            [
                (R[2, 0] - R[0, 2]),
                (R[1, 0] + R[0, 1]),
                (-R[0, 0] + R[1, 1] - R[2, 2]),
                (R[1, 2] + R[2, 1]),
            ],
            [
                (R[0, 1] - R[1, 0]),
                (R[2, 0] + R[0, 2]),
                (R[1, 2] + R[2, 1]),
                (-R[0, 0] - R[1, 1] + R[2, 2]),
            ],
        ]
    )  # Eq. 10 in Dill Quaternion RMSD paper (DOI:10.1002/jcc.20110)

    eigen = scipy.sparse.linalg.eigs(
        F, k=1, which="LR"
    )  # find max eigenvalue and eigenvector
    lmax = float(eigen[0][0])
    qmax = np.array(eigen[1][0:4])
    qmax = np.float64(qmax)
    qmax = np.ndarray.flatten(qmax)
    rmsd = math.sqrt(
        abs((np.sum(np.square(c1)) + np.sum(np.square(c2)) - 2 * lmax) / N)
    )  # square root of the minimum residual

    rot = np.array(
        [
            [
                (qmax[0] ** 2 + qmax[1] ** 2 - qmax[2] ** 2 - qmax[3] ** 2),
                2 * (qmax[1] * qmax[2] - qmax[0] * qmax[3]),
                2 * (qmax[1] * qmax[3] + qmax[0] * qmax[2]),
            ],
            [
                2 * (qmax[1] * qmax[2] + qmax[0] * qmax[3]),
                (qmax[0] ** 2 - qmax[1] ** 2 + qmax[2] ** 2 - qmax[3] ** 2),
                2 * (qmax[2] * qmax[3] - qmax[0] * qmax[1]),
            ],
            [
                2 * (qmax[1] * qmax[3] - qmax[0] * qmax[2]),
                2 * (qmax[2] * qmax[3] + qmax[0] * qmax[1]),
                (qmax[0] ** 2 - qmax[1] ** 2 - qmax[2] ** 2 + qmax[3] ** 2),
            ],
        ]
    )  # rotation matrix based on eigenvector corresponding $
    g_rmsd = (c1 - np.matmul(c2, rot)) / (N * rmsd)  # gradient of the rmsd

    return rmsd, g_rmsd


def linear_distance(coords1, coords2):
    return np.linalg.norm(coords2 - coords1)


def get_nudged_pe_grad(unit_tangent, gradient):
    """
    Returns the component of the gradient that acts perpendicular to the path tangent
    """
    pe_grad = gradient
    pe_grad_nudged_const = np.dot(pe_grad.flatten(), unit_tangent.flatten())
    pe_grad_nudged = pe_grad - pe_grad_nudged_const * unit_tangent
    return pe_grad_nudged


def get_seeding_chain(neb_obj, force_thre):
    for c in neb_obj.chain_trajectory:
        if c.get_maximum_grad_magnitude() <= force_thre:
            return c
    return neb_obj.chain_trajectory[-1]


def make_copy(obmol):
    copy_obmol = openbabel.OBMol()
    for atom in openbabel.OBMolAtomIter(obmol):
        copy_obmol.AddAtom(atom)

    for bond in openbabel.OBMolBondIter(obmol):
        copy_obmol.AddBond(bond)

    copy_obmol.SetTotalCharge(obmol.GetTotalCharge())
    copy_obmol.SetTotalSpinMultiplicity(obmol.GetTotalSpinMultiplicity())

    return copy_obmol


def load_obmol_from_fp(fp: Path) -> openbabel.OBMol:
    """
    takes in a pathlib file path as input and reads it in as an openbabel molecule
    """
    if not isinstance(fp, Path):
        fp = Path(fp)

    file_type = fp.suffix[1:]  # get what type of file this is

    obmol = openbabel.OBMol()
    obconversion = openbabel.OBConversion()
    obconversion.SetInFormat(file_type)

    obconversion.ReadFile(obmol, str(fp.resolve()))

    return make_copy(obmol)


def write_xyz(atoms, X, name=""):
    natoms = len(atoms)
    string = ""
    string += f"{natoms}\n{name}\n"
    for i in range(natoms):
        string += f"{atoms[i]} {X[i,0]} {X[i,1]} {X[i,2]} \n"
    return string


def bond_ord_number_to_string(i):
    bo2s = {1.0: "single", 1.5: "aromatic", 2.0: "double", 3.0: "triple"}
    try:
        st = bo2s[i]
    except KeyError as e:
        print(f"Bond order must be in [1, 1.5, 2, 3], received {i}")
        raise e
    return st


def from_number_to_element(i):
    return __ATOM_LIST__[i - 1].capitalize()


def atomic_number_to_symbol(n):
    ed = ElementData()
    return ed.from_atomic_number(n).symbol


def graph_to_smiles(mol):

    if mol.is_empty():
        smi2 = ""
    else:
        for e in mol.edges:
            s = mol.edges[e]["bond_order"]
            if s == "single":
                i = 1
            elif s == "double":
                i = 2
            elif s == "aromatic":
                i = 1.5
            elif s == "triple":
                i = 3
            mol.edges[e]["order"] = i

        # if 'OE_LICENSE' in os.environ:
        #     # Keiran oechem (this needs the license)
        #     smi = write_smiles(mol)
        #     oemol = oechem.OEGraphMol()
        #     oechem.OESmilesToMol(oemol, smi)
        #     smi2 = oechem.OEMolToSmiles(oemol)
        # else:
        try:
            b = write_smiles(mol)
            smi2 = Chem.CanonSmiles(b)
        except Exception as e:
            print(
                "This run is without OECHEM license. A graph has failed to be converted in smiles.\
                      Take note of the exception."
            )
            raise e
    return smi2


__ATOM_LIST__ = [
    "h",
    "he",
    "li",
    "be",
    "b",
    "c",
    "n",
    "o",
    "f",
    "ne",
    "na",
    "mg",
    "al",
    "si",
    "p",
    "s",
    "cl",
    "ar",
    "k",
    "ca",
    "sc",
    "ti",
    "v",
    "cr",
    "mn",
    "fe",
    "co",
    "ni",
    "cu",
    "zn",
    "ga",
    "ge",
    "as",
    "se",
    "br",
    "kr",
    "rb",
    "sr",
    "y",
    "zr",
    "nb",
    "mo",
    "tc",
    "ru",
    "rh",
    "pd",
    "ag",
    "cd",
    "in",
    "sn",
    "sb",
    "te",
    "i",
    "xe",
    "cs",
    "ba",
    "la",
    "ce",
    "pr",
    "nd",
    "pm",
    "sm",
    "eu",
    "gd",
    "tb",
    "dy",
    "ho",
    "er",
    "tm",
    "yb",
    "lu",
    "hf",
    "ta",
    "w",
    "re",
    "os",
    "ir",
    "pt",
    "au",
    "hg",
    "tl",
    "pb",
    "bi",
    "po",
    "at",
    "rn",
    "fr",
    "ra",
    "ac",
    "th",
    "pa",
    "u",
    "np",
    "pu",
]


def run_tc_local_optimization(td, tmp, return_optim_traj):
    from neb_dynamics.trajectory import Trajectory

    optim_fp = Path(tmp.name[:-4]) / "optim.xyz"
    tr = Trajectory.from_xyz(
        optim_fp, tot_charge=td.charge, tot_spinmult=td.spinmult)
    tr.update_tc_parameters(td)

    if return_optim_traj:
        return tr
    else:
        return tr[-1]


def is_even(n):
    return not np.mod(n, 2)


# def steepest_descent(node, engine: Engine, ss=1, max_steps=10) -> list[Node]:
#     history = []
#     last_node = node.copy()
#     # make sure the node isn't frozen so it returns a gradient
#     last_node.converged = False
#     try:
#         for i in range(max_steps):
#             grad = last_node.gradient
#             new_coords = last_node.coords - 1*ss*grad
#             node_new = last_node.update_coords(new_coords)
#             engine.compute_gradients([node_new])
#             history.append(node_new)
#             last_node = node_new.copy()
#     except Exception:
#         raise ElectronicStructureError(
#             trajectory=[], msg='Error while minimizing in early stop check.')
#     return history


def give_me_free_index(natural, graph):
    """
    Natural numbers that are not index in graph
    """
    for i in natural:
        if i not in graph.nodes:
            yield i


def naturals(n):
    """
    Natural numbers
    """
    yield n
    yield from naturals(n + 1)


def _load_info_from_tcin(file_path):
    method = None
    basis = None

    charge = None
    spinmult = None

    inp_kwds = {}
    tcin = open(file_path).read().splitlines()
    assert (
        len(tcin) >= 2
    ), "File path given needs to have at least method and basis specified."
    for line in tcin:
        key, val = line.split()
        if key == "run":
            continue
        elif key == "method":
            method = val
        elif key == "basis":
            basis = val
        elif key == "coordinates":
            continue
        elif key == "charge":
            charge = int(val)
        elif key == "spinmult":
            spinmult = int(val)

        else:
            inp_kwds[key] = val

    return method, basis, charge, spinmult, inp_kwds


def _calculate_chain_distances(chain_traj):
    distances = [None]  # None for the first chain
    for i, chain in enumerate(chain_traj):
        if i == 0:
            continue

        prev_chain = chain_traj[i - 1]
        dist = prev_chain._distance_to_chain(chain)
        distances.append(dist)
    return np.array(distances)


def _create_df(filenames: List[Path], v=True, refinement_results=False, out_at_beginning_name: bool = False):
    from neb_dynamics.TreeNode import TreeNode
    do_refine = False
    multi = []
    elem = []
    failed = []
    skipped = []
    n_steps = []
    n_grad_calls = []
    n_grad_calls_geoms = []

    activation_ens = []

    n_opt_steps = []
    n_opt_splits = []

    success_names = []

    tsg_list = []
    sys_size = []
    xtb = True
    for i, p in enumerate(filenames):
        rn = p.parent.stem
        if v:
            print(p)

        try:
            if out_at_beginning_name:
                out = open(p.parent / f"out_{p.stem}").read().splitlines()
            else:
                out = open(p.parent / f"{p.stem}_out").read().splitlines()
            # if 'Traceback (most recent call last):' in out[:50] or 'Terminated' in out:
            #     raise TypeError('failure')
            if 'Warning! A chain has electronic structure errors.                         Returning an unoptimized chain...' in out:
                raise FileNotFoundError('failure')

            if refinement_results:
                clean_fp = Path(str(p.resolve())+".xyz")
                ref = Refiner()
                ref_results = ref.read_leaves_from_disk(p)

            else:
                h = TreeNode.read_from_disk(p)
                clean_fp = p.parent / (str(p.stem)+'_clean.xyz')

            if clean_fp.exists():
                try:
                    out_chain = Chain.from_xyz(clean_fp, ChainInputs())

                except:
                    if v:
                        print("\t\terror in energies. recomputing")
                    tr = Trajectory.from_xyz(clean_fp)
                    out_chain = Chain.from_traj(tr, ChainInputs())
                    grads = out_chain.gradients
                    if v:
                        print(f"\t\twriting to {clean_fp}")
                    out_chain.write_to_disk(clean_fp)

                if not out_chain._energies_already_computed:
                    if v:
                        print("\t\terror in energies. recomputing")
                    tr = Trajectory.from_xyz(clean_fp)
                    out_chain = Chain.from_traj(tr, ChainInputs())
                    grads = out_chain.gradients
                    if v:
                        print(f"\t\twriting to {clean_fp}")
                    out_chain.write_to_disk(clean_fp)
            elif not clean_fp.exists() and refinement_results:
                print(clean_fp, ' did not succeed')
                raise FileNotFoundError("boo")
            elif not clean_fp.exists() and not refinement_results:
                out_chain = h.output_chain

            es = len(out_chain) == 12
            if v:
                print('elem_step: ', es)
            if refinement_results:
                n_splits = sum([len(leaf.get_optimization_history())
                               for leaf in ref_results])
                tot_steps = sum([len(leaf.data.chain_trajectory)
                                for leaf in ref_results])
                act_en = max([leaf.data.chain_trajectory[-1].get_eA_chain()
                             for leaf in ref_results])

            else:
                if v:
                    print([len(obj.chain_trajectory)
                          for obj in h.get_optimization_history()])
                n_splits = len(h.get_optimization_history())
                if v:
                    print(sum([len(obj.chain_trajectory)
                          for obj in h.get_optimization_history()]))
                tot_steps = sum([len(obj.chain_trajectory)
                                for obj in h.get_optimization_history()])
                act_en = max([leaf.data.chain_trajectory[-2].get_eA_chain()
                             for leaf in h.ordered_leaves])

            n_opt_steps.append(tot_steps)
            n_opt_splits.append(n_splits)

            ng_line = [line for line in out if len(
                line) > 3 and line[0] == '>']
            if v:
                print(ng_line)
            ng = sum([int(ngl.split()[2]) for ngl in ng_line])

            ng_geomopt = [line for line in out if len(
                line) > 3 and line[0] == '<']
            ng_geomopt = sum([int(ngl.split()[2]) for ngl in ng_geomopt])
            # ng = 69
            if v:
                print(ng, ng_geomopt)

            activation_ens.append(act_en)
            sys_size.append(len(out_chain[0].coords))

            if es:
                elem.append(p)
            else:
                multi.append(p)

            n = len(out_chain) / 12
            n_steps.append((i, n))
            success_names.append(rn)
            n_grad_calls.append(ng)
            n_grad_calls_geoms.append(ng_geomopt)

        except FileNotFoundError as e:
            if v:
                print(e)
            failed.append(p)

        except IndexError as e:
            if v:
                print(e)
            failed.append(p)

#         except KeyboardInterrupt:
#             skipped.append(p)

        if v:
            print("")

    import pandas as pd
    df = pd.DataFrame()
    all_rns = [fp.parent.stem for fp in filenames]
    df['reaction_name'] = all_rns

    df['success'] = [fp.parent.stem in success_names for fp in filenames]

    df['n_grad_calls'] = _mod_variable(n_grad_calls, all_rns, success_names)
    if v:
        print(n_grad_calls)
    if v:
        print(_mod_variable(n_grad_calls, all_rns, success_names))

    df['n_grad_calls_geoms'] = _mod_variable(
        n_grad_calls_geoms, all_rns, success_names)

    df["n_opt_splits"] = _mod_variable(n_opt_splits, all_rns, success_names)
    # df["n_opt_splits"] = [0 for rn in all_rns]

    df['n_rxn_steps'] = _mod_variable(
        [x[1] for x in n_steps], all_rns, success_names)
    # df['n_rxn_steps'] = [0 for rn in all_rns]

    df['n_opt_steps'] = _mod_variable(n_opt_steps, all_rns, success_names)

    df['file_path'] = all_rns

    df['activation_ens'] = _mod_variable(
        activation_ens, all_rns, success_names)

    df['activation_ens'].plot(kind='hist')

    # df['n_opt_splits'].plot(kind='hist')
    # print(success_names)

    return df


def _mod_variable(var, all_rns, success_names):
    ind = 0
    mod_var = []
    for i, rn in enumerate(all_rns):
        if rn in success_names:
            mod_var.append(var[ind])
            ind += 1
        else:
            mod_var.append(None)

    return mod_var


def get_fsm_tsg_from_chain(chain):
    ind_guess = round(len(chain) / 2)
    ind_guesses = [ind_guess-1, ind_guess, ind_guess+1]
    enes_guess = [chain.energies[ind_guesses[0]],
                  chain.energies[ind_guesses[1]], chain.energies[ind_guesses[2]]]
    ind_tsg = ind_guesses[np.argmax(enes_guess)]
    return chain[ind_tsg]
