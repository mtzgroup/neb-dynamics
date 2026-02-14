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


def get_fsm_tsg_from_chain(chain):
    ind_guess = round(len(chain) / 2)
    ind_guesses = [ind_guess-1, ind_guess, ind_guess+1]
    enes_guess = [chain.energies[ind_guesses[0]],
                  chain.energies[ind_guesses[1]], chain.energies[ind_guesses[2]]]
    ind_tsg = ind_guesses[np.argmax(enes_guess)]
    return chain[ind_tsg]


def compute_irc_chain(ts_node, engine, use_bigchem: bool = False, keywords={}, **kwargs):
    from neb_dynamics.chain import Chain

    engine.compute_energies([ts_node])
    irc_negative, irc_positive = engine.compute_sd_irc(
        ts=ts_node,
        use_bigchem=use_bigchem,
        **kwargs)

    min_negative = engine.compute_geometry_optimization(
        irc_negative[-1], keywords=keywords)[-1]
    min_positive = engine.compute_geometry_optimization(
        irc_positive[-1], keywords=keywords)[-1]
    irc_negative.append(min_negative)
    irc_positive.append(min_positive)
    irc_negative.reverse()
    irc_nodes = irc_negative + \
        [ts_node]+irc_positive
    irc = Chain.model_validate(
        {"nodes": irc_nodes})
    return irc


def get_maxene_node(arr, engine):
    """
    Perform binary search to find the maximum value in a unimodal array.

    :param arr: List of numbers where the values increase to a maximum and then decrease.
    :return: The maximum value in the array.
    """
    ngcs = 0
    left, right = 0, len(arr) - 1

    while left < right:
        mid = left + (right - left) // 2

        # Check if the midpoint is the maximum
        engine.compute_energies([arr[mid], arr[mid+1]])
        ngcs += 2

        if arr[mid].energy > arr[mid + 1].energy:
            # The maximum is in the left half (including mid)
            right = mid
        else:
            # The maximum is in the right half (excluding mid)
            left = mid + 1

    # When left == right, we've found the maximum
    print("binary search cost: ", ngcs, ' grad calls')
    return {
        "node": arr[left],
        "grad_calls": ngcs,
        "index": left
    }


def parse_nma_freq_data(hessres):
    modes = []
    freqs = []

    coords = []
    for i, line in enumerate(hessres.results.files['scr.geometry/Mass.weighted.modes.dat'].split("\n")):
        if len(line) == 0:
            continue
        if line[0] == '=':
            freqs.append(float(line.split()[3]))
            if i > 0:
                modes.append(coords)
                coords = []
        else:
            coords.append(float(line))

    refshape = hessres.input_data.structure.geometry.shape
    reshaped_normal_modes = []
    for mode in modes:
        reshaped_normal_modes.append(np.array(mode).reshape(refshape))

    return reshaped_normal_modes, freqs

def project_rigid_body_forces(R, F, masses=None):
    """
    Remove net translation and rotation from forces.

    Parameters
    ----------
    R : (N,3) ndarray
        Cartesian coordinates of atoms in the image.
    F : (N,3) ndarray
        Forces on atoms (e.g., NEB effective forces).
    masses : (N,) ndarray or None
        Atomic masses. If None, assume equal masses.

    Returns
    -------
    F_proj : (N,3) ndarray
        Forces with rigid translations and rotations removed.
    """
    N = R.shape[0]
    if masses is None:
        masses = np.ones(N)
    M = masses.sum()

    # Center of mass
    R_com = np.average(R, axis=0, weights=masses)
    s = R - R_com  # positions relative to COM

    # --- 1) Remove translation ---
    F_net = F.sum(axis=0)
    F = F - (masses[:, None] / M) * F_net

    # --- 2) Remove rotation ---
    # Torque from current forces
    L = np.sum(np.cross(s, F), axis=0)

    # Inertia tensor about COM
    I = np.zeros((3, 3))
    for a in range(N):
        r = s[a]
        m = masses[a]
        I += m * ((np.dot(r, r) * np.eye(3)) - np.outer(r, r))

    # Solve I ω = L
    try:
        omega = np.linalg.solve(I, L)
    except np.linalg.LinAlgError:
        # Handle singular inertia tensor (e.g., linear molecules)
        omega = np.linalg.pinv(I) @ L

    # Subtract rotational component
    F = F - masses[:, None] * np.cross(omega, s)

    return F


def rst7_to_coords_and_indices(data):
    """

    Args:
        data (str): rst7 text file, opened

    Returns:
        tuple(np.array, list): coordinates and indices of atoms in the rst7 file
    """
    coords = []
    indices_coordinates = []

    ind = 0
    for line in data.split("\n"):
        if ind==0:
            ind+=1
            continue
        if ind==1:
           natom = int(line.split()[0])
        if (len(line.split()) == 6) or (len(line.split()) == 3):
            if len(coords) == natom:
                break
            # if ind in qmindices:
            c = line.split()[:3]
            c = [float(x) for x in c]
            coords.append(c)

            c = line.split()[3:]
            c = [float(x) for x in c]
            if len(c) > 0:
                coords.append(c)
            indices_coordinates.append(ind)


        ind+=1
    # print(coords)
    return np.array(coords), indices_coordinates

def parse_symbols_from_prmtop(data):
    """

    Args:
        data (str): prmtop text file, opened

    Returns:
        list: symbols of atoms in the prmtop file
    """
    symbols = []

    begin = False
    skipped1line = 0
    for line in data.split("\n"):
        print(line.strip())
        if line.strip()=='%FLAG ATOMIC_NUMBER':
            begin = True
            continue
        if begin:
            if skipped1line:
                if line[0]=='%':
                    break
                symbols.extend(line.split())
            else:
                skipped1line = 1


    symbols = [atomic_number_to_symbol(int(n)) for n in symbols]
    return symbols


def parse_qmmm_gradients(text):
    """
    Parses QM and MM gradients from an ORCA-style output text file.
    """
    qm_grads = []
    mm_grads = []

    # State flags
    current_mode = None # Can be 'QM' or 'MM'

    for line in text.split("\n"):
        clean_line = line.strip()

        # Detect section starts
        if "dE/dX" in clean_line and "dE/dY" in clean_line:
            current_mode = 'QM'
            continue
        elif "MM / Point charge part" in clean_line:
            current_mode = 'MM'
            continue
        # Detect section ends (the dashed line or the start of the next summary)
        elif "Net gradient" in clean_line or (current_mode == 'MM' and "---" in clean_line):
            current_mode = None
            continue

        # Parse numerical data (expecting 3 floats per line)
        parts = clean_line.split()
        if current_mode and len(parts) == 3:
            try:
                floats = [float(x) for x in parts]
                if current_mode == 'QM':
                    qm_grads.append(floats)
                elif current_mode == 'MM':
                    mm_grads.append(floats)
            except ValueError:
                # Skip lines that don't contain valid numbers (headers/separators)
                continue

    return np.array(qm_grads), np.array(mm_grads)