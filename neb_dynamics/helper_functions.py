"""
this module contains helper general functions
"""
import scipy.sparse.linalg
import cProfile
import json
import multiprocessing as mp
import pickle
import signal
from itertools import repeat
from pathlib import Path
import warnings
import math
import numpy as np
import pandas as pd
from IPython.core.display import HTML
from scipy.signal import argrelextrema
import openbabel
from pysmiles import write_smiles
from openeye import oechem
import os
from rdkit import Chem

import sys


from neb_dynamics.elements import ElementData

warnings.filterwarnings('ignore')

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
    
    
def _get_ind_minima(chain):
    ind_minima = argrelextrema(chain.energies, np.less, order=1)[0]
    return ind_minima


def _get_ind_maxima(chain):
    maxima_indices = argrelextrema(chain.energies, np.greater, order=1)[0]
    if len(maxima_indices) > 1:
        ind_maxima = maxima_indices[0]
    else:
        ind_maxima = int(maxima_indices)
    return ind_maxima


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
    qmax = np.float_(qmax)
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

def create_friction_optimal_gi(traj, gi_inputs):
    print("GI: Scanning over friction parameter for geodesic interpolation.")
    sys.stdout.flush()
    frics = [0.0001, 0.001, 0.01, 0.1, 1]
    all_gis = [
        traj.run_geodesic(
        nimages=gi_inputs.nimages,
        friction=fric,
        nudge=gi_inputs.nudge,
        **gi_inputs.extra_kwds,
    ) for fric in frics
    ]
    eAs = []
    for gi in all_gis:
        try:
            eAs.append(max(gi.energies_xtb()) )
        except TypeError:
            eAs.append(10000000)
    ind_best = np.argmin(eAs)
    gi = all_gis[ind_best]
    print(f"GI: Used friction val: {frics[ind_best]}")
    sys.stdout.flush()
    return gi

def mass_weight_coords(labels, coords):
   weights = np.array([np.sqrt(get_mass(s)) for s in labels])
   weights = weights / sum(weights)
   coords = coords * weights.reshape(-1,1)
   return coords

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

def write_xyz(atoms, X, name=''):
    natoms = len(atoms)
    string = ''
    string += f'{natoms}\n{name}\n'
    for i in range(natoms):
        string += f'{atoms[i]} {X[i,0]} {X[i,1]} {X[i,2]} \n'
    return(string)
    
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

        if 'OE_LICENSE' in os.environ:
            # Keiran oechem (this needs the license)
            smi = write_smiles(mol)
            oemol = oechem.OEGraphMol()
            oechem.OESmilesToMol(oemol, smi)
            smi2 = oechem.OEMolToSmiles(oemol)
        else:
            try:
                b = write_smiles(mol)
                smi2 = Chem.CanonSmiles(b)
            except Exception as e:
                print('This run is without OECHEM license. A graph has failed to be converted in smiles. Take note of the exception.')
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
    from neb_dynamics.tdstructure import TDStructure
    from neb_dynamics.trajectory import Trajectory
    optim_fp = Path(tmp.name[:-4]) / "optim.xyz"
    tr = Trajectory.from_xyz(optim_fp,
        tot_charge=td.charge,
        tot_spinmult=td.spinmult)
    tr.update_tc_parameters(td)
    
    if return_optim_traj:
        return tr
    else:
        return tr[-1]
    

def is_even(n):
    return not np.mod(n, 2)