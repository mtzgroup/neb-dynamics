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
from retropaths.molecules.elements import ElementData

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
    ind_maxima = argrelextrema(chain.energies, np.greater, order=1)[0]
    return ind_maxima


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
