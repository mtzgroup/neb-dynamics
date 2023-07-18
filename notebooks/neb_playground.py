# -*- coding: utf-8 -*-
# +
from dataclasses import dataclass
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from neb_dynamics.Node import Node
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math
from itertools import product
import warnings
warnings.filterwarnings("ignore")
import retropaths.helper_functions  as hf

from retropaths.abinitio.trajectory import Trajectory



from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs, NEBInputs, GIInputs
from neb_dynamics.NEB import NEB
from neb_dynamics.Node2d import Node2D_Flower
from neb_dynamics.Node3D_TC import Node3D_TC

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')                                                                                                                                                           


from matplotlib.animation import FuncAnimation
import IPython
from pathlib import Path
# -

# # Generate some initial guess and NEB optimized path

tol = 0.005

reactions = hf.pload("/home/jdep/retropaths/data/reactions.p")

# +
cni = ChainInputs(als_max_steps=3)
nbi = NEBInputs(v=True,early_stop_force_thre=0.01, grad_thre=tol, en_thre=tol/2, rms_grad_thre=tol/2)
gii = GIInputs(nimages=10)

m = MSMEP(nbi,cni,gii)
# -

start, end = m.create_endpoints_from_rxn_name('Claisen-Rearrangement',reactions_object=reactions)

start.tc_model_method = 'wb97xd3'
start.tc_model_basis = 'def2-svp'
start_node = Node3D_TC(start)

start.energy_tc()

start2 = start.copy()
start2.tc_model_method = 'b3lyp'
start2.tc_model_basis = 'sto-3g'

start2.energy_tc()

# +
# gi = Trajectory([start, end]).run_geodesic(nimages=10)
# -

chain = Chain.from_traj(gi,parameters=cni)



h, out = m.find_mep_multistep(chain)

n.optimize_chain()

#
