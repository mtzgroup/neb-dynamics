# -*- coding: utf-8 -*-
# +
# import matplotlib.pyplot as plt
# import numpy as np
# from pathlib import Path
# from retropaths.abinitio.trajectory import Trajectory
# from retropaths.abinitio.tdstructure import TDStructure
# from neb_dynamics.Chain import Chain
# from neb_dynamics.NEB import NEB
# from neb_dynamics.Node import Node
# from neb_dynamics.MSMEP import MSMEP
# from matplotlib.animation import FuncAnimation
# from neb_dynamics.helper_functions import pairwise
# from neb_dynamics.ALS import ArmijoLineSearch
# from neb_dynamics.remapping_helpers import create_correct_product

# from dataclasses import dataclass
# -

from chemcloud import CCClient

client = CCClient()

client.configure()

client.supported_engines

client.hello_world("Big Poppa")

from chemcloud.models import Molecule
water = Molecule.from_data("pubchem:water")

from chemcloud.models import AtomicInput
atomic_input = AtomicInput(molecule=water, model={"method": "B3LYP", "basis": "6-31g"}, driver="energy")

future_result = client.compute(atomic_input, engine="terachem_fe")
future_result.status

result = future_result.get()

result

result.molecule.atomic_numbers


