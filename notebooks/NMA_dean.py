from neb_dynamics.NEB import Node3D
# from neb_dynamics.tdstructure import TDStructure
from retropaths.abinitio.tdstructure import TDStructure
from pathlib import Path

td = TDStructure.from_fp(Path("./claisen_ts_guess.xyz"))

td



node = Node3D()
