# +
from neb_dynamics.NEB import NEB
from chain import Chain
import os

del os.environ['OE_LICENSE']
# -

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from retropaths.abinitio.tdstructure import TDStructure
from retropaths.abinitio.trajectory import Trajectory
from neb_dynamics.Inputs import ChainInputs, NEBInputs, GIInputs
from neb_dynamics.MSMEP import MSMEP

tr = Trajectory.from_xyz(Path('/data/achang67/jan/endpoints.xyz'), tot_spinmult=2)

start = tr[0]
end = tr[1]

start_opt = start.xtb_geom_optimization()

end_opt = end.xtb_geom_optimization()

traj = Trajectory([start_opt, end_opt],spinmult=2)
gi = traj.run_geodesic(nimages=15)

c = Chain.from_traj(gi,parameters=ChainInputs(k=0.01))

m = MSMEP(neb_inputs=NEBInputs(grad_thre=0.001, rms_grad_thre=0.0005, v=True, early_stop_force_thre=0.05, early_stop_chain_rms_thre=0.002),gi_inputs=GIInputs(),
          chain_inputs=ChainInputs(k=0.01),
         spinmult=2)

h2.write_to_disk(Path("./alex_long_radicals"))

# +
from openbabel import pybel

ob_log_handler = pybel.ob.OBMessageHandler()
ob_log_handler.SetOutputLevel(0)
pybel.ob.obErrorLog.SetOutputLevel(0)


# -

h2,out2 = m.find_mep_multistep(c)

import matplotlib.pyplot as plt

out2[10].tdstructure.energy_xtb()

out[10].tdstructure.energy_xtb()

plt.plot(out.energies)
plt.plot(out2.energies)

h,out = m.find_mep_multistep(c)

out.plot_chain()



out.to_trajectory().write_trajectory("msmep_for_alex.xyz")

# !pwd
