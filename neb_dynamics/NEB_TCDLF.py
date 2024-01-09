from dataclasses import dataclass, field
import tempfile
import subprocess
from qcparse import parse
import shutil
from pathlib import Path
from neb_dynamics.Inputs import NEBInputs
from neb_dynamics.Chain import Chain
import matplotlib.pyplot as plt

import numpy as np

from retropaths.abinitio.trajectory import Trajectory


@dataclass
class NEB_TCDLF:
    initial_chain: Chain
    parameters: NEBInputs
    
    chain_trajectory: list[Chain] = field(default_factory=list)
    scf_maxit: int = 100
    converged: bool = False

    def _create_input_file(self):

        traj = self.initial_chain.to_trajectory()
        td_ref = traj[0]

        # make the geometry file
        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w+", delete=False) as tmp:
            traj.write_trajectory(tmp.name)

        method = td_ref.tc_model_method
        basis = td_ref.tc_model_basis

        # make the tc input file
        inp = f"""
            method {method}
            basis {basis}
            
            coordinates {tmp.name}
            charge {td_ref.charge}
            
            run ts
            nstep {self.parameters.max_steps}
            timings yes
            min_image {len(traj)}
            min_coordinates cartesian
            threall 1.0e-12
            maxit {self.scf_maxit}
            scrdir {tmp.name[:-4]}
            min_tolerance {self.parameters.tol}
            ts_method neb_frozen
            
            max_nebk {self.initial_chain.parameters.k}
            min_nebk {self.initial_chain.parameters.k - self.initial_chain.parameters.delta_k}
            
            end
            """

        with tempfile.NamedTemporaryFile(
            suffix=".in", mode="w+", delete=False
        ) as tmp_inp:
            tmp_inp.write(inp)

        return tmp, tmp_inp

    def _run_tc_calc(self, tmp_inp):
        # run the tc calc
        with tempfile.NamedTemporaryFile(
            suffix=".out", mode="w+", delete=False
        ) as tmp_out:
            out = subprocess.run(
                [f"terachem {tmp_inp.name}"], shell=True, capture_output=True
            )
            tmp_out.write(out.stdout.decode())

        return tmp_out
    
    def get_ens_from_fp(self, fp):
        lines = open(fp).read().splitlines()
        atomn = int(lines[0])
        inds = list(range(1, len(lines), atomn+2))
        ens = np.array([float(line.split()[0]) for line in np.array(lines)[inds]])
        return ens

    
    def get_chain_trajectory(self, data_dir):
    
        start = self.initial_chain[0].tdstructure
        start_en = self.initial_chain[0].energy
        end = self.initial_chain[-1].tdstructure
        end_en = self.initial_chain[-1].energy
        
        
        all_paths = list(data_dir.glob("neb_*.xyz"))
        max_ind = len(all_paths)
        
        all_fps = [data_dir / f'neb_{ind+1}.xyz' for ind in range(max_ind)]
        
        img_trajs = [Trajectory.from_xyz(fp).traj for fp in all_fps]
        img_ens = [self.get_ens_from_fp(fp) for fp in all_fps]
        
        chains_imgs = list(zip(*img_trajs))
        chains_ens = list(zip(*img_ens))
        
        all_trajs = []
        for imgs in chains_imgs:
            t = Trajectory([start]+list(imgs)+[end])
            t.update_tc_parameters(start)
            all_trajs.append(t)
        
        chain_trajectory = []
        for t, raw_ens in zip(all_trajs, chains_ens):
            ens = [start_en]+list(raw_ens)+[end_en] # dlf gives only the middle image energies
            c = Chain.from_traj(t, self.initial_chain.parameters)
            for node, en in zip(c.nodes, ens):
                node._cached_energy = en
            chain_trajectory.append(c) 
        
        return chain_trajectory

    def neb_converged(self, out_path):

        text_list = open(out_path).read().splitlines()
        for line in text_list:
            if line == "Converged!":
                return True
            if line == " NOT CONVERGED":
                return False
        
        return False

    def optimize_chain(self, remove_all=True):

        tmp, tmp_inp = self._create_input_file()
        tmp_out = self._run_tc_calc(tmp_inp)

        out_tr = Trajectory.from_xyz(Path(tmp.name[:-4]) / "nebpath.xyz")
        out_tr.traj.insert(0, self.initial_chain[0].tdstructure)
        out_tr.traj.append(self.initial_chain[-1].tdstructure)
        out_tr.update_tc_parameters(self.initial_chain[0].tdstructure)
        
        converged = self.neb_converged(tmp_out.name)
        out_chain = Chain.from_traj(out_tr, parameters=self.initial_chain.parameters)
        chain_traj = self.get_chain_trajectory(Path(tmp.name[:-4]))
        
        self.chain_trajectory.extend(chain_traj)

        # remove everything
        if remove_all:
            Path(tmp.name).unlink()
            Path(tmp_inp.name).unlink()
            Path(tmp_out.name).unlink()

            shutil.rmtree(tmp.name[:-4])  # delete scratch dir

        print(f"Converged? {converged}")

        self.optimized = out_chain
        self.converged = converged
        
    def plot_opt_history(self, do_3d=False, norm_path_len=True):

        s = 8
        fs = 18
        
        if do_3d:
            all_chains = self.chain_trajectory


            ens = np.array([c.energies-c.energies[0] for c in all_chains])
            
            if norm_path_len:
                all_integrated_path_lengths = np.array([c.integrated_path_length for c in all_chains])
            else:
                all_integrated_path_lengths = np.array([c.path_length for c in all_chains])
            opt_step = np.array(list(range(len(all_chains))))
            ax = plt.figure().add_subplot(projection='3d')

            # Plot a sin curve using the x and y axes.
            x = opt_step
            ys = all_integrated_path_lengths
            zs = ens
            for i, (xind, y) in enumerate(zip(x, ys)):
                if i < len(ys) -1:
                    ax.plot([xind]*len(y), y, 'o-',zs=zs[i], color='gray',markersize=3,alpha=.1)
                else:
                    ax.plot([xind]*len(y), y, 'o-',zs=zs[i], color='blue',markersize=3)
            ax.grid(False)

            ax.set_xlabel('optimization step')
            ax.set_ylabel('integrated path length')
            ax.set_zlabel('energy (hartrees)')

            # Customize the view angle so it's easier to see that the scatter points lie
            # on the plane y=0
            ax.view_init(elev=20., azim=-45, roll=0)
            plt.tight_layout()
            plt.show()
        
        else:
            f, ax = plt.subplots(figsize=(1.16 * s, s))

            
            for i, chain in enumerate(self.chain_trajectory):
                if norm_path_len:
                    path_len = chain.integrated_path_length
                else:
                    path_len = chain.path_length
                        
                    
                if i == len(self.chain_trajectory) - 1:
                    
                    
                    plt.plot(path_len, chain.energies, "o-", alpha=1)
                else:
                    plt.plot(
                        path_len,
                        chain.energies,
                        "o-",
                        alpha=0.1,
                        color="gray",
                    )

            plt.xlabel("Integrated path length", fontsize=fs)

            plt.ylabel("Energy (kcal/mol)", fontsize=fs)
            plt.xticks(fontsize=fs)
            plt.yticks(fontsize=fs)
            plt.show()

    def write_to_disk(self, fp: Path, write_history=True):
        out_traj = self.chain_trajectory[-1].to_trajectory()
        out_traj.write_trajectory(fp)

        if write_history:
            out_folder = fp.resolve().parent / (fp.stem + "_history")

            if out_folder.exists():
                shutil.rmtree(out_folder)
                
            if not out_folder.exists():
                out_folder.mkdir()

            for i, chain in enumerate(self.chain_trajectory):
                fp = out_folder / f"traj_{i}.xyz"
                chain.write_to_disk(fp)