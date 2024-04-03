from dataclasses import dataclass, field
import tempfile
import subprocess
from qcparse import parse
import shutil
from pathlib import Path
from neb_dynamics.Inputs import NEBInputs, ChainInputs
from neb_dynamics.Chain import Chain
import matplotlib.pyplot as plt

import numpy as np

from neb_dynamics.trajectory import Trajectory


@dataclass
class NEB_TCDLF:
    initial_chain: Chain
    parameters: NEBInputs
    
    chain_trajectory: list[Chain] = field(default_factory=list)
    scf_maxit: int = 100
    converged: bool = False
    min_images: int = None

    def __post_init__(self):
        if self.min_images is None:
            self.min_images = len(self.initial_chain)

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
            multiplicity {td_ref.spinmult}
            
            run ts
            nstep {self.parameters.max_steps}
            timings yes
            min_image {self.min_images}
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
                [f"terachem {tmp_inp.name} &> {tmp_inp.name}_output"], shell=True, capture_output=True
            )
            tmp_out.write(out.stdout.decode())

        return tmp_out
    
    @classmethod
    def get_ens_from_fp(cls, fp):
        lines = open(fp).read().splitlines()
        atomn = int(lines[0])
        inds = list(range(1, len(lines), atomn+2))
        ens = np.array([float(line.split()[0]) for line in np.array(lines)[inds]])
        return ens
    
    
    @classmethod
    def upsample_images_and_ens(cls, nested_imgs_list, nested_ens_list):
        """
        will take a list of lists that contain variable number of images, and will upsample them to tot_number 
        if they are below the number. 
        """
        tot_number = cls.get_number_to_upsample_to(nested_imgs_list)
        
        
        output_list = []
        output_ens = []
        for l, l_ens in zip(nested_imgs_list, nested_ens_list):
            if len(l) < tot_number:
                n_to_add = tot_number - len(l)
                new_list = l
                new_list_ens = list(l_ens)
                
                new_list.extend([l[-1]]*n_to_add)
                new_list_ens.extend([l_ens[-1]]*n_to_add)
                
                
                
                output_list.append(new_list)
                output_ens.append(new_list_ens)
            else:
                output_list.append(l)
                output_ens.append(l_ens)
        return output_list, output_ens
            
    
    @classmethod
    def get_number_to_upsample_to(cls, nested_imgs_list):
        max_n = 0
        for l in nested_imgs_list:
            if len(l) > max_n:
                max_n = len(l)
        return max_n
    

    @classmethod
    def get_chain_trajectory(cls, data_dir, parameters):
        
        all_paths = list(data_dir.glob("neb_*.xyz"))
        max_ind = len(all_paths)
        
        all_fps = [data_dir / f'neb_{ind}.xyz' for ind in range(1, max_ind+1)]
        
        # print(f"\n{all_paths=}\n{max_ind=}\n{all_fps=}")
        
        img_trajs = [Trajectory.from_xyz(fp).traj for fp in all_fps]
        img_ens = [cls.get_ens_from_fp(fp) for fp in all_fps]
        img_trajs, img_ens = cls.upsample_images_and_ens(img_trajs, img_ens)
        
        
        
        chains_imgs = list(zip(*img_trajs))
        chains_ens = list(zip(*img_ens))
        
        start = img_trajs[0][0]
        start_en = img_ens[0][0]
        end = img_trajs[-1][0]
        end_en = img_ens[-1][0]
        
        
        
        all_trajs = []
        for imgs in chains_imgs:
            t = Trajectory([start]+list(imgs)+[end])
            t.update_tc_parameters(start)
            all_trajs.append(t)
                    
        chain_trajectory = []
        for t, raw_ens in zip(all_trajs, chains_ens):
            ens = [start_en]+list(raw_ens)+[end_en] # dlf gives only the middle image energies
            c = Chain.from_traj(t, parameters)
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
        print(f"scr dir : {Path(tmp.name[:-4])}")
        chain_traj = self.get_chain_trajectory(Path(tmp.name[:-4]), parameters=self.initial_chain.parameters)
        
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
                
                
                
    def _chain_converged(self, chain_prev: Chain, chain_new: Chain) -> bool:
        """
        https://chemshell.org/static_files/py-chemshell/manual/build/html/opt.html?highlight=nudged
        """
        rms_grad_conv_ind, max_rms_grads = self._check_rms_grad_converged(chain_new)
        en_converged_indices, en_deltas = self._check_en_converged(
            chain_prev=chain_prev, chain_new=chain_new
        )

        grad_conv_ind, max_grad_components = self._check_grad_converged(chain=chain_new)

        converged_nodes_indices = np.intersect1d(
            en_converged_indices, rms_grad_conv_ind
        )
        converged_nodes_indices = np.intersect1d(converged_nodes_indices, grad_conv_ind)

        if chain_new.parameters.node_freezing:
            self._update_node_convergence(chain=chain_new, indices=converged_nodes_indices, prev_chain=chain_prev)
            self._copy_node_information_to_converged(new_chain=chain_new, old_chain=chain_prev)
            
        if self.parameters.v > 1:
            print("\n")
            [
                print(
                    f"\t\tnode{i} | ∆E : {en_deltas[i]} | Max(RMS Grad): {max_rms_grads[i]} | Max(Grad components): {max_grad_components[i]} | Converged? : {chain_new.nodes[i].converged}"
                )
                for i in range(len(chain_new))
            ]
        if self.parameters.v > 1:
            print(f"\t{len(converged_nodes_indices)} nodes have converged")
            
        barrier_height_converged = self._check_barrier_height_conv(chain_prev=chain_prev, chain_new=chain_new)
        ind_ts_guess = np.argmax(chain_new.energies)
        ts_guess = chain_new[ind_ts_guess]
        
        criteria_converged = [
            max(max_rms_grads) <= self.parameters.rms_grad_thre,
            # max(en_deltas) <= self.parameters.en_thre,
            # max(max_grad_components) <=self.parameters.grad_thre,
            np.amax(np.abs(ts_guess.gradient)) <= self.parameters.grad_thre,
            barrier_height_converged]
        # return len(converged_nodes_indices) == len(chain_new)

        # return sum(criteria_converged) >= 2
        return sum(criteria_converged) >= 3
    
    @classmethod
    def read_from_disk(cls, fp: Path, neb_inputs: NEBInputs, chain_inputs: ChainInputs):
        
        ct = cls.get_chain_trajectory(data_dir=fp, parameters=chain_inputs)
        
        obj = cls(initial_chain=ct[0], parameters=neb_inputs)
        obj.chain_trajectory = ct
        obj.optimized = ct[-1]
        return obj
