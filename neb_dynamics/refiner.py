from dataclasses import dataclass, field
from neb_dynamics.nodes.Node3D_TC import Node3D_TC  # , Node3D_TC_Local, Node3D_TC_TCPB
from neb_dynamics.Inputs import NEBInputs, ChainInputs, GIInputs
from neb_dynamics.constants import BOHR_TO_ANGSTROMS
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.Optimizer import Optimizer
from neb_dynamics.optimizers.VPO import VelocityProjectedOptimizer
from chain import Chain
from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.NEB import NEB

import numpy as np

tol = 0.001


@dataclass
class Refiner:
    method: str = "wb97xd3"
    basis: str = "def2-svp"
    kwds: dict = field(default_factory=dict)
    v: bool = False

    cni: ChainInputs = None

    nbi: NEBInputs = None

    optimizer: Optimizer = None

    gii: GIInputs = None

    resample_chain: bool = True

    def __post_init__(self):
        if self.cni is None:
            self.cni = ChainInputs(
                k=0.1,
                delta_k=0.09,
                node_class=Node3D_TC,
                friction_optimal_gi=False,
                do_parallel=True,
                node_freezing=True,
                node_conf_en_thre=0.1,
            )

        if self.nbi is None:
            self.nbi = NEBInputs(
                tol=tol * BOHR_TO_ANGSTROMS,
                v=1,
                max_steps=500,
                early_stop_chain_rms_thre=0.1,
                early_stop_force_thre=0.0001,
                early_stop_still_steps_thre=100,
                preopt_with_xtb=0,
            )

        if self.optimizer is None:
            self.optimizer = VelocityProjectedOptimizer(
                timestep=1.0, activation_tol=0.1
            )

        if self.gii is None:
            self.gii = GIInputs(nimages=12)

    def refine_xtb_chain(self, xtb_chain):

        init_c = self.convert_to_dft(xtb_chain)
        m = MSMEP(
            neb_inputs=self.nbi,
            chain_inputs=self.cni,
            gi_inputs=self.gii,
            optimizer=self.optimizer,
        )

        h_dft, out_dft = m.find_mep_multistep(init_c)
        return h_dft

    def convert_to_dft(self, xtb_chain: Chain):

        out_xtb = xtb_chain

        out_tr = out_xtb.to_trajectory()
        ref = out_tr[0]
        ref.tc_model_method = self.method
        ref.tc_model_basis = self.basis
        ref.tc_kwds = self.kwds
        out_tr.update_tc_parameters(ref)
        out_dft = Chain.from_traj(out_tr, parameters=self.cni)

        if self.resample_chain:
            out_dft, n_grad_calls = out_dft.resample_chain(
                out_dft,
                n=len(out_dft),
                method=self.method,
                basis=self.basis,
                kwds=self.kwds,
            )

        return out_dft

    def create_refined_leaves(self, seed_leaves):
        refined_leaves = []
        for leaf in seed_leaves:
            refined_leaf = self.refine_xtb_chain(leaf.data.optimized)
            refined_leaves.append(refined_leaf)
        return refined_leaves

    def join_output_leaves(self, refined_leaves):
        joined_nodes = []
        [
            joined_nodes.extend(leaf.output_chain.nodes)
            for leaf in refined_leaves
            if leaf
        ]
        joined_chain = Chain(nodes=joined_nodes, parameters=self.cni)
        return joined_chain

    def write_leaves_to_disk(self, out_directory, refined_leaves):
        out_directory.mkdir(exist_ok=True)

        for i, leaf in enumerate(refined_leaves):
            leaf.write_to_disk(out_directory / f"leaf_{i}")

    def read_leaves_from_disk(self, out_directory):
        read_in_leaves = []
        num_steps = []
        for i, _ in enumerate(list(out_directory.glob("*"))):
            fp = out_directory / f"leaf_{i}"
            try:
                tn = TreeNode.read_from_disk(fp)
            except IndexError:
                neb_obj = NEB.read_from_disk(fp / "node_0.xyz")
                tn = TreeNode(data=neb_obj, children=[], index=0)
            read_in_leaves.append(tn)
            num_steps.append(tn.get_num_opt_steps())
        if self.v:
            print("opt steps:", sum(num_steps))
        return read_in_leaves

    def output_chain_from_dir(self, out_directory):
        leaves = self.read_leaves_from_disk(out_directory)
        out_chain = self.join_output_leaves(leaves)
        return out_chain
