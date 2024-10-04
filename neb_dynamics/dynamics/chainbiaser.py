from dataclasses import dataclass
import numpy as np
from joblib import Parallel, delayed
from typing import List
from neb_dynamics.nodes.node import Node
import neb_dynamics.chainhelpers as ch
from neb_dynamics.nodes.node import StructureNode, XYNode
from neb_dynamics.helper_functions import RMSD


@dataclass
class ChainBiaser:
    AVAIL_DIST_FUNC_NAMES = ["per_node", "simp_frechet"]
    reference_chains: list  # list of Chain objects

    amplitude: float = 1.0
    sigma: float = 1.0

    distance_func: str = "simp_frechet"

    def distance_node_wise(self, chain):
        if self.distance_func == "per_node":
            tot_dist = 0
            for reference_chain in self.reference_chains:
                for n1, n2 in zip(
                    chain.coordinates[1:-1], reference_chain.coordinates[1:-1]
                ):
                    tot_dist += RMSD(n1, n2)[0]

        elif self.distance_func == "simp_frechet":
            tot_dist = 0
            # tot_dist = sum(self._simp_frechet_helper(coords) for coords in chain.coordinates[1:-1])
            for ref_chain in self.reference_chains:
                tot_dist += self.frechet_distance(
                    path1=chain.coordinates, path2=ref_chain.coordinates
                )
        else:
            raise ValueError(
                f"Invalid distance func name: {self.distance_func}. Available are: {self.AVAIL_DIST_FUNC_NAMES}"
            )

        return tot_dist / len(chain)

    def compute_euclidean_distance(self, p1, p2):
        """Calculates the Euclidean distance between two points p1 and p2."""

        if isinstance(p1, XYNode):
            p1 = p1.structure
            p2 = p2.structure
        elif isinstance(p1, StructureNode):
            p1, p2 = ch._get_mass_weighed_coords([p1, p2])

        return np.linalg.norm(np.array(p1) - np.array(p2))

    def compute_rmsd(self, p1, p2):
        """Calculates the RMSD distance between two points p1 and p2."""

        if isinstance(p1, XYNode):
            raise ValueError("Cannot compute RMSD for 2D potential.")
        elif isinstance(p1, StructureNode):
            dist, _ = RMSD(p1.coords, p2.coords)

        return dist

    def frechet_distance(self, path1, path2):
        """Calculates the Fréchet distance between two paths."""
        n, m = len(path1), len(path2)
        ca = np.full((n, m), -1.0)  # Memoization table

        def recursive_calculation(i, j):
            if ca[i, j] > -1:
                return ca[i, j]
            elif i == 0 and j == 0:
                ca[i, j] = self.compute_euclidean_distance(path1[0], path2[0])
            elif i > 0 and j == 0:
                ca[i, j] = max(
                    recursive_calculation(i - 1, 0),
                    self.compute_euclidean_distance(path1[i], path2[0]),
                )
            elif i == 0 and j > 0:
                ca[i, j] = max(
                    recursive_calculation(0, j - 1),
                    self.compute_euclidean_distance(path1[0], path2[j]),
                )
            elif i > 0 and j > 0:
                ca[i, j] = max(
                    min(
                        recursive_calculation(i - 1, j),
                        recursive_calculation(i - 1, j - 1),
                        recursive_calculation(i, j - 1),
                    ),
                    self.compute_euclidean_distance(path1[i], path2[j]),
                )
            else:
                ca[i, j] = float("inf")
            return ca[i, j]

        return recursive_calculation(n - 1, m - 1)

    def _simp_frechet_helper(self, coords):
        node_distances = np.array(
            [
                min(
                    (
                        np.linalg.norm(coords - ref_coord)
                        if len(ref_coord.shape) == 1
                        else RMSD(coords, ref_coord)[0]
                    )
                    for ref_coord in reference_chain.coordinates[1:-1]
                )
                for reference_chain in self.reference_chains
            ]
        )
        return node_distances.min()

    def energy_gaussian_bias(self, distance):
        return self.amplitude * np.exp(-(distance**2) / (2 * self.sigma**2))

    def compute_min_dist_to_ref(self, dist_func, node: Node, reference: List[Node]):
        return min([dist_func(node, ref) for ref in reference])

    def energy_chain_bias(self, chain):
        dist_to_chain = self.distance_node_wise(chain)
        return self.energy_gaussian_bias(dist_to_chain)

    def gradient_node_bias(self, node: Node, reference_chains: List[List[Node]] = None, dr=0.1):
        """
        computes the gradient acting on `node` from `reference` nodes.
        """
        n_atoms = len(node.coords)
        if reference_chains is None:
            reference_chains = self.reference_chains

        if isinstance(node, StructureNode):
            shape = (n_atoms, 3)
            directions = ["x", "y", "z"]
        elif isinstance(node, XYNode):
            shape = (n_atoms,)
            directions = ["x", "y"]
        else:
            raise ValueError(f"invalid input node type: {node}")
        final_grads = np.zeros(shape=shape)
        for reference in reference_chains:
            grads = np.zeros(shape=shape)
            if isinstance(node, StructureNode):
                orig_dist = self.compute_min_dist_to_ref(
                    dist_func=self.compute_rmsd,
                    node=node,
                    reference=reference,
                )
                for i in range(n_atoms):
                    for j in range(len(directions)):
                        disp_vector = np.zeros(shape=shape)
                        disp_vector[i, j] = dr

                        displaced_coord = node.coords + disp_vector
                        node_disp_direction = node.update_coords(displaced_coord)
                        new_dist = self.compute_min_dist_to_ref(
                            dist_func=self.compute_rmsd,
                            node=node_disp_direction,
                            reference=reference,
                        )
                        grad_direction = self.energy_gaussian_bias(
                            new_dist
                        ) - self.energy_gaussian_bias(orig_dist)
                        grads[i, j] = grad_direction

                grads = grads / dr

            elif isinstance(node, XYNode):
                orig_dist = self.compute_min_dist_to_ref(
                    dist_func=self.compute_euclidean_distance,
                    node=node,
                    reference=reference,
                )
                for i in range(n_atoms):

                    disp_vector = np.zeros(shape=shape)
                    disp_vector[i] = dr

                    displaced_coord = node.coords + disp_vector
                    node_disp_direction = node.update_coords(displaced_coord)
                    new_dist = self.compute_min_dist_to_ref(
                        dist_func=self.compute_euclidean_distance,
                        node=node_disp_direction,
                        reference=reference,
                    )
                    grad_direction = self.energy_gaussian_bias(
                        new_dist
                    ) - self.energy_gaussian_bias(orig_dist)
                    grads[i] = grad_direction

                grads = grads / dr

            final_grads += grads
        return final_grads

    def gradient_node_in_chain_bias(self, chain, node, ind_node, dr=0.1):
        n_atoms = len(node.coords)
        if isinstance(node, StructureNode):
            shape = (n_atoms, 3)
            directions = ["x", "y", "z"]
        elif isinstance(node, XYNode):
            shape = (n_atoms,)
            directions = ["x", "y"]
        else:
            raise ValueError(f"invalid input node type: {node}")
        grads = np.zeros(shape=shape)
        if isinstance(node, StructureNode):
            for i in range(n_atoms):
                for j in range(len(directions)):
                    disp_vector = np.zeros(shape=shape)
                    disp_vector[i, j] = dr

                    displaced_coord = node.coords + disp_vector
                    node_disp_direction = node.update_coords(displaced_coord)
                    fake_chain = chain.copy()
                    fake_chain.nodes[ind_node] = node_disp_direction

                    grad_direction = self.energy_chain_bias(
                        fake_chain
                    ) - self.energy_chain_bias(chain)
                    grads[i, j] = grad_direction

            return grads / dr
        elif isinstance(node, XYNode):
            for i in range(n_atoms):

                disp_vector = np.zeros(shape=shape)
                disp_vector[i] = dr

                displaced_coord = node.coords + disp_vector
                node_disp_direction = node.update_coords(displaced_coord)
                fake_chain = chain.copy()
                fake_chain.nodes[ind_node] = node_disp_direction

                grad_direction = self.energy_chain_bias(
                    fake_chain
                ) - self.energy_chain_bias(chain)
                grads[i] = grad_direction

            return grads / dr

    def gradient_chain_bias(self, chain):
        grad_bias = grad_chain_bias_function(chain, self.gradient_node_in_chain_bias)
        if isinstance(chain[0], StructureNode):
            mass_weights = ch._get_mass_weights(chain)
            mass_weights = mass_weights.reshape(-1, 1)
        else:
            mass_weights = 1.0 * len(chain)

        energy_weights = chain.energies_kcalmol[1:-1]
        grad_bias = grad_bias * mass_weights
        out_arr = [grad * weight for grad, weight in zip(grad_bias, energy_weights)]
        grad_bias = np.array(out_arr)

        return grad_bias


def grad_chain_bias_function(chain, grad_node_bias_fn):
    return np.array(
        Parallel(n_jobs=-1)(
            delayed(grad_node_bias_fn)(chain, node, ind_node)
            for ind_node, node in enumerate(chain[1:-1], start=1)
        )
    )
