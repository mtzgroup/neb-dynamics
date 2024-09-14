from dataclasses import dataclass
import numpy as np
from joblib import Parallel, delayed
import neb_dynamics.chainhelpers as ch


@dataclass
class ChainBiaser:
    AVAIL_DIST_FUNC_NAMES = ["per_node", "simp_frechet"]
    reference_chains: list  # list of Chain objects

    amplitude: float = 1.0
    sigma: float = 1.0

    distance_func: str = "simp_frechet"

    def node_wise_distance(self, chain):
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

    def euclidean_distance(self, p1, p2):
        """Calculates the Euclidean distance between two points p1 and p2."""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def frechet_distance(self, path1, path2):
        """Calculates the Fréchet distance between two paths."""
        n, m = len(path1), len(path2)
        ca = np.full((n, m), -1.0)  # Memoization table

        def recursive_calculation(i, j):
            if ca[i, j] > -1:
                return ca[i, j]
            elif i == 0 and j == 0:
                ca[i, j] = self.euclidean_distance(path1[0], path2[0])
            elif i > 0 and j == 0:
                ca[i, j] = max(
                    recursive_calculation(i - 1, 0),
                    self.euclidean_distance(path1[i], path2[0]),
                )
            elif i == 0 and j > 0:
                ca[i, j] = max(
                    recursive_calculation(0, j - 1),
                    self.euclidean_distance(path1[0], path2[j]),
                )
            elif i > 0 and j > 0:
                ca[i, j] = max(
                    min(
                        recursive_calculation(i - 1, j),
                        recursive_calculation(i - 1, j - 1),
                        recursive_calculation(i, j - 1),
                    ),
                    self.euclidean_distance(path1[i], path2[j]),
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

    def path_bias(self, distance):
        return self.amplitude * np.exp(-(distance**2) / (2 * self.sigma**2))

    def chain_bias(self, chain):
        dist_to_chain = self.node_wise_distance(chain)
        return self.path_bias(dist_to_chain)

    def grad_node_bias(self, chain, node, ind_node, dr=0.1):
        directions = ["x", "y", "z"]
        n_atoms = len(node.coords)
        grads = np.zeros((n_atoms, len(directions)))

        for i in range(n_atoms):
            for j in range(len(directions)):
                disp_vector = np.zeros((n_atoms, len(directions)))
                disp_vector[i, j] = dr

                displaced_coord = node.coords + disp_vector
                node_disp_direction = node.update_coords(displaced_coord)
                fake_chain = chain.copy()
                fake_chain.nodes[ind_node] = node_disp_direction

                grad_direction = self.chain_bias(fake_chain) - self.chain_bias(chain)
                grads[i, j] = grad_direction

        return grads / dr

    def grad_chain_bias(self, chain):
        grad_bias = grad_chain_bias_function(chain, self.grad_node_bias)
        mass_weights = ch._get_mass_weights(chain)
        energy_weights = chain.energies_kcalmol[1:-1]
        grad_bias = grad_bias * mass_weights.reshape(-1, 1)
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
