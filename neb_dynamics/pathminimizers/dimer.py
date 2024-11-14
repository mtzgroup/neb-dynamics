import numpy as np
from neb_dynamics.nodes.nodehelpers import displace_by_dr


class DimerMethod3D:
    def __init__(self, initial_node,
                 next_node,
                 prev_node,
                 engine,
                 dimer_length=0.01, rotation_tolerance=1e-3, translation_tolerance=1e-3, max_rotations=50, max_translations=200, learning_rate=0.1, max_steps=100):
        """
        Initialize Dimer Method parameters.

        Parameters:
        - initial_position: Initial position in the search space.
        - dimer_length: Separation distance between dimer images.
        - rotation_tolerance: Tolerance for the rotation convergence.
        - translation_tolerance: Tolerance for the translation convergence.
        - max_rotations: Maximum number of iterations for rotation.
        - max_translations: Maximum number of translation steps.
        - learning_rate: Step size for translation.
        """
        self.node = initial_node

        self.next_node = next_node
        self.prev_node = prev_node
        self.dimer_length = dimer_length
        self.rotation_tolerance = rotation_tolerance
        self.translation_tolerance = translation_tolerance
        self.max_rotations = max_rotations
        self.max_translations = max_translations
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.engine = engine

        self.traj = []

    def energy(self, node):
        """
        Placeholder function for calculating energy at a given position.
        Replace with actual energy calculation.
        """
        return self.engine.compute_energies([node])

    def gradient(self, node):
        """
        Placeholder function for calculating gradient at a given position.
        Replace with actual gradient calculation.
        """
        return self.engine.compute_gradients([node])

    def rotate_dimer(self, dimer_vector):
        """
        Rotate dimer to align with the lowest curvature mode.
        """

        for _ in range(self.max_rotations):
            grad1 = self.gradient(
                displace_by_dr(self.node, displacement=dimer_vector,
                               dr=self.dimer_length / 2))
            grad2 = self.gradient(
                displace_by_dr(self.node, displacement=-dimer_vector,
                               dr=self.dimer_length / 2))
            force_perpendicular = grad1 - grad2  # Difference of gradients

            torque = np.cross(dimer_vector, force_perpendicular)[0]
            rotation_direction = -torque / np.linalg.norm(torque)
            print(f"{torque=}   {rotation_direction=}")
            dimer_vector += 0.01 * rotation_direction  # Small step rotation
            dimer_vector /= np.linalg.norm(dimer_vector)  # Normalize

            # Check rotation convergence
            if np.linalg.norm(torque) < self.rotation_tolerance:
                break

        return dimer_vector

    def translate_dimer(self, dimer_vector):
        """
        Translate the dimer along the rotated vector.
        """
        for _ in range(self.max_translations):
            grad = self.gradient(self.node)
            grad_parallel = np.dot(
                grad.flatten(), dimer_vector.flatten()) * dimer_vector
            grad_perpendicular = grad - grad_parallel

            # Update position by moving against the perpendicular component of the gradient
            new_pos = displace_by_dr(
                self.node, displacement=grad_perpendicular, dr=-self.learning_rate)
            new_pos = displace_by_dr(
                new_pos, displacement=grad_parallel, dr=self.learning_rate)
            self.traj.append(new_pos)
            self.node = new_pos

            # Check translation convergence
            if np.linalg.norm(grad_perpendicular) < self.translation_tolerance:
                break

    def find_transition_state(self):
        """
        Run the Dimer Method to find the transition state.
        """
        dimer_vector = self.next_node.coords - self.prev_node.coords
        # Normalize the initial dimer vector
        dimer_vector /= np.linalg.norm(dimer_vector)

        for _ in range(self.max_steps):
            dimer_vector = self.rotate_dimer(dimer_vector)
            self.translate_dimer(dimer_vector)

        return self.node
