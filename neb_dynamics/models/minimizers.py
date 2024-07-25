from abc import ABC, abstractmethod
from pydantic import BaseModel, model_validator
from typing import List, TypeVar, Generic

import numpy as np

from ..nodes.node import Node
from ..engines.engine import Engine
from ..optimizer import Optimizer
from ..inputs import Parameters, NEBInputs


EngineType = TypeVar("EngineType", bound=Engine)
NodeType = TypeVar("NodeType", bound=Node)
OptimizerType = TypeVar("OptimizerType", bound=Optimizer)
ParametersType = TypeVar("ParametersType", bound=Parameters)


class PathMinimizer(
    ABC, BaseModel, Generic[NodeType, EngineType, OptimizerType, ParametersType]
):
    """Base class for Path Minimizers"""

    trajectory: List[List[NodeType]]
    engine: EngineType
    optimizer: OptimizerType
    parameters: ParametersType

    @abstractmethod
    def end_run(self) -> bool:
        """Check if the Run is Complete"""
        raise NotImplementedError()

    def run(self) -> None:
        """Run the Path Minimization"""

        while not self.end_run():
            # Compute Gradients
            self.engine.compute_gradients(self.trajectory[-1])

            # Compute Energies
            self.engine.compute_energies(self.trajectory[-1])

            # Create modified gradients
            mod_grads = self.generate_modified_gradients(self.trajectory[-1])

            # Optimize the Path. Returns a new chain
            coords = [node.coords for node in self.trajectory[-1]]
            new_chain = self.optimizer.take_step(coords, mod_grads)

            # Append the new chain to the trajectory
            self.trajectory.append(new_chain)

    def generate_modified_gradients(chain: List[NodeType]) -> List[np.ndarray]:
        """Generate the Modified Gradients"""
        for obj in chain:
            assert obj.gradient is not None, "Gradients must be computed"
            assert obj.energy is not None, "Energies must be computed"

        raise NotImplementedError()


class NEB(
    PathMinimizer[NEBInputs],
    Generic[NodeType, EngineType, OptimizerType],
):
    """Nudged Elastic Band (NEB) Minimizer"""

    @model_validator(mode="after")
    def ensure_endpoints_not_same(self):
        """Ensure that the endpoints are not the same"""
        assert self.trajectory[0] != self.trajectory[-1], "Endpoints must be different"

    def is_converged(self) -> bool:
        """Check if the NEB is converged"""

        raise NotImplementedError()

    def early_terminate(self) -> bool:
        """Check if the NEB should be terminated early"""

        raise NotImplementedError()

    def end_run(self) -> bool:
        """Check if the NEB is converged"""

        # Check if the Path is Converged
        if self.is_converged():
            return True

        # Check if the NEB should be Terminated Early
        if self.early_terminate():
            return True

        # Check if the Maximum Number of Steps has been Reached
        return len(self.trajectory) > self.parameters.max_steps

    def generate_modified_gradients(self, chain: List[NodeType]) -> List[np.ndarray]:
        """Generate the NEB Gradients"""

        raise NotImplementedError()
