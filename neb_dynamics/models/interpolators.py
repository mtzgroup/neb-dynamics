from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Dict, Any, Optional

from pydantic import BaseModel
from ..geodesic_interpolation.geodesic import run_geodesic_py
from ..elementarystep import check_if_elem_step
from ..treenode import TreeNode

from .minimizers import PathMinimizer, NodeType

MinimizerType = TypeVar("MinimizerType", bound=PathMinimizer)


class Interpolator(ABC, BaseModel, Generic[NodeType]):
    """Base class for Interpolators"""

    @abstractmethod
    def interpolate(
        self, objs: List[NodeType], number: int, **kwargs
    ) -> List[NodeType]:
        """Interpolate the path"""
        raise NotImplementedError()


class Geodesic(Interpolator[NodeType]):
    """Geodesic Interpolator"""

    params: Dict[str, Any]

    def interpolate(
        self,
        objs: List[NodeType],
        number: int,
    ) -> List[NodeType]:
        """Interpolate the path"""
        coords = run_geodesic_py(objs, number, **self.params)
