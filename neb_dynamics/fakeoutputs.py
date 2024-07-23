from dataclasses import dataclass
import numpy as np

@dataclass
class FakeQCIOResults:
    energy: float
    gradient: np.array

@dataclass
class FakeQCIOOutput:
    """
    class that has an attribute results that has another attribute `energy` and `gradient`
    for returning previously cached properties. Class exists to add order across different
    node types that were not created with QCIO.
    """
    results: FakeQCIOResults


