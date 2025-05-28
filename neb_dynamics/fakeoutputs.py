from dataclasses import dataclass
from pydantic import BaseModel


@dataclass
class FakeQCIOResults(BaseModel):
    energy: float
    gradient: list


@dataclass
class FakeQCIOOutput(BaseModel):
    """
    class that has an attribute results that has another attribute `energy` and `gradient`
    for returning previously cached properties. Class exists to add order across different
    node types that were not created with QCIO.
    """
    results: FakeQCIOResults
    success: bool = True
