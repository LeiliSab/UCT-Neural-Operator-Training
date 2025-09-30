# UFWI/optimization/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Operator(ABC):
    """Mapping **u = F(m)** """

    @abstractmethod
    def forward(self, m: np.ndarray) -> np.ndarray:
        ...
