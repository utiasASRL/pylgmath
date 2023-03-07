from __future__ import annotations

from typing import Optional

import numpy as np
from . import operations as tmop

class TimeMachine:
  """Light weight time machine class for SE_2(3) operations."""

  def __init__(self,
               *,
               time_machine: Optional[Transformation] = None,
               tau: Optional[np.ndarray] = None) -> None:
    """Constructor with several options.
    Args:
      time_machine (Optional[TimeMachine]): an existing instance of this class. (copy construction)
      tau (Optional[np.ndarray]): constructs from a time increment
    """

    # copy construction
    if time_machine is not None:
      self._tau = np.copy(time_machine.tau())
      return

    # constructs from a time increment
    if tau is not None:
      self._tau = np.copy(tau)
      return

    # default construction
    self._tau: np.ndarray = np.zeros((1))

  def assign(self,
             tau) -> None:
    """Assigns a new transformation to this class. (operator= in c++)
    Args:
      tau: constructs from a time increment
    """
    assert tau is not None
    self._tau = tau

  def matrix(self) -> np.ndarray:
    """Returns the transformation matrix representation."""
    D = np.zeros(self._tau.shape + (5, 5))
    D[..., range(5), range(5)] = 1
    D[..., 4, 3] = self._tau
    return D

  def tau(self) -> np.ndarray:
    """Returns the underlying time constant."""
    return self._tau

  def vec(self) -> np.ndarray:
    """Returns the corresponding Lie algebra using the logarithmic map."""
    return tmop.tran2vec(self.matrix())

  def inverse(self) -> Transformation:
    """Returns the inverse transformation."""
    temp = TimeMachine(tau=-self._tau)
    return temp

  def __matmul__(self, other) -> Transformation:
    return TimeMachine(tau = self.tau() + other.tau())

  def __mul__(self, other) -> Transformation:
    return TimeMachine(tau = self.tau() + other.tau())

  def __str__(self):
    return str(self.matrix())
