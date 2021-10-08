from __future__ import annotations

from typing import Optional

import numpy as np

from . import operations as so3op


class Rotation:
  """Light weight rotation class."""

  def __init__(self,
               *,
               rotation: Optional[Rotation] = None,
               C_ba: Optional[np.ndarray] = None,
               aaxis_ab: Optional[np.ndarray] = None,
               num_terms: int = 0) -> None:
    """Constructor with several options.
    Args:
      rotation (Optional[Rotation]): an existing instance of this class. (copy construction)
      C_ba (Optional[np.ndarray]): constructs from a 3x3 rotation matrix
      aaxis_ab (Optional[np.ndarray]): constructs using the exponential map from a 3x1 axis-angle vector
      num_terms (int): number of terms in the infintie series when converting aaxis_ab to C_ba; defaults to 0
        (analytical)
    """

    # copy construction
    if rotation is not None:
      self._C_ba = np.copy(rotation.matrix())
      return

    # constructs from a 3x3 rotation matrix
    if C_ba is not None:
      self._C_ba = np.copy(C_ba)
      self.reproject()
      return

    # construct using the exponential map from a 3x1 axis-angle vector
    if aaxis_ab is not None:
      assert aaxis_ab.shape[-2:] == (3, 1), "aaxis_ab has invalid shape."
      self._C_ba = so3op.vec2rot(aaxis_ab, num_terms)
      return

    # default construction
    self._C_ba = np.eye(3)

  def matrix(self) -> np.ndarray:
    """Returns the underlying rotation matrix representation."""
    return self._C_ba

  def vec(self) -> np.ndarray:
    """Returns the corresponding Lie algebra using the logarithmic map."""
    return so3op.rot2vec(self._C_ba)

  def inverse(self) -> Rotation:
    """Returns the inverse rotation."""
    temp = Rotation(C_ba=self._C_ba.swapaxes(-2, -1))
    temp.reproject()
    return temp

  def reproject(self) -> None:
    """Reprojects the rotation matrix back onto SO(3)."""
    self._C_ba = so3op.vec2rot(so3op.rot2vec(self._C_ba))

  def __matmul__(self, other: Rotation) -> Rotation:
    return Rotation(C_ba=self.matrix() @ other.matrix())

  def __mul__(self, other: Rotation) -> Rotation:
    return Rotation(C_ba=self.matrix() @ other.matrix())

  def __str__(self):
    return str(self.matrix())