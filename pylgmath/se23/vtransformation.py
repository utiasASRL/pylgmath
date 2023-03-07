from __future__ import annotations

from typing import Optional

import numpy as np

from ..so3 import operations as so3op
from ..se3 import operations as se3op
from . import operations as se23op


class VTransformation:
  """Light weight transformation + velocity class."""

  def __init__(self,
               *,
               vtransformation: Optional[VTransformation] = None,
               T_ba: Optional[np.ndarray] = None,
               C_ba: Optional[np.ndarray] = None,
               r_ba_in_a: Optional[np.ndarray] = None,
               v_ba_in_a: Optional[np.ndarray] = None,
               xi_ab: Optional[np.ndarray] = None,
               num_terms: int = 0) -> None:
    """Constructor with several options.
    Args:
      vtransformation (Optional[VTransformation]): an existing instance of this class. (copy construction)
      T_ba (Optional[np.ndarray]): constructs from a 4x5 transformation matrix
      C_ba (Optional[np.ndarray]): constructs from a 3x3 rotation matrix, a 3x1 translation vector and a 3x1 translation vector
      r_ba_in_a (Optional[np.ndarray]): constructs from a 3x3 rotation matrix, a 3x1 translation vector and a 3x1 translation vector
      v_ba_in_a (Optional[np.ndarray]): constructs from a 3x3 rotation matrix, a 3x1 translation vector and a 3x1 translation vector
      xi_ab (Optional[np.ndarray]): constructs using the exponential map from a 9x1 se23 algebra vector
      num_terms (int): number of terms in the infintie series when converting xi_ab to T_ba; defaults to 0 (analytical)
    """

    # copy construction
    if vtransformation is not None:
      self._C_ba = np.copy(vtransformation.C_ba())
      self._r_ab_inb = np.copy(vtransformation.r_ab_inb())
      self._v_ab_inb = np.copy(vtransformation.v_ab_inb())
      return

    # constructs from a 4x4 transformation matrix
    if T_ba is not None:
      self._C_ba = np.copy(T_ba[..., 0:3, 0:3])
      self._r_ab_inb = np.copy(T_ba[..., 0:3, 3:4])
      self._v_ab_inb = np.copy(T_ba[..., 0:3, 4:5])
      return

    # constructs from a 3x3 rotation matrix, a 3x1 translation vector and a 3x1 velocity vector

    if (C_ba is not None) and (r_ba_in_a is not None) and (v_ba_in_a is not None):
      assert C_ba.shape[:-2] == r_ba_in_a.shape[:-2] and C_ba.shape[:-2] == v_ba_in_a.shape[:-2]
      self._C_ba = np.copy(C_ba)
      self._r_ab_inb = -C_ba @ r_ba_in_a
      self._v_ab_inb = -C_ba @ v_ba_in_a
      self.reproject()
      return

    # constructs using the exponential map from a 9x1 se3 algebra vector
    if xi_ab is not None:
      assert xi_ab.shape[-2:] == (9, 1), f"xi_ab has invalid shape of {xi_ab.shape[-2:]}"
      self._C_ba, self._r_ab_inb, self._v_ab_inb = se23op.vec2tran(xi_ab, num_terms, False)
      return

    # default construction
    self._C_ba: np.ndarray = np.eye(3)
    self._r_ab_inb: np.ndarray = np.zeros((3, 1))
    self._v_ab_inb: np.ndarray = np.zeros((3, 1))

  def assign(self,
             *,
             T_ba: Optional[np.ndarray] = None,
             C_ba: Optional[np.ndarray] = None,
             r_ba_in_a: Optional[np.ndarray] = None,
             v_ba_in_a: Optional[np.ndarray] = None,
             xi_ab: Optional[np.ndarray] = None,
             num_terms: int = 0) -> None:
    """Assigns a new transformation to this class. (operator= in c++)
    Args:
      T_ba (Optional[np.ndarray]): constructs from a 4x5 transformation matrix
      C_ba (Optional[np.ndarray]): constructs from a 3x3 rotation matrix, a 3x1 translation vector and a 3x1 translation vector
      r_ba_in_a (Optional[np.ndarray]): constructs from a 3x3 rotation matrix, a 3x1 translation vector and a 3x1 translation vector
      v_ba_in_a (Optional[np.ndarray]): constructs from a 3x3 rotation matrix, a 3x1 translation vector and a 3x1 translation vector
      xi_ab (Optional[np.ndarray]): constructs using the exponential map from a 9x1 se23 algebra vector
      num_terms (int): number of terms in the infintie series when converting xi_ab to T_ba; defaults to 0 (analytical)
    """

    if T_ba is not None:
      self._C_ba[:] = T_ba[..., 0:3, 0:3]
      self._r_ab_inb[:] = T_ba[..., 0:3, 3:4]
      self._v_ab_inb[:] = T_ba[..., 0:3, 4:5]
      return

    if (C_ba is not None) and (r_ba_in_a is not None) and (v_ba_in_a is not None):
      assert C_ba.shape[:-2] == r_ba_in_a.shape[:-2] and C_ba.shape[:-2] == v_ba_in_a.shape[:-2]
      self._C_ba = np.copy(C_ba)
      self._r_ab_inb = -C_ba @ r_ba_in_a
      self.reproject()
      return

    if xi_ab is not None:
      assert xi_ab.shape[-2:] == (9, 1), "xi_ab has invalid shape."
      self._C_ba[:], self._r_ab_inb[:] = se23op.vec2tran(xi_ab, num_terms, False)
      return

  def matrix(self) -> np.ndarray:
    """Returns the transformation matrix representation."""
    T_ba = np.zeros(self._C_ba.shape[:-2] + (5, 5))
    T_ba[..., :3, :3] = self._C_ba
    T_ba[..., :3, 3:4] = self._r_ab_inb
    T_ba[..., :3, 4:5] = self._v_ab_inb
    T_ba[..., 3, 3] = 1
    T_ba[..., 4, 4] = 1
    return T_ba

  def C_ba(self) -> np.ndarray:
    """Returns the underlying rotation matrix."""
    return self._C_ba

  def r_ba_ina(self) -> np.ndarray:
    """Returns the "forward" translation r_ba_ina = -C_ba.transpose()*r_ab_inb."""
    return -self._C_ba.swapaxes(-2, -1) @ self._r_ab_inb

  def v_ba_ina(self) -> np.ndarray:
    """Returns the "forward" velocity v_ba_ina = -C_ba.transpose()*v_ab_inb."""
    return -self._C_ba.swapaxes(-2, -1) @ self._v_ab_inb

  def r_ab_inb(self) -> np.ndarray:
    """Returns the underlying r_ab_inb vector."""
    return self._r_ab_inb

  def v_ab_inb(self) -> np.ndarray:
    """Returns the underlying v_ab_inb vector."""
    return self._v_ab_inb

  def vec(self) -> np.ndarray:
    """Returns the corresponding Lie algebra using the logarithmic map."""
    return se23op.tran2vec(self._C_ba, self._r_ab_inb, self._v_ab_inb)

  def inverse(self) -> Transformation:
    """Returns the inverse transformation."""
    temp = VTransformation(C_ba=self._C_ba.swapaxes(-2, -1), r_ba_in_a=self._r_ab_inb, v_ba_in_a=self._v_ab_inb)
    temp.reproject()
    return temp

  def adjoint(self) -> np.ndarray:
    """Returns the 9x9 adjoint transformation matrix."""
    return se23op.tranAd(self._C_ba, self._r_ab_inb, self._v_ab_inb)

  def reproject(self) -> None:
    """Reprojects the transformation matrix back onto SE(3)."""
    self._C_ba = so3op.vec2rot(so3op.rot2vec(self._C_ba))

  def __matmul__(self, other) -> VTransformation:
    return VTransformation(T_ba=self.matrix() @ other.matrix())

  def __mul__(self, other) -> VTransformation:
    return VTransformation(T_ba=self.matrix() @ other.matrix())

  def __str__(self):
    return str(self.matrix())
