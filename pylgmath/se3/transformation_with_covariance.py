from __future__ import annotations

from typing import Optional
import numpy as np

from .transformation import Transformation


class TransformationWithCovariance(Transformation):
  """Light weight transformation class with added covariance propagation."""

  def __init__(self,
               *,
               tran_w_cov: Optional[TransformationWithCovariance] = None,
               covariance: Optional[np.ndarray] = None,
               init_cov_to_zero: bool = False,
               **kwargs) -> None:
    """Constructor with several options
    Args:
      tran_w_cov (Optional[TransformationWithCovariance]): an existing instance of this class. (copy construction)
      covariance (np.ndarray): the covariance matrix
      init_cov_to_zero (bool): whether or not to initialize covariance to zero; defaults to False.
    """
    # copy construction
    if tran_w_cov is not None:
      self._covariance = np.copy(tran_w_cov.cov())
      self._covariance_set = tran_w_cov.covariance_set()
      super().__init__(transformation=tran_w_cov)
      return

    super().__init__(**kwargs)

    # construction from a given covariance matrix
    if covariance is not None:
      # TODO assert that the covariance shape is consistent with the underlying transformation
      self._covariance = covariance
      self._covariance_set = True
      return

    # default construction
    self._covariance = np.zeros((6, 6))  # TODO make this the same shape as the underlying transformation
    self._covariance_set = init_cov_to_zero

  def cov(self) -> np.ndarray:
    """Returns a reference to the covariance matrix."""
    if not self._covariance_set:
      raise RuntimeError("Covariance accessed before being set.")
    return self._covariance

  def covariance_set(self) -> bool:
    """Returns whether the covariance has been set."""
    return self._covariance_set

  def set_covariance(self, cov: np.ndarray) -> None:
    """Sets the 6x6 covariance matrix.
    Args:
      cov: the 6x6 covariance matrix
    """
    assert cov.shape[:-2] == self._C_ba.shape[:-2] and cov.shape[-2:] == (6, 6)
    self._covariance = cov
    self._covariance_set = True

  def set_zero_covariance(self) -> None:
    """Sets the covariance to zero."""
    self._covariance = np.zeros_like(*self._C_ba.shape[:-2], 6, 6)
    self._covariance_set = True

  def inverse(self) -> TransformationWithCovariance:
    """Returns the inverse of this transformation including the covariance."""
    temp = TransformationWithCovariance(transformation=super().inverse())
    adjoint_of_inverse = temp.adjoint()
    temp.set_covariance(adjoint_of_inverse @ self._covariance @ adjoint_of_inverse.swapaxes(-2, -1))
    temp.covariance_set = self._covariance_set
    return temp
