from typing import Optional

import numpy as np
import numpy.linalg as npla

def hat(tau) -> np.ndarray:
  """Builds the 5x5 lie algebra matrix.

  The hat (^) operator, builds the 5x5 matrix from the time increment tau.
    hat(tau) = [1   0    ] 
               [0^T 1   0]
               [0^T tau 1]
  See Barfoot's book for more information.

  Args:
    tau: time increment 
  Returns:
    np.ndarray: the 5x5 Lie algebra matrix 
  """

  assert tau is not None

  d_hat = np.zeros(tau.shape + (5, 5))
  d_hat[..., 4, 3] = tau
  return d_hat


def _vec2tran_analytical(tau):

  D = np.tile(np.eye(5), (*tau.shape, 1, 1))
  D[..., 4, 3] = tau

  return D


def _vec2tran_numerical(tau, num_terms):

  D = np.tile(np.eye(4), (*tau.shape, 1, 1))

  x_small = hat(tau)
  x_small_n = np.tile(np.eye(4), (*tau.shape, 1, 1))

  for n in range(1, num_terms + 1):
    x_small_n = x_small_n @ (x_small / n)
    D = D + x_small_n

  return D


def _vec2tran_T(tau, num_terms=0):
  if num_terms != 0:
    return _vec2tran_numerical(tau, num_terms)
  else:
    return _vec2tran_analytical(tau)


_vec2tran_T_vec = np.vectorize(_vec2tran_T, signature='(),()->(5,5)')


def vec2tran(tau, num_terms: int = 0) -> np.ndarray:
  """Builds a transformation matrix using the analytical exponential map or infinite series evaluation of the
  exponential map.

  This function builds a Time Machine matrix, D, using the analytical exponential map, from the algebra vector,
  Both the analytical (num_terms = 0) or the numerical (num_terms > 0) may be evaluated.

  Args:
    tau: time increment
    num_terms (int): number of terms in the infinite series; use analytical solution of 0
  Returns:
    np.ndarray: D
  """
  D = _vec2tran_T_vec(tau, num_terms)
  return D


def tran2vec(D: np.ndarray):
  """Compute the matrix log of a transformation matrix.

  Compute the inverse of the exponential map (the logarithmic map). This lets us go from a 5x5 Time Machine matrix
  back to a 1x1 time increment algebra vector,
    tau = ln(D)

  Args:
    D (np.ndarray): 5x5 Time Machine matrix 
  Returns:
    np.ndarray: Time increment tau
  """
  assert D.shape[-2:] == (5, 5)

  return D[..., 4, 3]

def vec2se23conjugation(tau) -> np.ndarray:
  """Compute the (right) conjugation group action of the time machine on se23
    linear operator X such that X * xi = D * xi * D^-1
    D is a time machine, xi is in the lie algebra se_2(3)

  Args:
    tau: time increment of the time machine 
  Returns:
    np.ndarray: X, the linear operator that return the conjugation of an element of se_2(3) by the given time machine
  """

  X = np.tile(np.eye(9), (*tau.shape, 1, 1))
  X[...,0,3] = X[...,1,4] = X[...,2,5] = tau

  return X



