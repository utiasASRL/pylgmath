from typing import Optional

import numpy as np
import numpy.linalg as npla

from ..so3 import operations as so3op


def hat(xi_or_rho: np.ndarray, aaxis: Optional[np.ndarray] = None) -> np.ndarray:
  """Builds the 4x4 "skew symmetric matrix".

  The hat (^) operator, builds the 4x4 skew symmetric matrix from the 3x1 axis angle vector and 3x1 translation vector.
    hat(rho, aaxis) = [aaxis^ rho] = [0.0  -a3   a2  rho1]
                      [  0^T    0]   [ a3  0.0  -a1  rho2]
                                     [-a2   a1  0.0  rho3]
                                     [0.0  0.0  0.0   0.0]
  See eq. 4 in Barfoot-TRO-2014 for more information.

  Args:
    xi_or_rho (np.ndarray): either 6x1 se3 algebra vector xi, or 3x1 translation vector rho when aaxis is None
    aaxis (np.ndarray|None): 3x1 axis angle vector
  Returns:
    np.ndarray: the 4x4 "skew symmetric matrix"
  """
  if aaxis is None:
    assert xi_or_rho.shape[-2:] == (6, 1)
    rho = xi_or_rho[..., :3, :]
    aaxis = xi_or_rho[..., 3:, :]
  else:
    assert xi_or_rho.shape[-2:] == (3, 1)
    assert aaxis.shape[-2:] == (3, 1)
    rho = xi_or_rho

  x_hat = np.zeros(rho.shape[:-2] + (4, 4))
  x_hat[..., :3, :3] = so3op.hat(aaxis)
  x_hat[..., :3, 3:4] = rho
  return x_hat


def curlyhat(xi_or_rho, aaxis=None) -> np.ndarray:
  """Builds the 6x6 "curly hat" matrix (related to the skew symmetric matrix).

  The curly hat operator builds the 6x6 skew symmetric matrix from the 3x1 axis angle vector and 3x1 translation vector.
    curlyhat(rho, aaxis) = [aaxis^   rho^]
                           [     0 aaxis^]
  See eq. 12 in Barfoot-TRO-2014 for more information.

  Args:
    xi_or_rho (np.ndarray): either 6x1 se3 algebra vector xi, or 3x1 translation vector rho when aaxis is None
    aaxis (np.ndarray|None): 3x1 axis angle vector
  Returns:
    np.ndarray: the 6x6 "curly hat" matrix
  """
  if aaxis is None:
    assert xi_or_rho.shape[-2:] == (6, 1)
    rho = xi_or_rho[..., :3, :]
    aaxis = xi_or_rho[..., 3:, :]
  else:
    assert xi_or_rho.shape[-2:] == (3, 1)
    assert aaxis.shape[-2:] == (3, 1)
    rho = xi_or_rho

  x_curlyhat = np.zeros(rho.shape[:-2] + (6, 6))
  x_curlyhat[..., :3, :3] = x_curlyhat[..., 3:, 3:] = so3op.hat(aaxis)
  x_curlyhat[..., :3, 3:] = so3op.hat(rho)

  return x_curlyhat


def Cr2T(C_ab, r_ba_ina):
  T_ab = np.tile(np.eye(4), (*C_ab.shape[:-2], 1, 1))
  T_ab[..., 0:3, 0:3] = C_ab
  T_ab[..., 0:3, 3:4] = r_ba_ina
  return T_ab


def T2Cr(T_ab):
  C_ab = T_ab[..., 0:3, 0:3]
  r_ba_ina = T_ab[..., 0:3, 3:4]
  return C_ab, r_ba_ina


def validate_tran(T):
  """causes an error if the transformation matrix is not in SE(3)"""
  so3op.validate_rot(T[..., 0:3, 0:3])
  err = np.max(np.abs(np.array([0., 0., 0., 1.]) - T[..., 3, :]))
  if err > 1e-10:
    raise RuntimeError('The bottom row of the transformation matrix should be [0,0,0,1]. Maximum error: {}'.format(err))


def _vec2tran_analytical(rho_ba, aaxis_ba):

  C_ab = so3op.vec2rot(aaxis_ba)
  J_ab = so3op.vec2jac(aaxis_ba)
  r_ba_ina = J_ab @ rho_ba

  T_ab = np.tile(np.eye(4), (*rho_ba.shape[:-2], 1, 1))
  T_ab[..., 0:3, 0:3] = C_ab
  T_ab[..., 0:3, 3:4] = r_ba_ina

  return T_ab


def _vec2tran_numerical(rho_ba, aaxis_ba, num_terms):

  T_ab = np.tile(np.eye(4), (*aaxis_ba.shape[:-2], 1, 1))

  xi_ba = np.concatenate((rho_ba, aaxis_ba), axis=-2)

  x_small = hat(xi_ba)
  x_small_n = np.tile(np.eye(4), (*aaxis_ba.shape[:-2], 1, 1))

  for n in range(1, num_terms + 1):
    x_small_n = x_small_n @ (x_small / n)
    T_ab = T_ab + x_small_n

  return T_ab


def _vec2tran_T(xi_ba, num_terms=0):
  if num_terms != 0:
    return _vec2tran_numerical(xi_ba[..., :3, :], xi_ba[..., 3:, :], num_terms)
  else:
    return _vec2tran_analytical(xi_ba[..., :3, :], xi_ba[..., 3:, :])


_vec2tran_T_vec = np.vectorize(_vec2tran_T, signature='(6,1),()->(4,4)')


def vec2tran(xi_ba: np.ndarray, num_terms: int = 0, return_T: bool = True) -> np.ndarray:
  """Builds a transformation matrix using the analytical exponential map or infinite series evaluation of the
  exponential map.

  This function builds a transformation matrix, T_ab, using the analytical exponential map, from the se3 algebra vector,
  xi_ba,
    T_ab = exp(xi_ba^) = [ C_ab r_ba_ina],   xi_ba = [  rho_ba]
                         [  0^T        1]            [aaxis_ba]
  where C_ab is a 3x3 rotation matrix from 'b' to 'a', r_ba_ina is the 3x1 translation vector from 'a' to 'b' expressed
  in frame 'a', aaxis_ba is a 3x1 axis-angle vector, the magnitude of the angle of rotation can be recovered by finding
  the norm of the vector, and the axis of rotation is the unit-length vector that arises from normalization.
  Note that the angle around the axis, aaxis_ba, is a right-hand-rule (counter-clockwise positive) angle from 'a' to
  'b'. The parameter, rho_ba, is a special translation-like parameter related to 'twist' theory. It is most inuitively
  described as being like a constant linear velocity (expressed in the smoothly-moving frame) for a fixed duration;
  for example, consider the curve of a car driving 'x' meters while turning at a rate of 'y' rad/s.
  For more information see Barfoot-TRO-2014 Appendix B1.
  Alternatively, we that note that
    T_ba = exp(-xi_ba^) = exp(xi_ab^).
  Both the analytical (num_terms = 0) or the numerical (num_terms > 0) may be evaluated.

  Args:
    xi_ba (np.ndarray): 6x1 se3 algebra vector xi
    num_terms (int): number of terms in the infinite series; use analytical solution of 0
    return_T (bool): returns T_ab if True; otherwise returns C_ab and r_ba_ina
  Returns:
    np.ndarray: T_ab
    or
    Tuple[np.ndarray, np.ndarray]: C_ab and r_ba_ina
  """
  assert xi_ba.shape[-2:] == (6, 1)
  T_ab = _vec2tran_T_vec(xi_ba, num_terms)
  return T_ab if return_T else T2Cr(T_ab)


def tran2vec(T_or_C_ab: np.ndarray, r_ba_ina: Optional[np.ndarray] = None) -> np.ndarray:
  """Compute the matrix log of a transformation matrix.

  Compute the inverse of the exponential map (the logarithmic map). This lets us go from a 4x4 transformation matrix
  back to a 6x1 se3 algebra vector (composed of a 3x1 axis-angle vector and 3x1 twist-translation vector). In some
  cases, when the rotation in the transformation matrix is 'numerically off', this involves some 'projection' back to
  SE(3).
    xi_ba = ln(T_ab)
  where xi_ba is the 6x1 se3 algebra vector. Alternatively, we that note that
    xi_ab = -xi_ba = ln(T_ba) = ln(T_ab^{-1})
  See Barfoot-TRO-2014 Appendix B2 for more information.

  Args:
    T_or_C_ab (np.ndarray): either 4x4 transformation matrix T_ab, or 3x3 rotation matrix C_ab if r_ba_ina is None
    r_ba_ina (np.ndarray|None): 3x1 translation vector
  Returns:
    np.ndarray: 6x1 se3 algebra vector xi_ba
  """
  if r_ba_ina is None:
    assert T_or_C_ab.shape[-2:] == (4, 4)
    C_ab, r_ba_ina = T2Cr(T_or_C_ab)
  else:
    assert T_or_C_ab.shape[-2:] == (3, 3)
    assert r_ba_ina.shape[-2:] == (3, 1)
    C_ab = T_or_C_ab

  aaxis_ba = so3op.rot2vec(C_ab)
  rho_ba = so3op.vec2jacinv(aaxis_ba) @ r_ba_ina

  xi_ba = np.concatenate((rho_ba, aaxis_ba), axis=-2)

  return xi_ba


def tranAd(T_or_C_ab, r_ba_ina=None) -> np.ndarray:
  """Builds the 6x6 adjoint transformation matrix from either the 3x3 rotation matrix and 3x1 translation vector, or the
  4x4 transformation matrix.
  Adjoint(T_ab) = Adjoint([C_ab r_ba_ina]) = [C_ab r_ba_ina^*C_ab] = exp(curlyhat(xi_ba))
                          ([ 0^T        1])   [   0           C_ab]
  See eq. 101 in Barfoot-TRO-2014 for more information.

  Args:
    T_or_C_ab (np.ndarray): either 4x4 transformation matrix T_ab, or 3x3 rotation matrix C_ab if r_ba_ina is None
    r_ba_ina (np.ndarray|None): 3x1 translation vector
  Returns:
    np.ndarray: the 6x6 adjoint transformation matrix
  """
  if r_ba_ina is None:
    C_ab, r_ba_ina = T2Cr(T_or_C_ab)
  else:
    C_ab = T_or_C_ab

  adjoint_T_ab = np.zeros(T_or_C_ab.shape[:-2] + (6, 6))
  adjoint_T_ab[..., :3, :3] = C_ab
  adjoint_T_ab[..., 3:, 3:] = C_ab
  adjoint_T_ab[..., :3, 3:] = so3op.hat(r_ba_ina) @ C_ab

  return adjoint_T_ab


def _vec2Q(rho_ba, aaxis_ba):

  ang = npla.norm(aaxis_ba, axis=-2, keepdims=True)
  ang2 = ang * ang
  ang3 = ang2 * ang
  ang4 = ang3 * ang
  ang5 = ang4 * ang
  cang = np.cos(ang)
  sang = np.sin(ang)
  m2 = (ang - sang) / ang3
  m3 = (1.0 - 0.5 * ang2 - cang) / ang4
  m4 = 0.5 * (m3 - 3 * (ang - sang - ang3 / 6) / ang5)

  # construct matrix terms
  rx = so3op.hat(rho_ba)
  px = so3op.hat(aaxis_ba)
  pxrx = px @ rx
  rxpx = rx @ px
  pxrxpx = pxrx @ px

  # construct Q matrix
  return (0.5 * rx + m2 * (pxrx + rxpx + pxrxpx) - m3 * (px @ pxrx + rxpx @ px - 3 * pxrxpx) - m4 *
          (pxrxpx @ px + px @ pxrxpx))


def vec2Q(xi_or_rho_ba: np.ndarray, aaxis_ba: Optional[np.ndarray] = None) -> np.ndarray:
  """Constructs the 3x3 "Q" matrix, used in the 6x6 Jacobian of SE(3).

  See eqn 7.86a in Barfoot's book

  Args:
    xi_or_rho_ba (np.ndarray): either 6x1 se3 algebra vector xi, or 3x1 translation vector rho when aaxis is None
    aaxis_ba (np.ndarray|None): 3x1 axis angle vector
  Returns:
    np.ndarray: the 3x3 "Q" matrix
  """
  if aaxis_ba is None:
    assert xi_or_rho_ba.shape[-2:] == (6, 1)
    rho_ba = xi_or_rho_ba[..., :3, :]
    aaxis_ba = xi_or_rho_ba[..., 3:, :]
  else:
    assert xi_or_rho_ba.shape[-2:] == (3, 1)
    assert aaxis_ba.shape[-2:] == (3, 1)
    rho_ba = xi_or_rho_ba

  return _vec2Q(rho_ba, aaxis_ba)


def _vec2jac_analytical(rho_ba, aaxis_ba):

  J_ab = np.zeros(rho_ba.shape[:-2] + (6, 6))
  J_ab[..., :3, :3] = so3op.vec2jac(aaxis_ba)
  J_ab[..., 3:, 3:] = so3op.vec2jac(aaxis_ba)
  J_ab[..., :3, 3:] = _vec2Q(rho_ba, aaxis_ba)

  return J_ab


def _vec2jac_numerical(rho_ba, aaxis_ba, num_terms):
  J_ab = np.tile(np.eye(6), (*rho_ba.shape[:-2], 1, 1))

  xi_ba = np.concatenate((rho_ba, aaxis_ba), axis=-2)
  x_small = curlyhat(xi_ba)
  x_small_n = np.tile(np.eye(6), (*rho_ba.shape[:-2], 1, 1))

  for n in range(1, num_terms + 1):
    x_small_n = x_small_n @ (x_small / (n + 1))
    J_ab = J_ab + x_small_n

  return J_ab


def _vec2jac_rho_aaxis(rho_ba, aaxis_ba, num_terms=0):
  tolerance = 1e-12
  if (npla.norm(aaxis_ba, axis=-2)[0] < tolerance) or (num_terms != 0):
    return _vec2jac_numerical(rho_ba, aaxis_ba, num_terms)
  else:
    return _vec2jac_analytical(rho_ba, aaxis_ba)


_vec2jac_rho_aaxis_vec = np.vectorize(_vec2jac_rho_aaxis, signature='(3,1),(3,1),()->(6,6)')


def vec2jac(xi_or_rho_ba: np.ndarray, aaxis_ba: Optional[np.ndarray] = None, num_terms: int = 0) -> np.ndarray:
  """Build the 6x6 left Jacobian of SE(3).

  For the sake of a notation, we assign subscripts consistence with the transformation,
    J_ab = J(xi_ba)
  Where applicable, we also note that
    J(xi_ba) = Adjoint(exp(xi_ba^)) * J(-xi_ba),
  and
    Adjoint(exp(xi_ba^)) = identity + curlyhat(xi_ba) * J(xi_ba).
  For more information see eq. 100 in Barfoot-TRO-2014.

  Args:
    xi_or_rho_ba (np.ndarray): either 6x1 se3 algebra vector xi, or 3x1 translation vector rho when aaxis is None
    aaxis_ba (np.ndarray|None): 3x1 axis angle vector
    num_terms (int): number of terms used in the infinite series; use analytical solution if 0
  Returns:
    np.ndarray: the 6x6 left Jacobian of SE(3)
  """
  if aaxis_ba is None:
    assert xi_or_rho_ba.shape[-2:] == (6, 1)
    rho_ba = xi_or_rho_ba[..., :3, :]
    aaxis_ba = xi_or_rho_ba[..., 3:, :]
  else:
    assert xi_or_rho_ba.shape[-2:] == (3, 1)
    assert aaxis_ba.shape[-2:] == (3, 1)
    rho_ba = xi_or_rho_ba

  return _vec2jac_rho_aaxis_vec(rho_ba, aaxis_ba, num_terms)


def _vec2jacinv_analytical(rho_ba, aaxis_ba):

  J_ab_inv = np.zeros(rho_ba.shape[:-2] + (6, 6))
  J33_ab_inv = so3op.vec2jacinv(aaxis_ba)
  J_ab_inv[..., :3, :3] = J33_ab_inv
  J_ab_inv[..., 3:, 3:] = J33_ab_inv
  J_ab_inv[..., :3, 3:] = -J33_ab_inv @ _vec2Q(rho_ba, aaxis_ba) @ J33_ab_inv

  return J_ab_inv


def _vec2jacinv_numerical(rho_ba, aaxis_ba, num_terms):
  J_ab_inv = np.tile(np.eye(6), (*rho_ba.shape[:-2], 1, 1))

  xi_ba = np.concatenate((rho_ba, aaxis_ba), axis=-2)
  x_small = curlyhat(xi_ba)
  x_small_n = np.tile(np.eye(6), (*rho_ba.shape[:-2], 1, 1))

  bernoulli = np.array([
      1.0, -0.5, 1.0 / 6.0, 0.0, -1.0 / 30.0, 0.0, 1.0 / 42.0, 0.0, -1.0 / 30.0, 0.0, 5.0 / 66.0, 0.0, -691.0 / 2730.0,
      0.0, 7.0 / 6.0, 0.0, -3617.0 / 510.0, 0.0, 43867.0 / 798.0, 0.0, -174611.0 / 330.0
  ])

  for n in range(1, num_terms + 1):
    x_small_n = x_small_n @ (x_small / n)
    J_ab_inv = J_ab_inv + bernoulli[n] * x_small_n

  return J_ab_inv


def _vec2jacinv_rho_aaxis(rho_ba, aaxis_ba, num_terms=0):
  tolerance = 1e-12
  if (npla.norm(aaxis_ba, axis=-2)[0] < tolerance) or (num_terms != 0):
    return _vec2jacinv_numerical(rho_ba, aaxis_ba, num_terms)
  else:
    return _vec2jacinv_analytical(rho_ba, aaxis_ba)


_vec2jacinv_rho_aaxis_vec = np.vectorize(_vec2jacinv_rho_aaxis, signature='(3,1),(3,1),()->(6,6)')


def vec2jacinv(xi_or_rho_ba: np.ndarray, aaxis_ba: Optional[np.ndarray] = None, num_terms: int = 0) -> np.ndarray:
  """Build the 6x6 inverse Jacobian matrix of SE(3).

  For the sake of a notation, we assign subscripts consistence with the transformation,
    J_ab_inverse = J(xi_ba)^{-1},
  Please note that J_ab_inverse is not equivalent to J_ba:
    J(xi_ba)^{-1} != J(-xi_ba)
  For more information see eq. 103 in Barfoot-TRO-2014.

  Args:
    xi_or_rho_ba (np.ndarray): either 6x1 se3 algebra vector xi, or 3x1 translation vector rho when aaxis is None
    aaxis_ba (np.ndarray|None): 3x1 axis angle vector
    num_terms (int): number of terms used in the infinite series; use analytical solution if 0
  Returns:
    np.ndarray: the 6x6 inverse Jacobian matrix of SE(3)
  """
  if aaxis_ba is None:
    assert xi_or_rho_ba.shape[-2:] == (6, 1)
    rho_ba = xi_or_rho_ba[..., :3, :]
    aaxis_ba = xi_or_rho_ba[..., 3:, :]
  else:
    assert xi_or_rho_ba.shape[-2:] == (3, 1)
    assert aaxis_ba.shape[-2:] == (3, 1)
    rho_ba = xi_or_rho_ba

  return _vec2jacinv_rho_aaxis_vec(rho_ba, aaxis_ba, num_terms)


def point2fs(p, scale=None):
  """Turns a homogeneous point into a special 4x6 matrix

  See eq. 72 in Barfoot-TRO-2014 for more information.

  Args:
    p (np.ndarray): a point in homogeneous coordinate if scale is None; otherwise in cartesian coordinate
    scale (float): the scale
  Returns:
    np.ndarray: the special 4x6 matrix
  """
  if scale is None:
    assert p.shape[-2:] == (4, 1)
    scale = p[..., 3, 0]
    p = p[..., :-1, :]
  assert p.shape[-2:] == (3, 1)
  mat = np.zeros(p.shape[:-2] + (4, 6))
  mat[..., :3, :3] = np.array(scale)[..., None, None] * np.tile(np.eye(3), (*p.shape[:-2], 1, 1))
  mat[..., :3, 3:] = -so3op.hat(p)
  return mat


def point2sf(p, scale=None):
  """Turns a homogeneous point into a special 6x4 matrix

  See eq. 72 in Barfoot-TRO-2014 for more information.

  Args:
    p (np.ndarray): a point in homogeneous coordinate if scale is None; otherwise in cartesian coordinate
    scale (float): the scale
  Returns:
    np.ndarray: the special 4x6 matrix
  """
  if scale is None:
    assert p.shape[-2:] == (4, 1)
    scale = p[..., 3, 0]
    p = p[..., :-1, :]
  assert p.shape[-2:] == (3, 1)
  mat = np.zeros(p.shape[:-2] + (6, 4))
  mat[..., 3:, :3] = -so3op.hat(p)
  mat[..., :3, 5:] = p
  return mat
