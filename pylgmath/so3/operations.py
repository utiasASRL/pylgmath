import numpy as np
import numpy.linalg as npla


def hat(x: np.ndarray) -> np.ndarray:
  """Builds the 3x3 skew symmetric matrix

  The hat (^) operator, builds the 3x3 skew symmetric matrix from the 3x1 vector:
    v^ = [0.0  -v3   v2]
         [ v3  0.0  -v1]
         [-v2   v1  0.0]

  See eq. 5 in Barfoot-TRO-2014 for more information.

  Args:
    x (np.ndarray): a 3x1 vector
  Returns:
    np.ndarray: the 3x3 skew symmetric matrix of x
  """
  assert x.shape[-2:] == (3, 1)
  x_hat = np.zeros(x.shape[:-2] + (3, 3))
  x_hat[..., 0, 1] = -x[..., 2, 0]
  x_hat[..., 0, 2] = x[..., 1, 0]
  x_hat[..., 1, 0] = x[..., 2, 0]
  x_hat[..., 1, 2] = -x[..., 0, 0]
  x_hat[..., 2, 0] = -x[..., 1, 0]
  x_hat[..., 2, 1] = x[..., 0, 0]
  return x_hat


def hatinv(x):
  """Gets the 3x1 vector from a 3x3 skew symmetric matrix. Inverse of the hat operator.
  Args:
    x (np.ndarray): a 3x3 skew symmetric matrix
  Returns:
    np.ndarray: the 3x1 vector of x
  """
  assert x.shape[-2:] == (3, 3)
  x_hatinv = np.zeros(x.shape[:-2] + (3, 1))
  x_hatinv[..., 0, 0] = x[..., 2, 1]
  x_hatinv[..., 1, 0] = x[..., 0, 2]
  x_hatinv[..., 2, 0] = x[..., 1, 0]
  return x_hatinv


def validate_rot(C):
  E = np.swapaxes(C, -2, -1) @ C - np.eye(3)
  err = np.max(np.abs(E))
  if err > 1e-7:
    raise RuntimeError('The rotation matrix is not valid. Maximum error: {}'.format(err))


def _vec2rot_analytical(aaxis_ba):

  phi_ba = npla.norm(aaxis_ba, axis=-2, keepdims=True)
  axis = aaxis_ba / phi_ba

  sp = np.sin(phi_ba)
  cp = np.cos(phi_ba)

  return (cp * np.eye(3) + (1 - cp) * (axis @ axis.swapaxes(-1, -2)) + sp * hat(axis))


def _vec2rot_numerical(aaxis_ba, num_terms=10):

  C_ab = np.tile(np.eye(3), (*aaxis_ba.shape[:-2], 1, 1))

  x_small = hat(aaxis_ba)
  x_small_n = np.tile(np.eye(3), (*aaxis_ba.shape[:-2], 1, 1))

  for n in range(1, num_terms + 1):
    x_small_n = x_small_n @ (x_small / n)
    C_ab = C_ab + x_small_n

  return C_ab


def _vec2rot(aaxis_ba, num_terms=0):
  tolerance = 1e-12
  phi_ba = npla.norm(aaxis_ba, axis=-2)[0]
  if (phi_ba < tolerance) or (num_terms != 0):
    return _vec2rot_numerical(aaxis_ba, num_terms)
  else:
    return _vec2rot_analytical(aaxis_ba)


_vec2rot_vec = np.vectorize(_vec2rot, signature='(3,1),()->(3,3)')


def vec2rot(aaxis_ba: np.ndarray, num_terms: int = 0) -> np.ndarray:
  """Builds a rotation matrix using the exponential map.

  This function builds a rotation matrix, C_ab, using the exponential map (from an axis-angle parameterization).
    C_ab = exp(aaxis_ba^),
  where aaxis_ba is a 3x1 axis-angle vector, the magnitude of the angle of rotation can be recovered by finding the norm
  of the vector, and the axis of rotation is the unit-length vector that arises from normalization.
  Note that the angle around the axis, aaxis_ba, is a right-hand-rule (counter-clockwise positive) angle from 'a' to
  'b'.
  Alternatively, we that note that
    C_ba = exp(-aaxis_ba^) = exp(aaxis_ab^).
  Typical robotics convention has some oddity when it comes using this exponential map in practice. For example, if we
  wish to integrate the kinematics:
    d/dt C = omega^ * C,
  where omega is the 3x1 angular velocity, we employ the convention:
    C_20 = exp(deltaTime*-omega^) * C_10,
  Noting that omega is negative (left-hand-rule).
  For more information see eq. 97 in Barfoot-TRO-2014.

  Args:
    aaxis_ba (np.ndarray): the axis-angle vector of the rotation
    num_terms (int): number of terms used in the infinite series approximation of the exponential map
  Returns:
    np.ndarray: the 3x3 rotation matrix of aaxis_ba
  """
  assert aaxis_ba.shape[-2:] == (3, 1)
  return _vec2rot_vec(aaxis_ba, num_terms)


def _rot2vec(C_ab):
  phi_ba = np.arccos(np.clip(0.5 * (np.trace(C_ab) - 1), -1, 1))  # clip to avoid numerical issues
  sinphi_ba = np.sin(phi_ba)

  if np.abs(sinphi_ba) > 1e-9:  # General case: phi_ba is NOT near [0, pi, 2*pi]
    axis = (0.5 / sinphi_ba) * (np.array([C_ab[2, 1] - C_ab[1, 2], C_ab[0, 2] - C_ab[2, 0], C_ab[1, 0] - C_ab[0, 1]
                                         ]).reshape(3, 1))
    return phi_ba * axis

  elif np.abs(phi_ba) > 1e-9:  # phi_ba is near [pi, 2*pi]
    eigval, eigvec = npla.eig(C_ab)
    valid_eigval = np.abs(np.real(eigval) - 1) < 1e-10
    valid_axis = np.real(eigvec[:, valid_eigval])
    axis = valid_axis[:, valid_axis.shape[1] - 1].reshape(3, 1)
    aaxis_ba = phi_ba * axis
    if np.abs(np.trace(_vec2rot(aaxis_ba).T @ C_ab) - 3) > 1e-14:
      aaxis_ba = -aaxis_ba
    return aaxis_ba

  else:
    return np.array([[0., 0., 0.]]).T


_rot2vec_vec = np.vectorize(_rot2vec, signature='(3,3)->(3,1)')


def rot2vec(C_ab: np.ndarray) -> np.ndarray:
  """Computes the matrix log of a rotation matrix

  Compute the inverse of the exponential map (the logarithmic map). This lets us go from a 3x3 rotation matrix back to a
  3x1 axis angle parameterization. In some cases, when the rotation matrix is 'numerically off', this involves some
  'projection' back to SO(3).
    aaxis_ba = ln(C_ab)
  where aaxis_ba is a 3x1 axis angle, where the axis is normalized and the magnitude of the rotation can be recovered by
  finding the norm of the axis angle. Note that the angle around the axis, aaxis_ba, is a right-hand-rule
  (counter-clockwise positive) angle from 'a' to 'b'.
  Alternatively, we that note that
    aaxis_ab = -aaxis_ba = ln(C_ba) = ln(C_ab^T)
  See Barfoot-TRO-2014 Appendix B2 for more information.

  Args:
    C_ab (np.ndarray): a 3x3 rotation matrix
  Returns:
    np.ndarray: the 3x1 axis-angle vector of C_ab
  """
  assert C_ab.shape[-2:] == (3, 3)
  return _rot2vec_vec(C_ab)


def _vec2jac_analytical(aaxis_ba):
  phi_ba = npla.norm(aaxis_ba, axis=-2, keepdims=True)
  axis = aaxis_ba / phi_ba

  sph = np.sin(phi_ba) / phi_ba
  cph = (1 - np.cos(phi_ba)) / phi_ba

  return (sph * np.eye(3) + (1 - sph) * (axis @ np.swapaxes(axis, -1, -2)) + cph * hat(axis))


def _vec2jac_numerical(aaxis_ba, num_terms=10):
  J_ab = np.tile(np.eye(3), (*aaxis_ba.shape[:-2], 1, 1))

  x_small = hat(aaxis_ba)
  x_small_n = np.tile(np.eye(3), (*aaxis_ba.shape[:-2], 1, 1))

  for n in range(1, num_terms + 1):
    x_small_n = x_small_n @ (x_small / (n + 1))
    J_ab = J_ab + x_small_n

  return J_ab


def _vec2jac(aaxis_ba, num_terms=0):
  tolerance = 1e-12
  phi_ba = npla.norm(aaxis_ba, axis=-2)[0]
  if (phi_ba < tolerance) or (num_terms != 0):
    return _vec2jac_numerical(aaxis_ba, num_terms)
  else:
    return _vec2jac_analytical(aaxis_ba)


_vec2jac_vec = np.vectorize(_vec2jac, signature='(3,1),()->(3,3)')


def vec2jac(aaxis_ba: np.ndarray, num_terms: int = 0) -> np.ndarray:
  """Builds the 3x3 Jacobian matrix of SO(3)

  For the sake of a notation, we assign subscripts consistence with the rotation,
    J_ab = J(aaxis_ba),
  although we note to the SO(3) novice that this Jacobian is not a rotation matrix, and should be used with care.
  For more information see eq. 98 in Barfoot-TRO-2014.

  Args:
    aaxis_ba (np.ndarray): a 3x1 axis-angle vector
    num_terms (int): number of terms used in the infinite series approximation of the exponential map
  Returns:
    np.ndarray: the 3x3 Jacobian matrix of SO(3)
  """
  assert aaxis_ba.shape[-2:] == (3, 1)
  return _vec2jac_vec(aaxis_ba, num_terms)


def _vec2jacinv_analytical(aaxis_ba):
  phi_ba = npla.norm(aaxis_ba)
  axis = aaxis_ba / phi_ba
  halfphi = 0.5 * phi_ba

  return (halfphi / np.tan(halfphi) * np.eye(3) + (1 - halfphi / np.tan(halfphi)) * (axis * axis.T) -
          halfphi * hat(axis))


def _vec2jacinv_numerical(aaxis_ba, num_terms=10):
  J_ab_inverse = np.tile(np.eye(3), (*aaxis_ba.shape[:-2], 1, 1))

  x_small = hat(aaxis_ba)
  x_small_n = np.tile(np.eye(3), (*aaxis_ba.shape[:-2], 1, 1))

  bernoulli = np.array([
      1.0, -0.5, 1.0 / 6.0, 0.0, -1.0 / 30.0, 0.0, 1.0 / 42.0, 0.0, -1.0 / 30.0, 0.0, 5.0 / 66.0, 0.0, -691.0 / 2730.0,
      0.0, 7.0 / 6.0, 0.0, -3617.0 / 510.0, 0.0, 43867.0 / 798.0, 0.0, -174611.0 / 330.0
  ])

  for n in range(1, num_terms + 1):
    x_small_n = x_small_n @ (x_small / n)
    J_ab_inverse = J_ab_inverse + bernoulli[n] * x_small_n

  return J_ab_inverse


def _vec2jacinv(aaxis_ba, num_terms=0):
  tolerance = 1e-12
  phi_ba = npla.norm(aaxis_ba, axis=-2)[0]
  if (phi_ba < tolerance) or (num_terms != 0):
    return _vec2jacinv_numerical(aaxis_ba, num_terms)
  else:
    return _vec2jacinv_analytical(aaxis_ba)


_vec2jacinv_vec = np.vectorize(_vec2jacinv, signature='(3,1),()->(3,3)')


def vec2jacinv(aaxis_ba: np.ndarray, num_terms: int = 0) -> np.ndarray:
  """Builds the 3x3 inverse Jacobian matrix of SO(3)

  For the sake of a notation, we assign subscripts consistence with the rotation,
    J_ab_inverse = J(aaxis_ba)^{-1},
  although we note to the SO(3) novice that this Jacobian is not a rotation matrix, and should be used with care. Also,
  please note that J_ab_inverse is not equivalent to J_ba:
    J(aaxis_ba)^{-1} != J(-aaxis_ba)
  For more information see eq. 99 in Barfoot-TRO-2014.

  Args:
    aaxis_ba (np.ndarray): a 3x1 axis-angle vector
    num_terms (int): number of terms used in the infinite series approximation of the exponential map
  Returns:
    np.ndarray: the 3x3 inverse Jacobian matrix of SO(3)
  """
  assert aaxis_ba.shape[-2:] == (3, 1)
  return _vec2jacinv_vec(aaxis_ba, num_terms)