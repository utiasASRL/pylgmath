import numpy as np
import numpy.linalg as npla


def pol2cart(pts: np.ndarray) -> np.ndarray:
  """Converts points from polar to cartesian.
  Args:
    pts (np.ndarray): points in polar coordinate [rho, theta, phi]
  Returns:
    np.ndarray: points in cartesian coordinate [x, y, z]
  """
  assert pts.shape[-2:] == (3, 1)
  x = np.cos(pts[..., 2:3, :]) * np.sin(pts[..., 1:2, :]) * pts[..., 0:1, :]
  y = np.sin(pts[..., 2:3, :]) * np.sin(pts[..., 1:2, :]) * pts[..., 0:1, :]
  z = np.cos(pts[..., 1:2, :]) * pts[..., 0:1, :]
  return np.concatenate((x, y, z), axis=-2)


def cart2pol(pts: np.ndarray) -> np.ndarray:
  """Converts points from cartesian to polar.
  Args:
    pts (np.ndarray): points in cartesian coordinate [x, y, z]
  Returns:
    np.ndarray: points in polar coordinate [rho, theta, phi]
  """
  assert pts.shape[-2:] == (3, 1)
  rho = npla.norm(pts, axis=-2, keepdims=True)
  phi = np.arctan2(pts[..., 1:2, :], pts[..., 0:1, :])
  theta = np.arctan2(np.sqrt(pts[..., 0:1, :]**2 + pts[..., 1:2, :]**2), pts[..., 2:3, :])
  return np.concatenate((rho, theta, phi), axis=-2)