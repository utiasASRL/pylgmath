import numpy as np
import numpy.linalg as npla

from pylgmath import so3op

TEST_SIZE = 10000


def test_hat_hatinv():

  test_vecs = np.random.uniform(-np.pi / 2, np.pi / 2, size=(TEST_SIZE, 3, 1))

  mats = so3op.hat(test_vecs)
  vecs = so3op.hatinv(mats)

  # going back and forth
  assert np.allclose(test_vecs, vecs)
  # identity, rot^T = rot^-1
  assert np.allclose(np.swapaxes(mats, -2, -1), -mats)


def test_vec2rot_rot2vec():

  test_vecs = np.random.uniform(-np.pi / 2, np.pi / 2, size=(TEST_SIZE, 3, 1))

  mats = so3op.vec2rot(test_vecs)
  mats_numerical = so3op.vec2rot(test_vecs, 20)
  vecs = so3op.rot2vec(mats)

  # going back and forth
  assert np.allclose(test_vecs, vecs)
  # identity, rot^T = rot^-1
  assert np.allclose(np.swapaxes(mats, -2, -1), npla.inv(mats))
  # numerical vs analytical
  assert np.allclose(mats, mats_numerical, atol=1e-6)


def test_vec2rot_rot2vec_special_cases():

  test_vecs = np.expand_dims(np.array([
      [np.pi, 0, 0],
      [0, np.pi, 0],
      [0, 0, np.pi],
      [np.pi / 2, 0, 0],
      [0, np.pi / 2, 0],
      [0, 0, np.pi / 2],
  ]),
                             axis=-1)

  mats = so3op.vec2rot(test_vecs)
  mats_numerical = so3op.vec2rot(test_vecs, 20)
  vecs = so3op.rot2vec(mats)

  # going back and forth
  assert np.allclose(test_vecs, vecs)
  # identity, rot^T = rot^-1
  assert np.allclose(np.swapaxes(mats, -2, -1), npla.inv(mats))
  # numerical vs analytical
  assert np.allclose(mats, mats_numerical, atol=1e-6)


def test_vec2jac_vec2jacinv():

  test_vecs = np.random.uniform(-np.pi / 2, np.pi / 2, size=(TEST_SIZE, 3, 1))

  jac = so3op.vec2jac(test_vecs)
  jac_numerical = so3op.vec2jac(test_vecs, 20)
  jacinv = so3op.vec2jacinv(test_vecs)
  jacinv_numerical = so3op.vec2jacinv(test_vecs, 20)

  # jacinv vs inverse of jac
  assert np.allclose(jac, npla.inv(jacinv))
  # numerical vs analytical
  assert np.allclose(jac, jac_numerical, atol=1e-6)
  assert np.allclose(jacinv, jacinv_numerical, atol=1e-6)


def test_vec2jac_vec2jacinv_special_cases():

  test_vecs = np.expand_dims(np.array([
      [np.pi, 0, 0],
      [0, np.pi, 0],
      [0, 0, np.pi],
      [np.pi / 2, 0, 0],
      [0, np.pi / 2, 0],
      [0, 0, np.pi / 2],
  ]),
                             axis=-1)

  jac = so3op.vec2jac(test_vecs)
  jac_numerical = so3op.vec2jac(test_vecs, 20)
  jacinv = so3op.vec2jacinv(test_vecs)
  jacinv_numerical = so3op.vec2jacinv(test_vecs, 20)

  # jacinv vs inverse of jac
  assert np.allclose(jac, npla.inv(jacinv))
  # numerical vs analytical
  assert np.allclose(jac, jac_numerical, atol=1e-6)
  assert np.allclose(jacinv, jacinv_numerical, atol=1e-6)

def test_vec2N():

  test_vecs = np.random.uniform(-np.pi / 2, np.pi / 2, size=(TEST_SIZE, 3, 1))

  N = so3op.vec2N(test_vecs)
  N_numerical = so3op.vec2N(test_vecs, 20)

  assert np.allclose(N, N_numerical, atol=1e-6)

def test_vec2N_special_cases():

  test_vecs = np.expand_dims(np.array([
      [0, 0, 0],
      [np.pi, 0, 0],
      [0, np.pi, 0],
      [0, 0, np.pi],
      [np.pi / 2, 0, 0],
      [0, np.pi / 2, 0],
      [0, 0, np.pi / 2],
  ]),
                             axis=-1)

  N = so3op.vec2jac(test_vecs)
  N_numerical = so3op.vec2jac(test_vecs, 20)

  assert np.allclose(N, N_numerical, atol=1e-6)
