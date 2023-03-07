import numpy as np
import numpy.linalg as npla

from pylgmath import se23op

TEST_SIZE = 10000

def test_hat():

  test_vecs = np.random.uniform(-np.pi, np.pi, size=(TEST_SIZE, 9, 1))

  mats = se23op.hat(test_vecs)

  assert np.allclose(np.swapaxes(mats, -2, -1)[...,:3,:3], -mats[...,:3,:3])
  assert np.allclose(mats[...,3:,:], np.zeros((*mats.shape[:-2], 2, 5)))


def test_vec2tran_tran2vec():

  test_vecs = np.random.uniform(-np.pi / 2, np.pi / 2, size=(TEST_SIZE, 9, 1))

  mats = se23op.vec2tran(test_vecs)
  mats_numerical = se23op.vec2tran(test_vecs, 20)
  vecs = se23op.tran2vec(mats)

  # going back and forth
  assert np.allclose(test_vecs, vecs)
  # numerical vs analytical
  assert np.allclose(mats, mats_numerical, atol=1e-4)


def test_vec2tran_tran2vec_special_cases():

  test_vecs = np.expand_dims(np.array([
      [0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, np.pi, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, np.pi, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, np.pi, 0, 0, 0],
      [0, 0, 0, np.pi / 2, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, np.pi / 2, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, np.pi / 2, 0, 0, 0],
  ]),
                             axis=-1)

  mats = se23op.vec2tran(test_vecs)
  mats_numerical = se23op.vec2tran(test_vecs, 20)
  vecs = se23op.tran2vec(mats)

  # going back and forth
  assert np.allclose(test_vecs, vecs)
  # numerical vs analytical
  assert np.allclose(mats, mats_numerical, atol=1e-6)


def test_vec2jac_vec2jacinv():

  test_vecs = np.random.uniform(-np.pi / 2, np.pi / 2, size=(TEST_SIZE, 9, 1))

  jac = se23op.vec2jac(test_vecs)
  jac_numerical = se23op.vec2jac(test_vecs, num_terms=20)
  jacinv = se23op.vec2jacinv(test_vecs)
  jacinv_numerical = se23op.vec2jacinv(test_vecs, num_terms=20)

  # jacinv vs inverse of jac
  assert np.allclose(jac, npla.inv(jacinv))
  # numerical vs analytical
  assert np.allclose(jac, jac_numerical, atol=1e-6)
  assert np.allclose(jacinv, jacinv_numerical, atol=1e-4)


def test_vec2jac_vec2jacinv_special_cases():

  test_vecs = np.expand_dims(np.array([
      [0, 0, 0, 0, 0, 0, 0, 0, 0],
      [1, 1, 1, 1, 1, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, np.pi, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, np.pi, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, np.pi],
      [0, 0, 0, 0, 0, 0, np.pi / 2, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, np.pi / 2, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, np.pi / 2],
  ]),
                             axis=-1)
  jac = se23op.vec2jac(test_vecs)
  jac_numerical = se23op.vec2jac(test_vecs, num_terms=20)
  jacinv = se23op.vec2jacinv(test_vecs)
  jacinv_numerical = se23op.vec2jacinv(test_vecs, num_terms=20)

  # jacinv vs inverse of jac
  assert np.allclose(jac, npla.inv(jacinv))
  # numerical vs analytical
  assert np.allclose(jac, jac_numerical, atol=1e-6)
  assert np.allclose(jacinv, jacinv_numerical, atol=1e-6)

def test_tranAd():

  test_vecs = np.random.uniform(-np.pi, np.pi, size=(TEST_SIZE, 9, 1))

  # identity, Ad(T(v)) = I + curlyhat(v)*J(v)
  lhs = se23op.tranAd(se23op.vec2tran(test_vecs))
  rhs = np.tile(np.eye(9), (TEST_SIZE, 1, 1)) + se23op.curlyhat(test_vecs) @ se23op.vec2jac(test_vecs)
  assert np.allclose(lhs, rhs)
#
#
#def test_point2fs_point2sf():
#
#  test_vecs = np.random.uniform(-np.pi, np.pi, size=(TEST_SIZE, 4, 1))
#
#  _ = se23op.point2fs(test_vecs)
#  _ = se23op.point2sf(test_vecs)
