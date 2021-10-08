import numpy as np
import numpy.linalg as npla

from pylgmath import se3op

TEST_SIZE = 10000


def test_hat_curlyhat():

  test_vecs = np.random.uniform(-np.pi, np.pi, size=(TEST_SIZE, 6, 1))

  mats = se3op.hat(test_vecs)
  curlymats = se3op.curlyhat(test_vecs)


def test_vec2tran_tran2vec():

  test_vecs = np.random.uniform(-np.pi / 2, np.pi / 2, size=(TEST_SIZE, 6, 1))

  mats = se3op.vec2tran(test_vecs)
  mats_numerical = se3op.vec2tran(test_vecs, 20)
  vecs = se3op.tran2vec(mats)

  # going back and forth
  assert np.allclose(test_vecs, vecs)
  # numerical vs analytical
  assert np.allclose(mats, mats_numerical, atol=1e-4)


def test_vec2tran_tran2vec_special_cases():

  test_vecs = np.expand_dims(np.array([
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, np.pi, 0, 0],
      [0, 0, 0, 0, np.pi, 0],
      [0, 0, 0, 0, 0, np.pi],
      [0, 0, 0, np.pi / 2, 0, 0],
      [0, 0, 0, 0, np.pi / 2, 0],
      [0, 0, 0, 0, 0, np.pi / 2],
  ]),
                             axis=-1)

  mats = se3op.vec2tran(test_vecs)
  mats_numerical = se3op.vec2tran(test_vecs, 20)
  vecs = se3op.tran2vec(mats)

  # going back and forth
  assert np.allclose(test_vecs, vecs)
  # numerical vs analytical
  assert np.allclose(mats, mats_numerical, atol=1e-6)


def test_vec2jac_vec2jacinv():

  test_vecs = np.random.uniform(-np.pi / 2, np.pi / 2, size=(TEST_SIZE, 6, 1))

  jac = se3op.vec2jac(test_vecs)
  jac_numerical = se3op.vec2jac(test_vecs, num_terms=20)
  jacinv = se3op.vec2jacinv(test_vecs)
  jacinv_numerical = se3op.vec2jacinv(test_vecs, num_terms=20)

  # jacinv vs inverse of jac
  assert np.allclose(jac, npla.inv(jacinv))
  # numerical vs analytical
  assert np.allclose(jac, jac_numerical, atol=1e-6)
  assert np.allclose(jacinv, jacinv_numerical, atol=1e-4)


def test_vec2jac_vec2jacinv_special_cases():

  test_vecs = np.expand_dims(np.array([
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, np.pi, 0, 0],
      [0, 0, 0, 0, np.pi, 0],
      [0, 0, 0, 0, 0, np.pi],
      [0, 0, 0, np.pi / 2, 0, 0],
      [0, 0, 0, 0, np.pi / 2, 0],
      [0, 0, 0, 0, 0, np.pi / 2],
  ]),
                             axis=-1)

  jac = se3op.vec2jac(test_vecs)
  jac_numerical = se3op.vec2jac(test_vecs, num_terms=20)
  jacinv = se3op.vec2jacinv(test_vecs)
  jacinv_numerical = se3op.vec2jacinv(test_vecs, num_terms=20)

  # jacinv vs inverse of jac
  assert np.allclose(jac, npla.inv(jacinv))
  # numerical vs analytical
  assert np.allclose(jac, jac_numerical, atol=1e-6)
  assert np.allclose(jacinv, jacinv_numerical, atol=1e-6)


def test_tranAd():

  test_vecs = np.random.uniform(-np.pi, np.pi, size=(TEST_SIZE, 6, 1))

  # identity, Ad(T(v)) = I + curlyhat(v)*J(v)
  lhs = se3op.tranAd(se3op.vec2tran(test_vecs))
  rhs = np.tile(np.eye(6), (TEST_SIZE, 1, 1)) + se3op.curlyhat(test_vecs) @ se3op.vec2jac(test_vecs)
  assert np.allclose(lhs, rhs)


def test_point2fs_point2sf():

  test_vecs = np.random.uniform(-np.pi, np.pi, size=(TEST_SIZE, 4, 1))

  _ = se3op.point2fs(test_vecs)
  _ = se3op.point2sf(test_vecs)
