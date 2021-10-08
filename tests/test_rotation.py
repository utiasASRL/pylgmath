import numpy as np
import numpy.linalg as npla

from pylgmath import so3op, Rotation

TEST_SIZE = 10000


def test_constructor():
  # generate random transform from the most basic constructor for testing
  aaxis_ab_rand = np.random.uniform(-np.pi / 2, np.pi / 2, size=(TEST_SIZE, 3, 1))
  C_ba_rand = so3op.vec2rot(aaxis_ab_rand)
  rotation_rand = Rotation(C_ba=C_ba_rand)

  # default constructor
  test = Rotation()
  assert np.allclose(test.matrix(), np.eye(3))

  # copy constructor
  test = Rotation(rotation=rotation_rand)
  assert np.allclose(test.matrix(), rotation_rand.matrix())

  # construct from invalid C_ba with reprojection (ones to identity)
  test = Rotation(C_ba=np.ones((3, 3)))
  assert np.allclose(test.matrix(), np.eye(3))

  # construct from axis-angle vec (analytical)
  test = Rotation(aaxis_ab=aaxis_ab_rand)
  assert np.allclose(test.matrix(), rotation_rand.matrix())

  # construct from axis-angle vec (numerical)
  test = Rotation(aaxis_ab=aaxis_ab_rand, num_terms=20)
  assert np.allclose(test.matrix(), rotation_rand.matrix(), atol=1e-6)


def test_se3algebra():
  # generate random transform from the most basic constructor for testing
  aaxis_ab_rand = np.random.uniform(-np.pi / 2, np.pi / 2, size=(TEST_SIZE, 3, 1))

  # construct from axis-angle and then call .vec() to get se3 algebra vector.
  test = Rotation(aaxis_ab=aaxis_ab_rand)
  assert np.allclose(test.vec(), aaxis_ab_rand)


def test_inverse():
  # generate random transform from the most basic constructor for testing
  aaxis_ab_rand = np.random.uniform(-np.pi / 2, np.pi / 2, size=(TEST_SIZE, 3, 1))
  C_ba_rand = so3op.vec2rot(aaxis_ab_rand)

  # rotations to be tested
  test = Rotation(aaxis_ab=aaxis_ab_rand)
  test_inv = test.inverse()

  # compare to basic matrix inverse
  assert np.allclose(test_inv.matrix(), npla.inv(C_ba_rand))

  # product of inverse and self make identity
  assert np.allclose(test.matrix() @ test_inv.matrix(), np.eye(3))
  assert np.allclose((test * test_inv).matrix(), np.eye(3))