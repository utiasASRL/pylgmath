import numpy as np
import numpy.linalg as npla

from pylgmath import se23op, VTransformation

TEST_SIZE = 10000

def test_constructor():
  # generate random transform from the most basic constructor for testing
  xi_ab_rand = np.random.uniform(-np.pi / 2, np.pi / 2, size=(TEST_SIZE, 9, 1))
  xi_ab_rand[:,6:] = np.zeros((3,1))
  T_ba_rand = se23op.vec2tran(xi_ab_rand)
  vtransformation_rand = VTransformation(T_ba=T_ba_rand)

  # default constructor
  test = VTransformation()
  assert np.allclose(test.matrix(), np.eye(5))

  # copy constructor
  test = VTransformation(vtransformation=vtransformation_rand)
  assert np.allclose(test.matrix(), vtransformation_rand.matrix())

  # construct from invalid C_ba with reprojection (ones to identity)
  T_ba_project = T_ba_invalid = np.copy(T_ba_rand[0])
  T_ba_invalid[:3, :3] = np.ones((3, 3))
  T_ba_project[:3, :3] = np.eye(3)
  test = VTransformation(T_ba=T_ba_invalid)
  assert np.allclose(test.matrix(), T_ba_project)

  # construct from se3 algebra vec (analytical)
  test = VTransformation(xi_ab=xi_ab_rand)
  assert np.allclose(test.matrix(), vtransformation_rand.matrix())

  # construct from se3 algebra vec (numerical)
  test = VTransformation(xi_ab=xi_ab_rand, num_terms=20)
  assert np.allclose(test.matrix(), vtransformation_rand.matrix(), atol=1e-6)


def test_se3algebra():
  # generate random transform from the most basic constructor for testing
  xi_ab_rand = np.random.uniform(-np.pi / 2, np.pi / 2, size=(TEST_SIZE, 9, 1))

  # construct from axis-angle and then call .vec() to get se3 algebra vector.
  test = VTransformation(xi_ab=xi_ab_rand)
  assert np.allclose(test.vec(), xi_ab_rand)


def test_inverse():
  # generate random transform from the most basic constructor for testing
  xi_ab_rand = np.random.uniform(-np.pi / 2, np.pi / 2, size=(TEST_SIZE, 9, 1))
  T_ba_rand = se23op.vec2tran(xi_ab_rand)

  # transformations to be tested
  test = VTransformation(xi_ab=xi_ab_rand)
  test_inv = test.inverse()

  # compare to basic matrix inverse
  assert np.allclose(test_inv.matrix(), npla.inv(T_ba_rand))

  # product of inverse and self make identity
  assert np.allclose(test.matrix() @ test_inv.matrix(), np.eye(5))
  assert np.allclose((test * test_inv).matrix(), np.eye(5))
