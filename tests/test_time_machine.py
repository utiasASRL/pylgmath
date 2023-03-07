import numpy as np
import numpy.linalg as npla

from pylgmath import tmop, TimeMachine, se23op

TEST_SIZE = 1000


def test_constructor():
  # generate random transform from the most basic constructor for testing
  tau_rand = np.random.uniform(-10, 10, size=(TEST_SIZE, 1))
  D_rand = tmop.vec2tran(tau_rand)
  tm_rand = TimeMachine(tau=tau_rand)

  # default constructor
  test = TimeMachine()
  assert np.allclose(test.matrix(), np.eye(5))

  # copy constructor
  test = TimeMachine(time_machine=tm_rand)
  assert np.allclose(test.matrix(), tm_rand.matrix())

  # construct from algebra vec (analytical)
  test = TimeMachine(tau=tau_rand)
  assert np.allclose(test.matrix(), tm_rand.matrix())

def test_se3algebra():
  # generate random transform from the most basic constructor for testing
  tau_rand = np.random.uniform(-10, 10, size=(TEST_SIZE, 1))

  # construct from axis-angle and then call .vec() to get se3 algebra vector.
  test = TimeMachine(tau=tau_rand)
  assert np.allclose(test.vec(), tau_rand)


def test_inverse():
  # generate random transform from the most basic constructor for testing
  tau_rand = np.random.uniform(-10, 10, size=(TEST_SIZE, 1))

  D_rand = tmop.vec2tran(tau_rand)

  # transformations to be tested
  test = TimeMachine(tau=tau_rand)
  test_inv = test.inverse()

  # compare to basic matrix inverse
  assert np.allclose(test_inv.matrix(), npla.inv(D_rand))

  # product of inverse and self make identity
  assert np.allclose(test.matrix() @ test_inv.matrix(), np.eye(5))
  assert np.allclose((test * test_inv).matrix(), np.eye(5))

def test_conjugation():
  # generate random transform from the most basic constructor for testing
  tau_rand = np.random.uniform(-10, 10, size=(TEST_SIZE, 1))
  xi_rand  = np.random.uniform(-10, 10, size=(TEST_SIZE, 9, 1))
  D_rand = tmop.vec2tran(tau_rand)
  Di_rand = tmop.vec2tran(-tau_rand)
  X_rand = tmop.vec2se23conjugation(tau_rand)

  assert np.allclose(se23op.hat(X_rand @ xi_rand), Di_rand @ se23op.hat(xi_rand) @ D_rand)
