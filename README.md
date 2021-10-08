# pylgmath

pylgmath is a Python library for handling geometry in state estimation problems in robotics.
It is used to store, manipulate, and apply three-dimensional rotations and transformations and their associated uncertainties.

There are no minimal, constraint-free, singularity-free representations for these quantities, so lgmath exploits two different representations for the nominal and noisy parts of the uncertain random variable.

- Nominal rotations and transformations are represented using their composable, singularity-free _matrix Lie groups_, _SO(3)_ and _SE(3)_.
- Their uncertainties are represented as multiplicative perturbations on the minimal, constraint-free vectorspaces of their _Lie algebras_, **\*so\*\***(3)\* and **\*se\*\***(3)\*.

This library uses concepts and mathematics described in Timothy D. Barfoot's book [State Estimation for Robotics](asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf).
It is used for robotics research at the Autonomous Space Robotics Lab; most notably in the STEAM Engine, a library for Simultaneous Trajectory Estimation and Mapping.

## [License](./LICENSE)
