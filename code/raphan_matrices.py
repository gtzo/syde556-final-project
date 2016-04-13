"""
Matrix parameters for 3D eye model 
Taken from Schnabolk and Raphan (1994)

"The values of the matrix parameters were picked to obtain
states of the vel-pos integrator to match the approx.
orientation of the eye in steady state"
"""

import numpy as np

Hp = -0.03333 * np.identity(3)
Gp = 0.3333 * np.identity(3)
Cp = 29.44 * np.identity(3)
D = 0.1389 * np.identity(3)

