from __future__ import division
from nengo.utils.functions import piecewise

import numpy as np
import nengo 
import matplotlib.pyplot as plt
import raphan_matrices as rmat

# MODEL PARAMETERS
# =====
TAU_RECUR = 0.1
input_func = input_func = piecewise({0: 0, 0.2: 5, 0.3: 0, 0.44: -10, 0.54: 0, 0.8: 5, 0.9: 0})
control_func = piecewise({0: 1, 0.6: 0.5}) # function to define integrator behaviour

model = nengo.Network(label='1dinteg')

with model:
    A = nengo.Ensemble(200, dimensions=2, radius=1.5)

    inp = nengo.Node(input_func)
    control = nengo.Node(output=control_func)

    # wire the stimulus input
    nengo.Connection(inp, A, transform=[[TAU_RECUR], [0]], synapse=TAU_RECUR)
    # wire the control quality input  
    nengo.Connection(control, A[1], synapse=0.005)

    # set up the recurrent input
    nengo.Connection(A, A[0],
                     function = lambda x: x[0] * x[1],
                     synapse=TAU_RECUR)
    
    A_probe = nengo.Probe(A, 'decoded_output', synapse=0.01)

# create sim
sim = nengo.Simulator(model)
sim.run(1.4)

# plot
t = sim.trange()
dt = t[1]-t[0]

input_sig = list(map(input_func, t))
control_sig = list(map(control_func, t))
ref = dt * np.cumsum(input_sig)

plt.figure(figsize=(6, 8))
plt.subplot(2, 1, 1)
plt.plot(t, input_sig, label='Input')
plt.ylim(-11, 11)
plt.ylabel('Input')
plt.legend(loc="lower left", frameon=False)

plt.subplot(2, 1, 2)
plt.plot(t, ref, 'k--', label='Exact')
plt.plot(t, sim.data[A_probe][:,0], label='A (value)')
plt.plot(t, sim.data[A_probe][:,1], label='A (control)')
plt.ylim([-1.1, 1.1])
plt.xlabel('Time (s)')
plt.ylabel('x(t)')
plt.legend(loc="lower left", frameon=False)
plt.show()
