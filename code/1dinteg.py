from __future__ import division
from nengo.utils.functions import piecewise
from nengo.utils.matplotlib import rasterplot
from scipy.integrate import odeint

import numpy as np
import nengo 
import matplotlib.pyplot as plt
import raphan_matrices as rmat

"""
# EYE GLOBE DYNAMICS
# ====
# omega is 3x1
# phi is scalar
# n is 3x1 
# ====
def eye_globe(state, torque, t):
    phi, n, torque = state # unpack rotation state

    if phi == 0:
        n = torque / np.linalg.norm(torque)

    f = [-(rmat.BJ * omega + KJ * phi * n) + np.linalg.inv(rmat.J) * torque,
         np.dot(omega, n),
         np.cross(omega, n) / 2 + np.cross(n, np.cross(omega, n)) / 2 / np.tan(phi / 2)
         ] 

    return f

def eye_globe_solver(t):
    state = [] # simulated result
    for torque in t:
        f = [-(rmat.BJ * omega + KJ * phi * n) + np.linalg.inv(rmat.J) * torque,
             np.dot(omega, n),
             np.cross(omega, n) / 2 + np.cross(n, np.cross(omega, n)) / 2 / np.tan(phi / 2)
             ] 

        f = f * dt
        state.append(f)

    return state
"""
        
# MODEL PARAMETERS
# =====
tau = rmat.tau
tau_c = rmat.tau_drift 
# ====
model = nengo.Network('Eye control', seed=5)

with model:
    stim_pitch = nengo.Node(piecewise({0:1, .5:0}))
    stim_roll = nengo.Node(piecewise({0:0, .5:0}))
    stim_yaw = nengo.Node(piecewise({0:0, .8:1, .9:0}))

    velocity = nengo.Ensemble(500, dimensions=3)
    position = nengo.Ensemble(500, dimensions=3)
    motorneurons = nengo.Ensemble(500, dimensions=3)
    torque = nengo.Ensemble(500, dimensions=3)
    
    def feedback(x):
        return (-tau/tau_c + 1)*x # approximation of synaptic dynamics
    
    conn_stim = nengo.Connection(stim_pitch, velocity[0])
    conn_stim = nengo.Connection(stim_roll, velocity[1])
    conn_stim = nengo.Connection(stim_yaw, velocity[2])

    conn_vp = nengo.Connection(velocity, position, transform=tau, synapse=tau)
    conn_fb = nengo.Connection(position, position, function=feedback, synapse=tau)

    # premotor - motor coupling
    conn_pmn = nengo.Connection(position, motorneurons)
    conn_vmn = nengo.Connection(velocity, motorneurons)

    # transduction
    conn_torque = nengo.Connection(motorneurons, torque, transform=2.493e-2, synapse=tau)

    stim_pitch = nengo.Probe(stim_pitch)
    stim_roll = nengo.Probe(stim_roll)
    stim_yaw = nengo.Probe(stim_yaw)

    position_p = nengo.Probe(position[0], synapse=.01)
    velocity_p = nengo.Probe(velocity[0], synapse=.01)
    motorneurons_p = nengo.Probe(motorneurons[0], synapse=.01)
    torque_p = nengo.Probe(torque[0], synapse=.01)
    
    position_r = nengo.Probe(position[1], synapse=.01)
    velocity_r = nengo.Probe(velocity[1], synapse=.01)
    motorneurons_r = nengo.Probe(motorneurons[1], synapse=.01)
    torque_r = nengo.Probe(torque[1], synapse=.01)

    position_y = nengo.Probe(position[2], synapse=.01)
    velocity_y = nengo.Probe(velocity[2], synapse=.01)
    motorneurons_y = nengo.Probe(motorneurons[2], synapse=.01)
    torque_y = nengo.Probe(torque[2], synapse=.01)

# Simulate neurons
sim = nengo.Simulator(model)
sim.run(1)

plt.figure(1)
plt.subplot(311)
plt.plot(sim.trange(), sim.data[stim_pitch], label = "pitch")
plt.plot(sim.trange(), sim.data[position_p], label = "position")
plt.plot(sim.trange(), sim.data[velocity_p], label = "velocity")
plt.legend(loc="best");

plt.subplot(312)
plt.ylim([-0.2, 1.2])
plt.plot(sim.trange(), sim.data[stim_roll], label = "roll")
plt.plot(sim.trange(), sim.data[position_r], label = "position")
plt.plot(sim.trange(), sim.data[velocity_r], label = "velocity")
plt.legend(loc="best");

plt.subplot(313)
plt.plot(sim.trange(), sim.data[stim_yaw], label = "yaw")
plt.plot(sim.trange(), sim.data[position_y], label = "position")
plt.plot(sim.trange(), sim.data[velocity_y], label = "velocity")
plt.legend(loc="best");

plt.figure(2)
plt.title('Motor neurons')
plt.subplot(311)
plt.plot(sim.trange(), sim.data[motorneurons_p], label = "pitch motor neurons")
plt.legend(loc="best");

plt.subplot(312)
plt.ylim([-0.2, 1.2])
plt.plot(sim.trange(), sim.data[motorneurons_r], label = "roll motor neurons")
plt.legend(loc="best");

plt.subplot(313)
plt.plot(sim.trange(), sim.data[motorneurons_y], label = "yaw motor neurons")
plt.legend(loc="best");

"""
plt.figure(3)
plt.plot(sim.trange(), sim.data[torque_p], label = "torque output")
plt.legend(loc="best");
"""

plt.show()
