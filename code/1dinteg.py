from __future__ import division
from nengo.utils.functions import piecewise
from nengo.utils.matplotlib import rasterplot
from scipy.integrate import odeint

import numpy as np
import nengo 
import matplotlib.pyplot as plt
import raphan_matrices as rmat

# EYE GLOBE DYNAMICS
# ====
# omega is 3x1
# phi is scalar
# n is 3x1 
# torque is 3d
# ====
def eye_globe(t):
    dt = 0.001

    phi = 0
    n = np.asarray([[0,0,0]]).T
    w = 0
    n_init = False
    
    simulation = []

    for idx, torque in enumerate(t):
        # Update angular velocity
        if idx == 0:
            w = np.dot(np.linalg.inv(rmat.J), torque) * dt # initialize w 

        else:
            w_step = -1 * (np.dot(rmat.BJ, w) + phi*np.dot(rmat.KJ, n) + np.dot(np.linalg.inv(rmat.J), torque))
            w_step = w_step * dt
            w += w_step

        # Update rotation axis
        if not n_init and np.any(torque):
            n = w / np.linalg.norm(w)
            n_init = True

        elif n_init:
            n_step = np.cross(w, n, axis=0) / 2 + np.cross(n, np.cross(w, n, axis=0), axis=0) 

            if not phi == 0:
                n_step = n_step / (2*np.tan(phi/2))
            
            n_step = n_step * dt
            n += n_step

        # Update rotation angle
        phi_step = np.dot(w.T, n) * dt
        phi_step = phi_step[0]
        phi += phi_step

        state = [w, phi, n]
        print state
        simulation.append(state)

    return simulation

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

    velocity = nengo.Ensemble(1000, dimensions=3)
    position = nengo.Ensemble(1000, dimensions=3)
    motorneurons = nengo.Ensemble(1000, dimensions=3)
    torque = nengo.Ensemble(1000, dimensions=3)
    
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

# ====
# Neuron population graphs
plt.figure(1)
plt.suptitle('Pitch-roll-yaw neuron population activity')
plt.subplot(311)
plt.plot(sim.trange(), sim.data[stim_pitch], label = "input")
plt.plot(sim.trange(), sim.data[position_p], label = "position")
plt.plot(sim.trange(), sim.data[velocity_p], label = "velocity")
plt.title('Pitch')
plt.legend(loc="best");

plt.subplot(312)
plt.ylim([-0.2, 1.2])
plt.plot(sim.trange(), sim.data[stim_roll], label = "input")
plt.plot(sim.trange(), sim.data[position_r], label = "position")
plt.plot(sim.trange(), sim.data[velocity_r], label = "velocity")
plt.title('Roll')
plt.legend(loc="best");

plt.subplot(313)
plt.plot(sim.trange(), sim.data[stim_yaw], label = "input")
plt.plot(sim.trange(), sim.data[position_y], label = "position")
plt.plot(sim.trange(), sim.data[velocity_y], label = "velocity")
plt.title('Yaw')
plt.legend(loc="best");
# ====

# ====
# Motor neuron graphs
plt.figure(2)
plt.suptitle('Motor neuron activity')
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
# ====

# ====
# Applied torque graphs
plt.figure(3)
plt.suptitle('Applied torque (kg m^2 / s^2)')
plt.subplot(311)
plt.ylim([-0.02, 0.07])
plt.plot(sim.trange(), sim.data[torque_p], label = "pitch")
plt.legend(loc="best");

plt.subplot(312)
plt.ylim([-0.02, 0.07])
plt.plot(sim.trange(), sim.data[torque_r], label = "roll")
plt.legend(loc="best");

plt.subplot(313)
plt.ylim([-0.02, 0.07])
plt.plot(sim.trange(), sim.data[torque_y], label = "yaw")
plt.legend(loc="best");
# ====

torque = []
for i in range(0, len(sim.data[torque_p])):
    torque.append([sim.data[torque_p][i], sim.data[torque_r][i], sim.data[torque_y][i]])

kinem = eye_globe(np.asarray(torque))

#plt.show()
