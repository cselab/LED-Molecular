#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.distributions as tdist
import pickle
import time
import os

torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_tensor_type(torch.FloatTensor)

class muellerBrownPotential(nn.Module):
    def __init__(self):
        # According to:
        # Location of SaddlePoints and Minimum Energy Paths by a ConstrainedSimplex Optimization Procedure
        self.A = [-200, -100, -170, 15]
        self.b = [0,0,11,0.6]
        self.a = [-1, -1, -6.5, 0.7]
        self.c = [-10, -10, -6.5, 0.7]
        self.X_bar = [[1, 0], [0, 0.5], [-0.5,1.5], [-1,1]]
        self.ndims = 2
        self.noise = tdist.Normal(0.0, 1.0)

    def potential(self, x):
        V = Variable(torch.zeros(x.size()[1:]))
        for i in range(4):
            V += self.A[i]*( torch.exp(self.a[i]*torch.pow(x[0]-self.X_bar[i][0],2) \
                + self.b[i]*(x[0]-self.X_bar[i][0])*(x[1]-self.X_bar[i][1]) \
                + self.c[i]*torch.pow(x[1]-self.X_bar[i][1],2)) )
        return V

    def force(self, position):
        position = position.clone().detach().requires_grad_(True)
        self.v = self.potential(position)
        self.v = -self.v
        self.v.backward(retain_graph=True)
        self.F = position.grad.clone().detach()
        position.grad.zero_()
        return self.F

    # def plotPotential(self, save=False, minx=-1.5, maxx=1.5, miny=-0.5, maxy=2.5):
    def plotPotential(self, save=False, minx=-2, maxx=2, miny=-1, maxy=3, ax=None):
        grid_width = max(maxx-minx, maxy-miny) / 300.0
        X = np.mgrid[minx : maxx : grid_width, miny : maxy : grid_width]
        X = Variable(torch.tensor(X))
        V = self.potential(X)
        V -= V.min() - 1e-6
        V = np.log(V)
        V = V.clamp(max=7, min=3)
        vmin = np.nanmin(V[V != -np.inf])
        vmax = np.nanmax(V)
        if ax is None: fig, ax = plt.subplots()
        mp = ax.contourf(X[0], X[1], V, 24, cmap=plt.get_cmap("Reds"), levels=np.linspace(vmin, vmax, 24), extend='both')
        cbar = fig.colorbar(mp, ax=ax)
        cbar.set_label('$\log(V)$', rotation=0, labelpad=20)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('Brown-Muller Potential')
        if save==True:
            plt.savefig("./Figures/Muller_Brown_Potential.png")
            # plt.show()
        return fig, ax

    def runTrajectories(self, time_steps, masses, initial_positions, kB, Temp, gamma, dt, save_every=1):
        """
        Langevin Dynamics
        Variance of noise given by 2*kT*dt/(m*gamma)
        where gamma is the friction coefficient, m is the mass
        """
        masses = torch.tensor(masses)
        X0 = torch.tensor(initial_positions)

        N_particles = masses.size()[0]
        self.N_particles = N_particles
        print("Simulating {:} particles.".format(N_particles))

        assert(time_steps % save_every == 0)
        time_steps_save = int(time_steps/save_every)
        traj = torch.zeros((time_steps_save, N_particles, 2))
        traj[0] = X0
        positions = X0

        # Calculate initial force
        forces = self.getForces(traj[0])
        velocities = torch.zeros_like(forces)

        for t in range(time_steps-1):
            if t % 1000==0: print("# {:}/{:} simulated. {:3.2f} % #".format(t, time_steps-1, t/(time_steps-1)*100))
            # positions = traj[t]

            new_half_vel = self.getHalfVelocities(velocities, dt, forces, gamma, kB, Temp, masses)
            new_positions = self.getNewPositions(positions, new_half_vel, dt)
            new_forces = self.getForces(new_positions)
            new_velocities = self.getNewVelocities(new_half_vel, new_forces, gamma, kB, Temp, dt, masses)

            # print("####")
            # print(forces)
            # print(new_half_vel)
            # print(new_positions)
            # print(new_forces)
            # print(new_velocities)

            velocities = new_velocities
            forces = new_forces
            positions = new_positions
            # traj[t+1] = new_positions
            if (t+1) % save_every == 0:
                print("Saving time-step {:}".format(t+1))
                indx = int((t+1)/save_every)
                traj[indx] = new_positions


        traj = traj.clone().detach().numpy()
        # Exchanging the time - N_particles axis
        traj = np.swapaxes(traj, 0,1)
        return traj

    def getNewPositions(self, positions, new_half_vel, dt):
        positions += new_half_vel * dt
        return positions

    def getForces(self, positions):
        N_particles = positions.size()[0]
        forces = []
        # Calculating new forces
        for i in range(N_particles):
            force = self.force(positions[i])
            forces.append(force)
        forces = torch.stack(forces)
        return forces

    def getNewVelocities(self, new_half_velocities, new_forces, gamma, kB, Temp, dt, masses):
        # Calculating the velocity of the particles at time t + dt/2
        N_particles, dim = new_half_velocities.size()
        root_term = dt * kB * Temp * gamma * (1/masses)
        # root_term = dt * 2.0 * kB * Temp * gamma * (1/masses)
        root_term = torch.sqrt(root_term)
        # The same along all dimensions (x-y-z)
        root_term = torch.reshape(root_term, (-1,1))
        masses = torch.reshape(masses, (-1,1))
        noise_term = root_term * self.noise.sample((N_particles, dim))

        new_velocities = new_half_velocities + \
                        0.5*dt*(new_forces * (1/masses) - gamma * new_half_velocities) + \
                        noise_term
        return new_velocities


    def getHalfVelocities(self, velocities, dt, forces, gamma, kB, Temp, masses):
        # Calculating the velocity of the particles at time t + dt/2
        N_particles, dim = velocities.size()
        # torch.sqrt(torch.tensor((2.0*kT*dt)/mGamma))

        # root_term = dt * 2.0 * kB * Temp * gamma * (1/masses)
        root_term = dt * kB * Temp * gamma * (1./masses)
        root_term = torch.sqrt(root_term)

        # The same along all dimensions (x-y-z)
        root_term = torch.reshape(root_term, (-1,1))
        masses = torch.reshape(masses, (-1,1))

        noise_term = root_term * self.noise.sample((N_particles, dim))

        new_half_vel = velocities + \
                        0.5 * dt * ((forces/masses) - gamma * velocities) + \
                        noise_term
        return new_half_vel


    def plotTrajectories(self, trajectories, save=False):
        # print(np.shape(trajectories))
        minx = np.min(trajectories[:,:,0]) - 0.25
        miny = np.min(trajectories[:,:,1]) - 0.25
        maxx = np.max(trajectories[:,:,0]) + 0.25
        maxy = np.max(trajectories[:,:,1]) + 0.25
        fig, ax = self.plotPotential(save=False, minx=minx, maxx=maxx, miny=miny, maxy=maxy)
        for traj in trajectories:
            traj = np.array(traj)
            ax.plot(traj[:, 0], traj[:, 1], 'x-')
        plt.show()
        if save==True: plt.savefig("./Figures/Muller_Brown_Trajectories.png")

if __name__ == '__main__':
    mbp = muellerBrownPotential()
    # mbp.plotPotential(save=False)
    
    # time_steps = 1100000
    # N_particles = 100
    # T_SUBSAMPLE = 10
    # dt = 0.001

    # time_steps = 1100000
    # N_particles = 20
    # T_SUBSAMPLE = 10
    # dt = 0.001

    # time_steps = 110
    # N_particles = 10
    # T_SUBSAMPLE = 10
    # dt = 0.001

    # time_steps = 1000000
    # N_particles = 1
    # T_SUBSAMPLE = 10
    # dt = 0.001

    N_particles_total = 100
    # N_particles_total = 10

    for ic in range(N_particles_total):


        time_steps = 1100000
        # time_steps = 1100
        T_SUBSAMPLE = 1
        dt = 0.01

        N_particles = 1

        Temp = 15.0 # Temperature
        kB = 1.0    # Boltzmann constant
        gamma = 1.0  # Friction parameter for Langevin Thermostat (gamma = 0.0 for NVE simulation)
        # dt = 0.001
        masses = [1.0] * N_particles
        masses = np.array(masses)

        # How many particles ?
        x0 = np.random.uniform(-1.5, 1.2, N_particles)
        x1 = np.random.uniform(-0.2, 2.0, N_particles)
        X0 = np.stack((x0, x1), axis=1)

        print(np.shape(masses))
        print(np.shape(X0))
        print(X0)
        assert(len(np.shape(masses))==1)
        assert(len(np.shape(X0))==2)

        start_time = time.time()

        trajectories = mbp.runTrajectories(time_steps, masses, X0, kB, Temp, gamma, dt, save_every=T_SUBSAMPLE)

        stop_time = time.time()

        total_time = stop_time - start_time
        time_per_iter = total_time/time_steps/N_particles
        time_per_iter = time_per_iter*T_SUBSAMPLE

        trajectories = np.array(trajectories)
        # trajectories = trajectories[:,::T_SUBSAMPLE]

        dt = T_SUBSAMPLE * dt
        trajectories = np.array(trajectories)

        print("Time per iteration: {:}".format(time_per_iter))
        data = {
            "time_per_iter":time_per_iter,
            "trajectories":trajectories,
            "dt":dt,
            "N_particles":N_particles,
        }
        data_path = "./Simulation_Data/"
        os.makedirs(data_path, exist_ok=True)

        with open(data_path + "/trajectories_ic_{:}.pickle".format(ic), "wb") as file:
            # Pickle the "data" dictionary using the highest protocol available.
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

        # # PLOTTING
        # mbp.plotTrajectories(trajectories, save=False)


