#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.distributions as tdist
import pickle


class muellerBrownPotential(nn.Module):
    def __init__(self):
        # According to:
        # Location of SaddlePoints and Minimum Energy Paths by a ConstrainedSimplex Optimization Procedure
        self.A = [-200, -100, -170, 15]
        self.b = [0, 0, 11, 0.6]
        self.a = [-1, -1, -6.5, 0.7]
        self.c = [-10, -10, -6.5, 0.7]
        self.X_bar = [[1, 0], [0, 0.5], [-0.5, 1.5], [-1, 1]]

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

    def plotPotential(self,
                      save=False,
                      minx=-2,
                      maxx=2,
                      miny=-1,
                      maxy=3,
                      ax=None,
                      cmap=plt.get_cmap("Reds"),
                      title=True,
                      colorbar=True,
                      left=False):
        grid_width = max(maxx - minx, maxy - miny) / 300.0
        X = np.mgrid[minx:maxx:grid_width, miny:maxy:grid_width]
        X = Variable(torch.tensor(X))
        V = self.potential(X)
        V -= V.min() - 1e-6

        V = V.cpu().numpy()
        X = X.cpu().numpy()

        V = np.log(V)
        # V = V.clamp(max=7, min=3)
        V = np.clip(V, 3, 7)
        vmin = np.nanmin(V[V != -np.inf])
        vmax = np.nanmax(V)
        if ax is None: fig, ax = plt.subplots()
        mp = ax.contourf(X[0],
                         X[1],
                         V,
                         24,
                         cmap=cmap,
                         levels=np.linspace(vmin, vmax, 24),
                         extend='both')
        # Setting integer tick labels
        vmin_int = int(vmin)
        vmax_int = int(vmax)
        ticks = np.linspace(vmin_int, vmax_int, int(vmax_int - vmin_int) + 1)
        if colorbar:
            if not left:
                cbar = plt.colorbar(mp, ax=ax, ticks=ticks)
                # cbar = plt.colorbar(mp, ax=ax, ticks=ticks, fraction=0.05, pad=0.2)
                # cbar = plt.colorbar(mp, ax=ax, ticks=ticks, fraction=0.05, pad=0.2)
                cbar.set_label('$\log(V)$', rotation=0, labelpad=30)
            else:
                cbar = plt.colorbar(mp,
                                    ax=[ax],
                                    location='left',
                                    ticks=ticks,
                                    pad=0.15)
                cbar.set_label('$\log(V)$', rotation=0, labelpad=30)

        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        if title: plt.title(r'Brown-M\"{u}ller Potential')
        if save == True:
            plt.savefig("./Figures/Muller_Brown_Potential.png")
            # plt.show()
        return ax

    def runTrajectory(self, time_steps, x0, kT, dt, mGamma):
        """
        Langevin Dynamics
        Transition path sampling and the calculation of rate constants, Equation (A3)
        Variance of noise given by 2*kT*dt/(m*gamma)
        where gamma is the friction coefficient, m is the mass
        """
        noise = tdist.Normal(
            0.0, torch.sqrt(torch.tensor((2.0 * kT * dt) / mGamma)))
        noise_ = noise.sample((time_steps - 1, 2))

        traj = torch.zeros((time_steps, 2))

        x0 = torch.tensor(x0)
        traj[0] = x0

        for t in range(time_steps - 1):
            if t % 1000 == 0:
                print("# {:}/{:} simulated. {:3.2f} % #".format(
                    t, time_steps - 1, t / (time_steps - 1) * 100))
            traj[t +
                 1] = traj[t] + dt * self.force(traj[t]) / mGamma + noise_[t]
        traj = traj.clone().detach().numpy()
        return traj

    def plotTrajectories(self, trajectories, save=False):
        fig, ax = self.plotPotential(save=False)
        for traj in trajectories:
            traj = np.array(traj)
            ax.plot(traj[:, 0], traj[:, 1], 'x-')
        if save == True: plt.savefig("./Figures/Muller_Brown_Trajectories.pdf")
