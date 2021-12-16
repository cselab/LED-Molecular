#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
import numpy as np
from scipy.integrate import ode
import sys


def lorenz(t0, u0, sigma, rho, beta):
    dudt = np.zeros(np.shape(u0))
    dudt[0] = sigma * (u0[1] - u0[0])
    dudt[1] = u0[0] * (rho - u0[2]) - u0[1]
    dudt[2] = u0[0] * u0[1] - beta * u0[2]
    return dudt


def evolveLorenz3D(u0, total_time, dt_model):
    sigma = 10
    rho = 28
    beta = 8. / 3
    dimensions = 3
    dt = 0.01
    subsampling = int(dt_model / dt)
    N_steps = int(total_time / dt)
    N_steps_model = int(total_time / dt_model)
    u0 = np.reshape(u0, (dimensions))
    # t0 = 0
    # u = np.zeros((dimensions, N_steps))
    # u = []
    # for k in range(N_steps):
    #   print(u0)
    #   u0 = u0 + dt*lorenz(t0, u0, sigma, rho, beta)
    #   t0 += dt
    #   u.append(u0.copy())
    t0 = 0
    u = []
    r = ode(lorenz)
    r.set_initial_value(u0, t0).set_f_params(sigma, rho, beta)
    k = 0
    t_model = dt_model
    while r.successful() and k < N_steps:
        # print("Iteration {:}, Time={:.2f}".format(k, r.t))
        # print(u0)
        r.integrate(r.t + dt)
        u0 = r.y.copy()
        t0 += dt
        # Only saving at dt_model time stamps
        if np.abs(r.t - t_model) < 1e-6:
            u.append(u0.copy())
            t_model += dt_model
        k += 1
    u = np.array(u)
    u = np.reshape(u, (N_steps_model, dimensions))
    return u
