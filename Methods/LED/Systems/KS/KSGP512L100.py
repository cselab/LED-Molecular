#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
from .ks_solver.KS import *


def evolveKSGP512L100(u0, total_time, dt_model, tstart):
    u0 = np.reshape(u0, (-1))
    #------------------------------------------------------------------------------
    # define data and initialize simulation
    L = 100 / (2 * pi)
    N = 512
    # dt   = dt_model
    # dt   = 0.025
    dt = 0.0025
    dns = KS(L=L, N=N, dt=dt, tend=total_time, u0=u0)
    #------------------------------------------------------------------------------
    # simulate initial transient
    dns.simulate()
    # convert to physical space
    dns.fou2real()
    # print(np.shape(u0))
    # print(np.shape(dns.uu[0]))
    # print(np.linalg.norm(u0-dns.uu[0]))
    # print(ark)
    # print(dt_model)
    subsample = int(dt_model / dt)
    # u = dns.uu[1:]
    u = dns.uu
    # Subsampling
    u = u[::subsample]
    # Removing the first time-step
    u = u[1:]
    # print(np.shape(u))
    # print(ark)
    u = np.reshape(u, (-1, N))
    return u
