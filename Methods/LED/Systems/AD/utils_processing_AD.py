#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

from .brownian_dynamics import *
import numpy as np

from ... import Utils as utils


def computeStateDistributionStatisticsSystemAD(state_dist_statistics, target,
                                               output):
    ##############################################
    ### Post-processing single sequence results
    ##############################################
    print("# computeStateDistributionStatisticsSystemAD() #")

    # (5, 801, 100, 1)
    # (5, 801, 100, 1)

    print(np.shape(target))
    print(np.shape(output))
    N_samples = np.shape(target)[2]
    D = np.shape(target)[3]
    print("Number of samples {:}".format(N_samples))
    max_target = np.amax(target)
    min_target = np.amin(target)
    bounds = [min_target, max_target]
    print("bounds {:}".format(bounds))
    LL = bounds[1] - bounds[0]
    nbins = utils.getNumberOfBins(N_samples, dim=D, rule="rice")
    print("nbins {:}".format(nbins))

    hist_data = [[
        utils.evaluateL1HistErrorVector(target[ic, t], output[ic, t], nbins,
                                        bounds)
        for t in range(np.shape(target[ic])[0])
    ] for ic in range(np.shape(target)[0])]

    error_density = np.array(
        [[hist_data[ic][t][0] for t in range(np.shape(target[ic])[0])]
         for ic in range(np.shape(target)[0])])

    error_density = np.stack(error_density)
    # print(np.shape(error_density))
    # (3, 801)

    wasserstein_distance = [[
        utils.evaluateWassersteinDistance(target[ic, t], output[ic, t])
        for t in range(np.shape(target[ic])[0])
    ] for ic in range(np.shape(target)[0])]
    wasserstein_distance = np.array(wasserstein_distance)
    print(np.shape(wasserstein_distance))

    # Average over ICS
    error_density = np.mean(error_density, axis=0)
    wasserstein_distance = np.mean(wasserstein_distance, axis=0)

    error_density_ = np.mean(error_density)
    wasserstein_distance_ = np.mean(wasserstein_distance)
    print("_" * 30)
    print("Wasserstein distance = {:}".format(wasserstein_distance_))
    print("L1_hist_error = {:}".format(error_density_))
    print("_" * 30)

    state_dist_wasserstein_distance_avg = np.mean(wasserstein_distance)
    state_dist_L1_hist_error_avg = np.mean(error_density)
    # print(wasserstein_distance)
    state_dist_statistics.update({
        "state_dist_wasserstein_distance_avg":
        state_dist_wasserstein_distance_avg,
        "state_dist_L1_hist_error_avg":
        state_dist_L1_hist_error_avg,
        "state_dist_L1_hist_error_all":
        error_density,
        "state_dist_wasserstein_distance_all":
        wasserstein_distance,
    })
    return state_dist_statistics


def prepareSolverAD1D(D, tstart):
    num_particles = 200
    dimension = 1
    L = np.array((1.0))
    boundaries = ReflectiveBoundaries(-L / 2, L / 2)
    advection = DriftCosine([1.], [0.1])
    initial_positions = np.zeros((num_particles, dimension))
    # Selection of dt
    # dt = 0.1
    dt = 0.01
    # dt = 0.001
    sim = BrownianDynamics(initial_positions, D, dt, boundaries, advection)
    sim.time = tstart
    return sim


def prepareSolverAD3D(D, tstart):
    num_particles = 1000
    dimension = 3
    L = np.array((1.0, 1.0, 1.0))
    boundaries = ReflectiveBoundaries(-L / 2, L / 2)
    advection = DriftCosine([1., 1.74, 0.0], [0.2, 1.0, 0.5])
    initial_positions = np.zeros((num_particles, dimension))  # 3D
    # Selection of dt
    dt = 0.01
    sim = BrownianDynamics(initial_positions, D, dt, boundaries, advection)
    sim.time = tstart
    return sim


def evolveAD(sim,
             initial_positions,
             total_time,
             dt_model,
             t_jump=0.0,
             dimension=3):
    num_particles = np.shape(initial_positions)[1]
    initial_positions = np.reshape(initial_positions, (-1, dimension))
    assert np.shape(initial_positions)[0] == num_particles
    sim.time += t_jump
    sim.positions = initial_positions
    dump_every = dt_model
    ndumps = int(total_time / dump_every)
    positions_evol = []
    for i in range(ndumps):
        sim.advance(int(dump_every / sim.dt))
        positions = sim.positions.copy()
        positions_evol.append(positions)

    positions_evol = np.array(positions_evol)
    positions_evol = np.reshape(positions_evol, (-1, num_particles, dimension))
    return positions_evol


# def evolveAD3DD(u0, total_time, dt_model, tstart, D=None):
#     num_particles = np.shape(u0)[1]
#     # D=0.2
#     # dt=1e-3
#     if D == 0.002:
#         dt = 0.01
#     elif D == 0.02:
#         dt = 0.01
#     elif D == 0.2:
#         dt = 0.001
#     elif D == 2.0:
#         dt = 1e-4
#     else:
#         raise ValueError("Invalid D.")
#     L = np.array((1.0, 1.0, 1.0))
#     dimension = 3
#     initial_positions = np.reshape(u0, (-1, dimension))
#     assert np.shape(initial_positions)[0] == num_particles
#     boundaries = ReflectiveBoundaries(-L / 2, L / 2)
#     advection = DriftCosine([1., 1.74, 0.0], [0.2, 1.0, 0.5])
#     sim = BrownianDynamics(initial_positions, D, dt, boundaries, advection)
#     sim.time = tstart
#     dump_every = dt_model
#     ndumps = int(total_time / dump_every)
#     positions_arr = []
#     for i in range(ndumps):
#         sim.advance(int(dump_every / dt))
#         positions = sim.positions.copy()
#         positions_arr.append(positions)
#     positions_arr = np.array(positions_arr)
#     u = np.reshape(positions_arr, (-1, num_particles, dimension))
#     # print(np.shape(u))
#     return u
