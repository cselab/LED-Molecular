#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
# from  ... import Utils as utils


def calculate_density(pos):
    m = 18.015 / 6.02214076e23  #g
    Nics = pos.shape[0]
    Ntraj = pos.shape[1]
    N = pos.shape[2]
    Nd = float(N)  # number of particles
    rho = np.zeros(Ntraj)
    for ic in range(Nics):
        for t in range(Ntraj):
            H, edges = np.histogramdd(
                pos[ic, t, :, :], bins=(50, 50, 50),
                density=True)  # = bin_count/bin_volume / sample_count  (in A)
            # revert to density
            rhoi = H * N * m * 1.e24  # in units g/cm^3
            rho[t] += np.average(rhoi)  # array of rho values over time
    rho /= float(Nics)  # average over initial conditions
    return rho


def get_com(pos):
    # compute center of mass - for same masses = geometric center of positions
    # pos for instances (Ntraj,Natoms,dim)
    Natoms = pos.shape[1]
    comt = np.average(pos, axis=1)
    com_ext = np.tile(comt, (Natoms, 1, 1))
    return np.swapaxes(com_ext, 0, 1)


def get_msd_traj(input_positions):
    pos = input_positions.copy()
    Ntraj = pos.shape[0]
    N = pos.shape[1]
    MSD_itr = np.zeros_like(pos)  # for each instance, atom and dim
    for i in range(1, Ntraj):
        # first subtract CoM drift
        pos[i:, :, :] = pos[i:, :, :] - get_com(pos[i:, :, :])
        pos[:-i, :, :] = pos[:-i, :, :] - get_com(pos[:-i, :, :])
        MSD_itr[i, :, :] = np.average((pos[i:, :, :] - pos[:-i, :, :])**2,
                                      axis=0)
    MSD_tr = np.average(np.sum(MSD_itr, axis=2),
                        axis=1)  # sum over dim and average over atoms
    return MSD_tr


def calculate_diffusion_coefficient(pos):
    # calculate MSD[Ntraj]
    Nics = pos.shape[0]
    Ntraj = pos.shape[1]
    MSD_col = np.zeros(Ntraj)
    for ic in range(Nics):
        MSDi = get_msd_traj(pos[ic])
        MSD_col += MSDi
    MSD_col /= float(Nics)  # average over initial conditions
    time = np.arange(Ntraj)  # ps
    a = np.polyfit(time, MSD_col, deg=1)  # linear fit
    #line_fit = np.poly1d(a)
    D = a[0] / 6.0 * 100.  # in 10^-4 cm^2/s
    return D


def get_com_traj(pos):
    assert (len(np.shape(pos)) == 4)
    com_traj = np.average(pos, axis=2)
    return com_traj


def addResultsSystemCGW(model, results, statistics, testing_mode):
    if "autoencoder" in testing_mode:
        targets_all = results["input_sequence_all"]
        predictions_all = results["input_decoded_all"]
    else:
        targets_all = results["targets_all"]
        predictions_all = results["predictions_all"]

    dt = results["dt"]

    # targets_all shape [N_ICS, T, N_PARTICLES, 3]
    # targets_all shape [1, T, N_PARTICLES, 3]

    density_targ = calculate_density(targets_all)
    density_pred = calculate_density(predictions_all)
    results["density_traj_target"] = density_targ
    results["density_traj_predict"] = density_pred

    results["diffusion_coeff_target"] = calculate_diffusion_coefficient(
        targets_all)
    results["diffusion_coeff_predict"] = calculate_diffusion_coefficient(
        predictions_all)

    # print(np.shape(predictions_all))
    # print(np.shape(targets_all))

    # print(np.min(targets_all[:,:,:,0]))
    # print(np.max(targets_all[:,:,:,0]))

    com_target = get_com_traj(targets_all)
    com_pred = get_com_traj(predictions_all)

    results["com_target"] = com_target
    results["com_predict"] = com_pred
    # (Nics, Ntraj, 3)
    # calculate absolute error in trajectory and initial conditions
    error_com_alldir = np.average(np.average(np.abs(com_target - com_pred),
                                             axis=1),
                                  axis=0)
    results["abs_error_com_target_predict"] = error_com_alldir  # 3 directions

    # Do you want to save it in a log-file ? Like a number that determines the effiency ? Probably not..
    # results["fields_2_save_2_logfile"].append("erminas_result_single_number")
    return results
