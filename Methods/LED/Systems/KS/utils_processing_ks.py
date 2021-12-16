#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
from ... import Utils as utils


def addResultsSystemKS(model, results, statistics):
    # assert("ux_uxx_density_target" not in results)
    # assert("ux_uxx_density_predicted" not in results)
    # assert("ux_uxx_l1_hist_error" not in results)
    # assert("ux_uxx_wasserstein_distance" not in results)
    # assert("ux_uxx_mesh" not in results)
    results.update({
        "ux_uxx_density_target":
        statistics["ux_uxx_density_target"],
        "ux_uxx_density_predicted":
        statistics["ux_uxx_density_predicted"],
        "ux_uxx_l1_hist_error_vec":
        statistics["ux_uxx_l1_hist_error_vec"],
        "ux_uxx_wasserstein_distance":
        statistics["ux_uxx_wasserstein_distance"],
        "ux_uxx_l1_hist_error":
        statistics["ux_uxx_l1_hist_error"],
        "ux_uxx_mesh":
        statistics["ux_uxx_mesh"],
    })
    results["fields_2_save_2_logfile"].append("ux_uxx_l1_hist_error")
    results["fields_2_save_2_logfile"].append("ux_uxx_wasserstein_distance")
    return results


def computeStateDistributionStatisticsSystemKS(state_dist_statistics,
                                               targets_all, predictions_all):
    n_ics, T, D = np.shape(targets_all)
    targets_ = np.reshape(targets_all, (n_ics * T, D))
    predictions_ = np.reshape(predictions_all, (n_ics * T, D))
    L = 22
    D = 64
    dx = L / (D - 1)
    targets_ux = np.diff(targets_, axis=1) / dx
    targets_uxx = np.diff(targets_ux, axis=1) / dx
    targets_ux = targets_ux[:, :-1]
    targets_uxx = np.reshape(targets_uxx, (-1))
    targets_ux = np.reshape(targets_ux, (-1))

    predictions_ux = np.diff(predictions_, axis=1) / dx
    predictions_uxx = np.diff(predictions_ux, axis=1) / dx
    predictions_ux = predictions_ux[:, :-1]
    predictions_uxx = np.reshape(predictions_uxx, (-1))
    predictions_ux = np.reshape(predictions_ux, (-1))

    num_samples = np.shape(targets_ux)[0]
    nbins = utils.getNumberOfBins(num_samples, rule="rice")
    # plt.hist2d(targets_ux, targets_uxx, (nbins,nbins), norm=LogNorm(), cmap=plt.get_cmap("Reds"), normed=True)
    # plt.colorbar()
    # plt.show()
    # plt.hist2d(predictions_ux, predictions_uxx, (nbins,nbins), norm=LogNorm(), cmap=plt.get_cmap("Reds"), normed=True)
    # plt.colorbar()
    # plt.show()
    data1 = np.concatenate((targets_ux[np.newaxis], targets_uxx[np.newaxis]),
                           axis=0).T
    data2 = np.concatenate(
        (predictions_ux[np.newaxis], predictions_uxx[np.newaxis]), axis=0).T
    ux_uxx_bounds = [[np.min(data1[:, 0]),
                      np.max(data1[:, 0])],
                     [np.min(data1[:, 1]),
                      np.max(data1[:, 1])]]
    ux_uxx_l1_hist_error, ux_uxx_l1_hist_error_vec, ux_uxx_density_target, ux_uxx_density_predicted, ux_uxx_bin_centers = utils.evaluateL1HistErrorVector(
        data1, data2, nbins, ux_uxx_bounds)
    ux_uxx_wasserstein_distance = utils.evaluateWassersteinDistance(
        data1, data2)
    # print("ux_uxx_l1_hist_error = {:}".format(ux_uxx_l1_hist_error))
    # print("ux_uxx_wasserstein_distance = {:}".format(ux_uxx_wasserstein_distance))
    ux_uxx_mesh = np.meshgrid(ux_uxx_bin_centers[0], ux_uxx_bin_centers[1])

    assert ("ux_uxx_density_target" not in state_dist_statistics)
    assert ("ux_uxx_density_predicted" not in state_dist_statistics)
    assert ("ux_uxx_l1_hist_error" not in state_dist_statistics)
    assert ("ux_uxx_l1_hist_error_vec" not in state_dist_statistics)
    assert ("ux_uxx_wasserstein_distance" not in state_dist_statistics)
    assert ("ux_uxx_mesh" not in state_dist_statistics)
    state_dist_statistics.update({
        "ux_uxx_density_target": ux_uxx_density_target,
        "ux_uxx_density_predicted": ux_uxx_density_predicted,
        "ux_uxx_l1_hist_error": ux_uxx_l1_hist_error,
        "ux_uxx_l1_hist_error_vec": ux_uxx_l1_hist_error_vec,
        "ux_uxx_wasserstein_distance": ux_uxx_wasserstein_distance,
        "ux_uxx_mesh": ux_uxx_mesh,
    })
    return state_dist_statistics
