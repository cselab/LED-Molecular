#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
import socket
import os
# Plotting parameters
import matplotlib

from scipy.interpolate import interpn
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.neighbors import NearestNeighbors

from ... import Utils as utils
from . import utils_processing_alanine

import os.path

hostname = socket.gethostname()
print("PLOTTING HOSTNAME: {:}".format(hostname))
CLUSTER = True if ((hostname[:2] == 'eu') or (hostname[:5] == 'daint') or
                   (hostname[:3] == 'nid') or
                   (hostname[:14] == 'barrycontainer')) else False

if (hostname[:2] == 'eu'):
    CLUSTER_NAME = "euler"
elif (hostname[:5] == 'daint'):
    CLUSTER_NAME = "daint"
elif (hostname[:3] == 'nid'):
    CLUSTER_NAME = "daint"
elif (hostname[:14] == 'barrycontainer'):
    CLUSTER_NAME = "barry"
else:
    CLUSTER_NAME = "local"

print("CLUSTER={:}, CLUSTER_NAME={:}".format(CLUSTER, CLUSTER_NAME))

if CLUSTER: matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
import io

print("-V- Matplotlib Version = {:}".format(matplotlib.__version__))

from matplotlib import colors
import six
color_dict = dict(six.iteritems(colors.cnames))

color_labels = [
    'tab:red', 'tab:blue', 'tab:green', 'tab:brown', 'tab:orange', 'tab:cyan',
    'tab:olive', 'tab:pink', 'tab:gray', 'tab:purple'
]

linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
linemarkers = ["s", "d", "o", ">", "*", "x", "<", ">"]
linemarkerswidth = [3, 2, 2, 2, 4, 2, 2, 2]

FONTSIZE = 18
font = {'size': FONTSIZE, 'family': 'Times New Roman'}
matplotlib.rc('xtick', labelsize=FONTSIZE)
matplotlib.rc('ytick', labelsize=FONTSIZE)
matplotlib.rc('font', **font)
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'

if CLUSTER_NAME in ["local", "barry"]:
    # Plotting parameters
    rc('text', usetex=True)
    plt.rcParams["text.usetex"] = True
    plt.rcParams['xtick.major.pad'] = '10'
    plt.rcParams['ytick.major.pad'] = '10'

FIGTYPE = "pdf"
# FIGTYPE="pdf"


def getClusterColormapsAlanine():
    cluster_colormaps = {
        0: plt.get_cmap("Blues"),
        1: plt.get_cmap("Greens"),
        2: plt.get_cmap("Oranges"),
        3: plt.get_cmap("Purples"),
        4: plt.get_cmap("Reds"),
        None: plt.get_cmap("Reds"),
    }
    return cluster_colormaps


def addRamachandranPlotText(ax, radius=30, fontsize=None):
    # ax.set_xlabel(r"Dihedral angle $\phi$")
    # ax.set_ylabel(r"Dihedral angle $\psi$")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\psi$")
    cluster_labels = utils_processing_alanine.getRamaClusterLabelsAlanine()
    cluster_centers = utils_processing_alanine.getRamaClusterCentersAlanine()
    for cluster_id, label in cluster_labels.items():
        centers = cluster_centers[cluster_id]
        label = "$" + label + "$"
        for center in centers:
            # ax.text(center[0],center[1],r"{:}".format(label), ha='center', va='center')
            ax.text(center[0], center[1], r"{:}".format(label), ha='center', fontsize=fontsize)
            # circle = plt.Circle(center+np.array([5,5]), radius, color='k', fill=False, linewidth=2, zorder=2, linestyle=':')
            circle = plt.Circle(center + np.array([5, 5]),
                                radius,
                                color='k',
                                fill=False,
                                linewidth=2,
                                zorder=2,
                                linestyle='--')
            ax.add_artist(circle)

    ax.set_xlim([-180, 180])
    ax.set_ylim([-180, 180])
    ax.set_aspect('equal')

    # ax.text(-155,90,"C5")
    # ax.text(-70,90,"C7eq")
    # ax.text(145,90,"C5")
    # ax.text(-155,-150,"C5")
    # ax.text(-70,-150,"C7eq")
    # ax.text(145,-150,"C5")
    # ax.text(-170,-90,r"$\alpha_R^{\prime \prime}$")
    # ax.text(140,-90,r"$\alpha_R^{\prime \prime}$")
    # ax.text(-70,-90,r"$\alpha_R$")
    # ax.text(70,0,r"$\alpha_L$")
    # ax.set_xlim([-180, 180])
    # ax.set_ylim([-180, 180])
    # x = [-180, 13]
    # y = [74, 74]
    # ax.plot(x,y,color='black')
    # x = [128, 180]
    # ax.plot(x,y,color='black')
    # x = [13, 13]
    # y = [-180, 180]
    # ax.plot(x,y,color='black')
    # x = [128, 128]
    # ax.plot(x,y,color='black')
    # x = [-180, 13]
    # y = [-125, -125]
    # ax.plot(x,y,color='black')
    # x = [128, 180]
    # ax.plot(x,y,color='black')
    # x = [-134, -134]
    # y = [-125, 74]
    # ax.plot(x,y,color='black')
    # x = [-110, -110]
    # y = [-180, -125]
    # ax.plot(x,y,color='black')
    # y = [74, 180]
    # ax.plot(x,y,color='black')
    # x = [-180,-180,180,180,-180]
    # y = [-180,180,180,-180,-180]
    # ax.plot(x,y,color='black')
    return ax


def plotTransitionTimesAlanine(model, results, set_name, testing_mode):
    cluster_labels = utils_processing_alanine.getRamaClusterLabelsAlanine()

    transitions_to_print = [
        tuple((0, 1)),
        tuple((0, 2)),
        tuple((1, 0)),
        tuple((1, 2)),
        tuple((2, 1)),
        tuple((2, 0)),
        # tuple((3,0)),
        # tuple((3,1)),
        # tuple((3,2)),
    ]

    times_target = results["cluster_transition_times_target"]
    times_pred = results["cluster_transition_times_pred"]
    times_errors = results["cluster_transition_times_errors"]

    file_path = model.getFigureDir(
    ) + "/{:}_{:}_rama_clusters_mean_passage_times_LATEX.{:}".format(
        testing_mode, set_name, "txt")

    writeLatentTransitionTimesToLatexTableFile(model,
                                               file_path,
                                               cluster_labels,
                                               transitions_to_print,
                                               times_target,
                                               times_pred,
                                               times_error=times_errors)

    file_path = model.getFigureDir(
    ) + "/{:}_{:}_rama_clusters_mean_passage_times.{:}".format(
        testing_mode, set_name, "txt")

    writeLatentTransitionTimesToFileAlanine(model, file_path, cluster_labels,
                                            times_target, times_pred,
                                            times_errors)

    times_target = results["cluster_transition_times_target_msm_Dt_small"]
    times_pred = results["cluster_transition_times_pred_msm_Dt_small"]
    times_errors = results["cluster_transition_times_errors_msm_Dt_small"]

    file_path = model.getFigureDir(
    ) + "/{:}_{:}_rama_clusters_mean_passage_times_msm_dt_small_LATEX.{:}".format(
        testing_mode, set_name, "txt")

    writeLatentTransitionTimesToLatexTableFile(model,
                                               file_path,
                                               cluster_labels,
                                               transitions_to_print,
                                               times_target,
                                               times_pred,
                                               times_error=times_errors)

    file_path = model.getFigureDir(
    ) + "/{:}_{:}_rama_clusters_mean_passage_times_msm_dt_small.{:}".format(
        testing_mode, set_name, "txt")

    writeLatentTransitionTimesToFileAlanine(model, file_path, cluster_labels,
                                            times_target, times_pred,
                                            times_errors)

    return results


def writeLatentTransitionTimesToFileAlanine(
    model,
    file_path,
    cluster_labels,
    times_target,
    times_pred,
    times_errors,
    print_target=True,
    print_prediction=True,
    print_error=True,
):
    nstates = len(cluster_labels)

    already_written = False
    write_mode = "w"
    if print_target:
        if already_written: write_mode = "a+"
        lines_target = utils.getMeanTransitionTimesFileLines(
            cluster_labels, times_target, nstates)
        with io.open(file_path, write_mode) as f:
            f.write("TARGET")
            f.write("\n")
            for line in lines_target:
                f.write(line)
                f.write("\n")
            f.write("\n")
        already_written = True

    if print_prediction:
        if already_written: write_mode = "a+"
        lines_pred = utils.getMeanTransitionTimesFileLines(
            cluster_labels, times_pred, nstates)
        with io.open(file_path, write_mode) as f:
            f.write("PREDICTION")
            f.write("\n")
            for line in lines_pred:
                f.write(line)
                f.write("\n")
            f.write("\n")
        already_written = True

    if print_error:
        if already_written: write_mode = "a+"
        n_states_errors = len(utils_processing_alanine.
                              getRamaClustersOverWhichToComputeMFPTError())
        lines_errors = utils.getMeanTransitionTimesFileLines(
            cluster_labels, times_errors, n_states_errors)
        with io.open(file_path, write_mode) as f:
            f.write("RELATIVE ERROR")
            f.write("\n")
            for line in lines_errors:
                f.write(line)
                f.write("\n")
            f.write("\n")
        already_written = True


def writeLatentTransitionTimesToLatexTableFile(
    model,
    file_path,
    cluster_labels,
    transitions_to_print,
    times_target,
    times_pred,
    times_error=None,
):
    nstates = len(cluster_labels)
    lines = []
    errors_relative = []
    for i in range(nstates):
        for j in range(nstates):
            if i != j:
                if tuple((i, j)) in transitions_to_print:
                    transition_str_i = "{:}".format(cluster_labels[i])
                    transition_str_j = "{:}".format(cluster_labels[j])
                    transition_str = "$T_{" + transition_str_i + " \\to " + transition_str_j + "}$"
                    transition_str += "&"
                    value = times_target[tuple((i, j))]
                    value_str = "${:.3f}$".format(value)
                    transition_str += str(value_str)
                    transition_str += "&"
                    value = times_pred[tuple((i, j))]
                    value_str = "${:.3f}$".format(value)
                    transition_str += str(value_str)
                    transition_str += "&"

                    if times_error is not None:
                        value = times_error[tuple((i, j))]
                    else:
                        raise ValueError()
                        time_target = cluster_transition_times_target[tuple(
                            (i, j))]
                        if mean_time_target > 0.0:
                            error = np.abs(mean_time_target - mean_time_pred)
                            error = error / mean_time_target
                        else:
                            error = 0.0
                        value = error

                    errors_relative.append(value)
                    value_str = "${:2.0f}$".format(value * 100.0)
                    transition_str += str(value_str)
                    transition_str += " " * 5
                    # print(transition_str)
                    lines.append(transition_str)
    error_relative = np.mean(np.array(errors_relative))
    value = error_relative
    value_str = "${:.2f}$".format(value * 100.0)
    line = "Mean error = " + value_str
    # print(line)
    lines.append(line)

    write_mode = "w"
    with io.open(file_path, write_mode) as f:
        for line in lines:
            f.write(line)
            f.write("\n")
        f.write("\n")
    return 0


def plotStateDistributionsSystemAlanine(model, results, set_name,
                                        testing_mode):
    makeRamaPlots(model, set_name, testing_mode, results)
    # plotClusteredDynamics(model, results, set_name, testing_mode)
    return 0


def plotLatentMetaStableStatesAndLatentTransitionTimesAlanine(
        model, results, set_name, testing_mode):

    if "autoencoder" in testing_mode:
        data_all = results["input_sequence_all"]
        latent_states_all = results["latent_states_all"]
        clustered_trajectories_target = results[
            "clustered_trajectories_target"]

    elif "iterative" in testing_mode:
        data_all = results["predictions_all"][:, :-1]
        latent_states_all = results["latent_states_all"][:, 1:]
        clustered_trajectories_target = results[
            "clustered_trajectories_pred"][:, 1:]

    elif "teacher_forcing" in testing_mode:
        data_all = results["targets_all"][:, :-1]
        latent_states_all = results["latent_states_all"][:, 1:]
        clustered_trajectories_target = results[
            "clustered_trajectories_target"][:, :-1]
    else:
        raise ValueError(
            "testing_mode = {:} not recognized.".format(testing_mode))

    latent_state_regions = results["free_energy_latent_state_regions"]
    latent_cluster_labels = results["free_energy_latent_cluster_labels"]
    latent_state_freenergy = results["latent_state_freenergy"]
    latent_state_free_energy_grid = results["latent_state_free_energy_grid"]

    rama_cluster_labels = utils_processing_alanine.getRamaClusterLabelsAlanine(
    )

    colormaps = [
        plt.get_cmap("Reds"),
        plt.get_cmap("Reds"),
        plt.get_cmap("Reds"),
        plt.get_cmap("Reds"),
        plt.get_cmap("Reds"),
        plt.get_cmap("Reds"),
    ]

    latent_states_all = np.reshape(latent_states_all,
                                   (-1, np.shape(latent_states_all)[2]))
    data_all = np.reshape(data_all, (-1, np.shape(data_all)[2]))
    clustered_trajectories_target = np.reshape(clustered_trajectories_target,
                                               (-1))

    latent_cluster_labels_map2Rama = []
    latent_cluster_labels_map2RamaIdx = []

    mapLatentRegion2RamaState = {}
    mapRamaState2LatentRegion = {}

    for region_iter in range(len(latent_state_regions)):
        if region_iter > len(colormaps) - 1:
            break
        latent_region = latent_state_regions[region_iter]
        idx_region = np.bitwise_and(latent_states_all > latent_region[0],
                                    latent_states_all < latent_region[1])
        idx_region = idx_region[:, 0]
        idx_region = np.where(idx_region)[0]
        # print(idx_region)
        # print(ark)
        # idx_region = idx_region[::20]
        idx_region = list(idx_region)
        if 0 in idx_region: idx_region.remove(0)

        # print(idx_region[:5])
        if len(idx_region) > 0:
            # idx_region = np.array(idx_region)
            data_all_region = data_all[idx_region]
            clustered_trajectories_target_region = clustered_trajectories_target[
                idx_region]

            cluster_rama_idx = mostCommonElementInList(
                list(clustered_trajectories_target_region),
                exempt=len(rama_cluster_labels))
            rama_cluster_label = rama_cluster_labels[cluster_rama_idx]
            latent_cluster_labels_map2Rama.append(rama_cluster_label)
            latent_cluster_labels_map2RamaIdx.append(cluster_rama_idx)
            # print(cluster_rama_idx)

            print("Region {:} mapped to state {:}.".format(
                latent_region, rama_cluster_label))
            mapLatentRegion2RamaState.update({region_iter: cluster_rama_idx})
            mapRamaState2LatentRegion.update({cluster_rama_idx: region_iter})

            # for log_norm in [True, False]:
            for log_norm in [False]:
                # for with_colorbar in [True, False]:
                for with_colorbar in [True]:
                    log_norm_str = "Normed" if log_norm else ""
                    colorbar_str = "Colorbar" if with_colorbar else ""
                    fig, ax = plt.subplots(figsize=(4.25 * 1, 4.25 * 5/6 * 1))
                    textfontsize=12
                    phi = data_all_region[:, 21] * 180 / np.pi
                    psi = data_all_region[:, 20] * 180 / np.pi
                    if len(phi) > 0:
                        ax, mp = createRamachandranPlot(
                            phi,
                            psi,
                            ax,
                            cmap=colormaps[region_iter],
                            with_colorbar=with_colorbar,
                            log_norm=log_norm,
                            fontsize=textfontsize,
                            )
                    ax.set_xlabel(r"$\phi$")
                    ax.set_ylabel(r"$\psi$")
                    fig.tight_layout()
                    plt.savefig(
                        model.getFigureDir() +
                        "/{:}_{:}_latent_meta_stable_states_region_{:}_{:}{:}.{:}"
                        .format(testing_mode, set_name, latent_region,
                                log_norm_str, colorbar_str, FIGTYPE),
                        bbox_inches="tight",
                        dpi=100)
                    plt.close()

    # times_target = results["cluster_transition_times_target"]
    # latent_cluster_mean_passage_times = results[
    #     "latent_cluster_mean_passage_times"]

    # latent_cluster_mean_passage_times_mapped_to_rama_states = {}
    # for rama_key in times_target:
    #     rama_i, rama_j = rama_key
    #     if rama_i in mapRamaState2LatentRegion and rama_j in mapRamaState2LatentRegion:
    #         i = mapRamaState2LatentRegion[rama_i]
    #         j = mapRamaState2LatentRegion[rama_j]
    #         key = tuple((i, j))
    #         if key in latent_cluster_mean_passage_times:
    #             latent_cluster_mean_passage_times_mapped_to_rama_states[
    #                 rama_key] = latent_cluster_mean_passage_times[key]
    #         else:
    #             latent_cluster_mean_passage_times_mapped_to_rama_states[
    #                 rama_key] = 0.0
    #     else:
    #         latent_cluster_mean_passage_times_mapped_to_rama_states[
    #             rama_key] = 0.0

    # cluster_labels = utils_processing_alanine.getRamaClusterLabelsAlanine()

    # transitions_to_print = [
    #     tuple((0, 1)),
    #     tuple((0, 2)),
    #     tuple((1, 0)),
    #     tuple((1, 2)),
    #     tuple((2, 1)),
    #     tuple((2, 0)),
    #     # tuple((3,0)),
    #     # tuple((3,1)),
    #     # tuple((3,2)),
    # ]

    # times_pred = latent_cluster_mean_passage_times_mapped_to_rama_states

    # cluster_transitions_considered_in_the_error = utils_processing_alanine.getRamaClustersOverWhichToComputeMFPTErrorKeys(
    # )
    # times_errors, _ = utils.computeErrorOnTimes(
    #     times_target, times_pred, cluster_transitions_considered_in_the_error)

    # file_path = model.getFigureDir(
    # ) + "/{:}_{:}_latent_meta_stable_states_mean_passage_times_maped_to_rama_LATEX.{:}".format(
    #     testing_mode, set_name, "txt")

    # writeLatentTransitionTimesToLatexTableFile(model,
    #                                            file_path,
    #                                            cluster_labels,
    #                                            transitions_to_print,
    #                                            times_target,
    #                                            times_pred,
    #                                            times_error=times_errors)

    # file_path = model.getFigureDir(
    # ) + "/{:}_{:}_latent_meta_stable_states_mean_passage_times_maped_to_rama.{:}".format(
    #     testing_mode, set_name, "txt")

    # writeLatentTransitionTimesToFileAlanine(model, file_path, cluster_labels,
    #                                         times_target, times_pred,
    #                                         times_errors)

    return 0


def mostCommonElementInList(list_, exempt=None):
    from collections import Counter
    temp = Counter(list_)
    # Removing the cluster_idx=len(clusters) corresponding to None
    if exempt in temp: del temp[exempt]
    # Getting the state with the maximum occurence
    key = max(temp, key=temp.get)
    # print(temp)
    return key


def truncate(data, n_splits):
    n_traj = np.shape(data)[0]
    to_remove = n_traj % n_splits
    data = data[:-to_remove or None]
    return data

def plotLatentDynamicsComparisonSystemAlanine(model, set_name):
    print("# plotLatentDynamicsComparisonSystemAlanine() #")
    # tf_targets_all, tf_latent_states_all, latent_state_regions = utils.plotLatentDynamicsComparisonSystemCommon(
    #     model, set_name)
    assert (model.params["latent_state_dim"] == 1)

    testing_mode = "teacher_forcing_forecasting"
    data_path = model.getResultsDir() + "/results_{:}_{:}".format(testing_mode, set_name)
    results = utils.loadData(data_path, model.save_format)
    # Results of teacher forcing the MD data
    tf_latent_state_free_energy_grid = results["latent_state_free_energy_grid"]
    tf_latent_states_flatten_range = results["latent_states_flatten_range"]
    tf_latent_state_freenergy_ = results["latent_state_freenergy"]
    tf_free_energy_latent_state_regions = results[
        "free_energy_latent_state_regions"]
    tf_latent_states_all = results["latent_states_all"]
    tf_covariance_factor = results["covariance_factor"]
    tf_targets_all = results["targets_all"]

    testing_mode = "iterative_latent_forecasting"
    data_path = model.getResultsDir() + "/results_{:}_{:}".format(
        testing_mode, set_name)
    results = utils.loadData(data_path, model.save_format)
    # Results of iterative latent forecasting
    lf_latent_state_free_energy_grid = results["latent_state_free_energy_grid"]
    lf_latent_states_flatten_range = results["latent_states_flatten_range"]
    lf_latent_state_freenergy_ = results["latent_state_freenergy"]
    lf_free_energy_latent_state_regions = results[
        "free_energy_latent_state_regions"]
    lf_latent_states_all = results["latent_states_all"]
    lf_covariance_factor = results["covariance_factor"]


    testing_mode = "iterative_latent_forecasting"

    for random_seed in [10, 20, 30, 40]:
        # Loading the results
        data_path = model.getResultsDir(
        ) + "-RS_{:}".format(random_seed) + "/results_{:}_{:}".format(testing_mode, "test")
        results_lf = utils.loadData(data_path, model.save_format)
        latent_states_all = results_lf["latent_states_all"]
        del results_lf
        lf_latent_states_all = np.concatenate((lf_latent_states_all, latent_states_all), axis=0)

    testing_mode = "teacher_forcing_forecasting"
    data_path = model.getResultsDir() + "/results_{:}_{:}".format(
        testing_mode, "train")
    results = utils.loadData(data_path, model.save_format)
    # Results of iterative latent forecasting
    train_latent_state_free_energy_grid = results["latent_state_free_energy_grid"]
    train_latent_states_flatten_range = results["latent_states_flatten_range"]
    train_latent_state_freenergy_ = results["latent_state_freenergy"]
    train_free_energy_latent_state_regions = results[
        "free_energy_latent_state_regions"]
    train_latent_states_all = results["latent_states_all"]
    train_covariance_factor = results["covariance_factor"]

    testing_mode = "teacher_forcing_forecasting"
    data_path = model.getResultsDir() + "/results_{:}_{:}".format(
        testing_mode, "val")
    results = utils.loadData(data_path, model.save_format)
    # Results of iterative latent forecasting
    val_latent_state_free_energy_grid = results["latent_state_free_energy_grid"]
    val_latent_states_flatten_range = results["latent_states_flatten_range"]
    val_latent_state_freenergy_ = results["latent_state_freenergy"]
    val_free_energy_latent_state_regions = results[
        "free_energy_latent_state_regions"]
    val_latent_states_all = results["latent_states_all"]
    val_covariance_factor = results["covariance_factor"]
    train_latent_states_all = np.concatenate((train_latent_states_all, val_latent_states_all), axis=0)


    max_ = np.max([np.max(tf_latent_states_all), np.max(lf_latent_states_all), np.max(train_latent_states_all)])
    min_ = np.min([np.min(tf_latent_states_all), np.min(lf_latent_states_all), np.min(train_latent_states_all)])
    latent_states_flatten_range = max_ - min_
    max_ += 0.05 * latent_states_flatten_range
    min_ -= 0.05 * latent_states_flatten_range
    latent_state_free_energy_grid = np.linspace(min_, max_, 400)


    margin = 0.2 * (np.max(lf_latent_state_freenergy_) -
                    np.min(lf_latent_state_freenergy_))
    freenergy_min_global_margin = np.min(lf_latent_state_freenergy_) - margin

    # lf_latent_states_all = lf_latent_states_all[:1104]
    print("Shapes of datasets for free energy comparison:")
    print("np.shape(train_latent_states_all) = {:}".format(np.shape(train_latent_states_all)))
    print("np.shape(tf_latent_states_all) = {:}".format(np.shape(tf_latent_states_all)))
    print("np.shape(lf_latent_states_all) = {:}".format(np.shape(lf_latent_states_all)))
    # 1104
    # np.shape(train_latent_states_all) = (192, 3600, 1)
    # np.shape(tf_latent_states_all) = (248, 3600, 1)
    # np.shape(lf_latent_states_all) = (1232, 3600, 1)

    # print(ark)
    n_splits_train = 96
    n_splits_lf = 96
    n_splits_tf = 1

    # n_splits_train = 1
    # n_splits_lf = 1
    # n_splits_tf = 1
    # train_latent_states_all = train_latent_states_all[:10]
    # tf_latent_states_all = tf_latent_states_all[:10]
    # lf_latent_states_all = lf_latent_states_all[:10]

    train_latent_states_all = truncate(train_latent_states_all, n_splits_train)
    lf_latent_states_all = truncate(lf_latent_states_all, n_splits_lf)
    tf_latent_states_all = truncate(tf_latent_states_all, n_splits_tf)

    train_latent_state_freenergy_bootstrap = utils.calculateLatentFreeEnergyWithUncertainty(
        train_latent_states_all, n_splits_train, tf_covariance_factor,
        latent_state_free_energy_grid)
    train_latent_state_freenergy_mean = np.mean(train_latent_state_freenergy_bootstrap, axis=0)
    train_latent_state_freenergy_std = np.std(train_latent_state_freenergy_bootstrap, axis=0)
    train_latent_state_freenergy_std = train_latent_state_freenergy_std / np.sqrt(n_splits_train)

    lf_latent_state_freenergy_bootstrap = utils.calculateLatentFreeEnergyWithUncertainty(
        lf_latent_states_all, n_splits_lf, tf_covariance_factor,
        latent_state_free_energy_grid)
    lf_latent_state_freenergy_mean = np.mean(lf_latent_state_freenergy_bootstrap, axis=0)
    lf_latent_state_freenergy_std = np.std(lf_latent_state_freenergy_bootstrap, axis=0)
    lf_latent_state_freenergy_std = lf_latent_state_freenergy_std / np.sqrt(n_splits_lf)

    tf_latent_state_freenergy_bootstrap = utils.calculateLatentFreeEnergyWithUncertainty(
        tf_latent_states_all, n_splits_tf, tf_covariance_factor,
        latent_state_free_energy_grid)
    tf_latent_state_freenergy_mean = np.mean(tf_latent_state_freenergy_bootstrap, axis=0)
    # tf_latent_state_freenergy_std = np.std(tf_latent_state_freenergy_bootstrap, axis=0)
    # tf_latent_state_freenergy_bootstrap = tf_latent_state_freenergy_bootstrap / np.sqrt(n_splits_tf)


    fig, ax = plt.subplots(figsize=(4, 3))

    plt.plot(latent_state_free_energy_grid,
             train_latent_state_freenergy_mean,
             "y",
             label="MD train",
             linewidth=2)
    plt.fill_between(latent_state_free_energy_grid, train_latent_state_freenergy_mean-train_latent_state_freenergy_std, train_latent_state_freenergy_mean+train_latent_state_freenergy_std, alpha=0.4, color="y")

    plt.plot(latent_state_free_energy_grid,
             tf_latent_state_freenergy_mean,
             "g",
             label="MD all",
             linewidth=2)
    # plt.fill_between(latent_state_free_energy_grid, tf_latent_state_freenergy_mean-tf_latent_state_freenergy_std, tf_latent_state_freenergy_mean+tf_latent_state_freenergy_std, alpha=0.4, color="g")

    plt.plot(latent_state_free_energy_grid,
             lf_latent_state_freenergy_mean,
             "b",
             label="LED Iterative",
             linewidth=2)
    plt.fill_between(latent_state_free_energy_grid, lf_latent_state_freenergy_mean-lf_latent_state_freenergy_std, lf_latent_state_freenergy_mean+lf_latent_state_freenergy_std, alpha=0.4, color="b")


    for region in tf_free_energy_latent_state_regions:
        size = region[1] - region[0]
        idx_region = np.bitwise_and(
            tf_latent_state_free_energy_grid > region[0],
            tf_latent_state_free_energy_grid < region[1])
        idx_region = np.where(idx_region)[0]
        if len(idx_region) > 0:
            # freenergy_min = np.min(tf_latent_state_freenergy_[idx_region])

            idx_region = np.bitwise_and(
                lf_latent_state_free_energy_grid > region[0],
                lf_latent_state_free_energy_grid < region[1])
            idx_region = np.where(idx_region)[0]
            if len(idx_region) > 0:
                # freenergy_min2 = np.min(lf_latent_state_freenergy_[idx_region])
                # freenergy_min = np.min([freenergy_min, freenergy_min2])
                # margin = 0.05 * (np.max(tf_latent_state_freenergy_)-np.min(tf_latent_state_freenergy_))

                plt.errorbar((region[0] + region[1]) / 2,
                             freenergy_min_global_margin,
                             xerr=size,
                             color="k",
                             capsize=8,
                             capthick=4,
                             linewidth=4,
                             clip_on=False)
                plt.gca().set_ylim(bottom=freenergy_min_global_margin)

    plt.gca().set_ylim(top=np.max(tf_latent_state_freenergy_mean))
    plt.ylabel(r'$F/\kappa_B T $')
    plt.xlabel(r'$z$')
    fig.tight_layout()
    # plt.xlim([-4.5,4.5])
    plt.savefig(model.getFigureDir() +
                "/Comparison_latent_dynamics_calibrated_bootstrap_{:}.{:}".format(
                    set_name, FIGTYPE),
                bbox_inches="tight",
                dpi=300)
    plt.legend(loc="upper center",
               bbox_to_anchor=(0.5, 1.625),
               borderaxespad=0.,
               frameon=False)
    plt.savefig(model.getFigureDir() +
                "/Comparison_latent_dynamics_calibrated_bootstrap_{:}_legend.{:}".format(
                    set_name, FIGTYPE),
                bbox_inches="tight",
                dpi=300)
    plt.close()


    # Calculating relevant range
    max_ = np.percentile(tf_latent_states_all, 99)
    min_ = np.percentile(tf_latent_states_all, 1)
    rel_index = np.where(np.logical_and(latent_state_free_energy_grid>=min_, latent_state_free_energy_grid<=max_))[0]

    tf_latent_state_freenergy = tf_latent_state_freenergy_mean[rel_index]
    lf_latent_state_freenergy = lf_latent_state_freenergy_mean[rel_index]
    train_latent_state_freenergy = train_latent_state_freenergy_mean[rel_index]
    latent_state_free_energy_grid = latent_state_free_energy_grid[rel_index]

    # print(rel_index)



    error = np.abs(tf_latent_state_freenergy - lf_latent_state_freenergy)
    error = error * error
    total_error = np.trapz(error, x=latent_state_free_energy_grid)
    total_error = np.sqrt(total_error)
    print("(LED) FREE ENERGY MEAN SQUARE ROOT ERROR {:}".format(total_error))

    error_train = np.abs(tf_latent_state_freenergy - train_latent_state_freenergy)
    error_train = error_train * error_train
    total_error_train = np.trapz(error_train, x=latent_state_free_energy_grid)
    total_error_train = np.sqrt(total_error_train)

    print("(train) FREE ENERGY MEAN SQUARE ROOT ERROR {:}".format(total_error_train))

    file_path = model.getFigureDir(
    ) + "/{:}_latent_free_energy_RMSE_ERROR.{:}".format(set_name, "txt")

    with io.open(file_path, "w") as f:
        f.write("(LED) FREE ENERGY MEAN SQUARE ROOT ERROR {:}\n".format(total_error))
        f.write("(train) FREE ENERGY MEAN SQUARE ROOT ERROR {:}\n".format(total_error_train))
        f.write("\n")




    # print(ark)




    # latent_state_regions = tf_free_energy_latent_state_regions

    # # tf_covariance_factor = 5.0
    # # lf_covariance_factor = 5.0

    # # print(tf_covariance_factor)
    # # print(lf_covariance_factor)

    # covariance_factor = tf_covariance_factor
    # # max_ = np.max([np.max(tf_latent_states_all), np.max(lf_latent_states_all)])
    # # min_ = np.min([np.min(tf_latent_states_all), np.min(lf_latent_states_all)])

    # max_ = np.percentile(tf_latent_states_all, 99)
    # min_ = np.percentile(tf_latent_states_all, 1)
    # # print(max_, min_)

    # latent_states_flatten_range = max_ - min_
    # latent_state_free_energy_grid = np.linspace(min_, max_, 100)

    # tf_latent_states_flatten = tf_latent_states_all.flatten()
    # tf_latent_state_freenergy = utils.calculateLatentFreeEnergy(
    #     tf_latent_states_flatten, covariance_factor,
    #     latent_state_free_energy_grid)

    # lf_latent_states_flatten = lf_latent_states_all.flatten()
    # lf_latent_state_freenergy = utils.calculateLatentFreeEnergy(
    #     lf_latent_states_flatten, covariance_factor,
    #     latent_state_free_energy_grid)

    # error = np.abs(tf_latent_state_freenergy - lf_latent_state_freenergy)
    # error = error * error
    # total_error = np.trapz(error, x=latent_state_free_energy_grid)
    # total_error = np.sqrt(total_error)
    # print("(LED) FREE ENERGY MEAN SQUARE ROOT ERROR {:}".format(total_error))

    # file_path = model.getFigureDir(
    # ) + "/{:}_latent_free_energy_RMSE_ERROR.{:}".format(set_name, "txt")

    # with io.open(file_path, "w") as f:
    #     f.write("FREE ENERGY MEAN SQUARE ROOT ERROR {:}".format(total_error))
    #     f.write("\n")

    # train_latent_states_flatten = train_latent_states_all.flatten()
    # train_latent_state_freenergy = utils.calculateLatentFreeEnergy(
    #     train_latent_states_flatten, covariance_factor,
    #     latent_state_free_energy_grid)
    # error = np.abs(tf_latent_state_freenergy - train_latent_state_freenergy)
    # error = error * error
    # total_error = np.trapz(error, x=latent_state_free_energy_grid)
    # total_error = np.sqrt(total_error)
    # print("(train) FREE ENERGY MEAN SQUARE ROOT ERROR {:}".format(total_error))


    # colormaps = [
    #     plt.get_cmap("Reds"),
    #     plt.get_cmap("Reds"),
    #     plt.get_cmap("Reds"),
    #     plt.get_cmap("Reds"),
    #     plt.get_cmap("Reds"),
    #     plt.get_cmap("Reds"),
    # ]

    # latent_states = np.reshape(tf_latent_states_all,
    #                            (-1, np.shape(tf_latent_states_all)[2]))
    # targets_all = np.reshape(tf_targets_all, (-1, np.shape(tf_targets_all)[2]))
    # for region_iter in range(len(latent_state_regions)):
    #     if region_iter > len(colormaps) - 1:
    #         break
    #     latent_region = latent_state_regions[region_iter]
    #     idx_region = np.bitwise_and(latent_states > latent_region[0],
    #                                 latent_states < latent_region[1])
    #     idx_region = idx_region[:, 0]
    #     idx_region = np.where(idx_region)[0]
    #     # print(idx_region)
    #     # print(ark)
    #     # idx_region = idx_region[::20]
    #     idx_region = list(idx_region)
    #     if 0 in idx_region: idx_region.remove(0)
    #     idx_region = np.array(idx_region) - 1
    #     # print(idx_region[:5])
    #     if len(idx_region) > 0:
    #         # idx_region = np.array(idx_region)
    #         targets_all_region = targets_all[idx_region]
    #         # for log_norm in [True, False]:
    #         for log_norm in [False]:
    #             # for with_colorbar in [True, False]:
    #             for with_colorbar in [True]:
    #                 log_norm_str = "Normed" if log_norm else ""
    #                 colorbar_str = "Colorbar" if with_colorbar else ""

    #                 fig, ax = plt.subplots()
    #                 phi = targets_all_region[:, 21] * 180 / np.pi
    #                 psi = targets_all_region[:, 20] * 180 / np.pi
    #                 if len(phi) > 0:
    #                     ax, mp = createRamachandranPlot(
    #                         phi,
    #                         psi,
    #                         ax,
    #                         cmap=colormaps[region_iter],
    #                         with_colorbar=with_colorbar,
    #                         log_norm=log_norm)
    #                 ax.set_xlabel(r"$\phi$")
    #                 ax.set_ylabel(r"$\psi$")
    #                 fig.tight_layout()
    #                 fig_path = model.getFigureDir(
    #                 ) + "/Comparison_latent_dynamics_{:}_region_{:}_ramaplot{:}{:}.{:}".format(
    #                     set_name, latent_region, log_norm_str, colorbar_str,
    #                     FIGTYPE)
    #                 plt.savefig(fig_path, bbox_inches="tight", dpi=100)
    #                 plt.close()
    return 0


def plotClusteredDynamics(model, results, set_name, testing_mode):
    if "autoencoder" in testing_mode:
        trajectories_target = results["input_sequence_all"]
        trajectories_pred = results["input_decoded_all"]
        latent_states_all = results["latent_states_all"]
    else:
        trajectories_target = results["targets_all"]
        trajectories_pred = results["predictions_all"]
        latent_states_all = results["latent_states_all"]

    print(np.shape(trajectories_pred))
    print(np.shape(trajectories_target))
    print(np.shape(latent_states_all))

    #####################################################################
    #    Plotting the clustering of the latent space
    #    and the clustering in the state space along with the Potential
    ####################################################################

    latent_states_pred = np.reshape(latent_states_all,
                                    (-1, *np.shape(latent_states_all)[2:]))
    trajectory_pred = np.reshape(trajectories_pred,
                                 (-1, *np.shape(trajectories_pred)[2:]))
    trajectory_target = np.reshape(trajectories_target,
                                   (-1, *np.shape(trajectories_target)[2:]))
    print(np.shape(trajectory_pred))
    print(np.shape(trajectory_target))
    print(np.shape(latent_states_pred))

    n_clusters = 4

    data = latent_states_pred
    print(np.shape(data))

    for cluster_type in ["GaussianMixture", "SpectralClustering"]:
        # for cluster_type in ["GaussianMixture", "SpectralClustering", "DBSCAN"]:
        # for cluster_type in ["DBSCAN"]:
        clustered_trajectory, trajectory_pred, trajectory_target, latent_states_pred = getClusteredTrajectory(
            trajectory_pred,
            trajectory_target,
            latent_states_pred,
            clustering_type=cluster_type,
            n_clusters=n_clusters)
        plotClusteredLatentDynamics(model, set_name, testing_mode,
                                    clustered_trajectory, trajectory_pred,
                                    latent_states_pred, n_clusters,
                                    cluster_type)


def plotClusteredLatentDynamics(model, set_name, testing_mode,
                                clustered_trajectory, trajectory_pred,
                                latent_states_pred, n_clusters, cluster_type):
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    # The trajectory is the predicted output, (after fed to the RNN) while the latent state is the one at the encoder output, before fed to RNN: fixing for this discrepancy:
    if "autoencoder" in testing_mode:
        trajectory_pred = np.array(trajectory_pred)
        latent_states_pred = np.array(latent_states_pred)
        clustered_trajectory = clustered_trajectory
    else:
        # trajectory_pred = np.array(trajectory_pred[:-1])
        trajectory_pred = np.array(trajectory_pred[1:])
        latent_states_pred = np.array(latent_states_pred[1:])
        clustered_trajectory = clustered_trajectory[1:]

    stateBAD = np.array(trajectory_pred)

    LABEL_COLOR_MAPS = getClusterColormapsAlanine()
    for cluster_id in range(n_clusters):
        for log_norm in [True, False]:
            for with_colorbar in [True, False]:
                log_norm_str = "Normed" if log_norm else ""
                colorbar_str = "Colorbar" if with_colorbar else ""
                fig, ax = plt.subplots()
                # print(np.shape(clustered_trajectory))
                phi = stateBAD[clustered_trajectory == cluster_id,
                               21] * 180 / np.pi
                psi = stateBAD[clustered_trajectory == cluster_id,
                               20] * 180 / np.pi
                # print("####################")
                # print(np.max(phi))
                # print(np.min(phi))
                # print(np.max(psi))
                # print(np.min(psi))
                # print(with_colorbar)
                # print(log_norm)
                if len(phi) > 0:
                    ax, mp = createRamachandranPlot(
                        phi,
                        psi,
                        ax,
                        cmap=LABEL_COLOR_MAPS[cluster_id],
                        with_colorbar=with_colorbar,
                        log_norm=log_norm)
                ax.set_xlabel(r"$\phi$")
                ax.set_ylabel(r"$\psi$")
                fig.tight_layout()
                plt.savefig(
                    model.getFigureDir() +
                    "/{:}_ramachandran_distr_scatter_{:}_clusters_{:}_C{:}{:}{:}.{:}"
                    .format(testing_mode, set_name, cluster_type, cluster_id,
                            log_norm_str, colorbar_str, FIGTYPE),
                    bbox_inches="tight",
                    dpi=100)
                plt.close()

    for log_norm in [True, False]:
        for with_colorbar in [True, False]:
            log_norm_str = "Normed" if log_norm else ""
            colorbar_str = "Colorbar" if with_colorbar else ""
            fig, ax = plt.subplots()
            for cluster_id in range(n_clusters):
                phi = stateBAD[clustered_trajectory == cluster_id,
                               21] * 180 / np.pi
                psi = stateBAD[clustered_trajectory == cluster_id,
                               20] * 180 / np.pi
                if len(phi) > 0:
                    ax, mp = createRamachandranPlot(
                        phi,
                        psi,
                        ax,
                        cmap=LABEL_COLOR_MAPS[cluster_id],
                        with_colorbar=with_colorbar,
                        log_norm=log_norm)
            ax.set_xlabel(r"$\phi$")
            ax.set_ylabel(r"$\psi$")
            fig.tight_layout()
            plt.savefig(
                model.getFigureDir() +
                "/{:}_ramachandran_distr_scatter_{:}_clusters_{:}_ALL{:}{:}.{:}"
                .format(testing_mode, set_name, cluster_type, log_norm_str,
                        colorbar_str, FIGTYPE),
                bbox_inches="tight",
                dpi=300)
            plt.close()

    print(np.shape(latent_states_pred))
    if np.shape(latent_states_pred)[1] > 1:
        # print(latent_states_pred)
        pca = PCA(n_components=2)
        pca.fit(latent_states_pred)
        latent_states_pca = pca.transform(latent_states_pred)

        for cluster_id in range(n_clusters):
            fig, ax = plt.subplots()
            # print(np.shape(clustered_trajectory))
            X = latent_states_pca[clustered_trajectory == cluster_id, 0]
            Y = latent_states_pca[clustered_trajectory == cluster_id, 1]
            ax.set_ylabel(r"PCA mode 1")
            ax.set_xlabel(r"PCA mode 2")
            if len(X) > 0:
                utils.scatterDensityLatentDynamicsPlot(
                    X,
                    Y,
                    ax=ax,
                    cmap=LABEL_COLOR_MAPS[cluster_id],
                    bins=50,
                    with_colorbar=True,
                    log_norm=False)
            fig.tight_layout()
            plt.savefig(
                model.getFigureDir() +
                "/{:}_PCA_on_latent_scatter_{:}_clusters_{:}_C{:}.{:}".format(
                    testing_mode, set_name, cluster_type, cluster_id, FIGTYPE),
                bbox_inches="tight",
                dpi=300)
            plt.close()

        fig, ax = plt.subplots()
        for cluster_id in range(n_clusters):
            # print(np.shape(clustered_trajectory))
            X = latent_states_pca[clustered_trajectory == cluster_id, 0]
            Y = latent_states_pca[clustered_trajectory == cluster_id, 1]
            if len(X) > 0:
                utils.scatterDensityLatentDynamicsPlot(
                    X,
                    Y,
                    ax=ax,
                    cmap=LABEL_COLOR_MAPS[cluster_id],
                    bins=20,
                    with_colorbar=False,
                    log_norm=True)
            # color_ = LABEL_COLOR_MAPS[cluster_id](0.6)
            # if len(X) > 0: mp = ax.scatter(X, Y, c=color_,rasterized=True)
        ax.set_ylabel(r"PCA mode 1")
        ax.set_xlabel(r"PCA mode 2")
        fig.tight_layout()
        plt.savefig(
            model.getFigureDir() +
            "/{:}_PCA_on_latent_scatter_{:}_clusters_{:}_all.{:}".format(
                testing_mode, set_name, cluster_type, FIGTYPE),
            bbox_inches="tight",
            dpi=300)
        plt.close()
    elif np.shape(latent_states_pred)[1] == 1:
        latent_states_pred_x = latent_states_pred[:-1]
        latent_states_pred_y = latent_states_pred[1:]
        # clustered_trajectory_ = clustered_trajectory[:-1]
        clustered_trajectory_ = clustered_trajectory[1:]

        print(np.shape(clustered_trajectory))
        print(np.shape(latent_states_pred_x))
        print(np.shape(latent_states_pred_y))
        print(np.shape(clustered_trajectory_))

        for cluster_id in range(n_clusters):
            fig, ax = plt.subplots()
            # print(np.shape(clustered_trajectory_))
            X = latent_states_pred_x[clustered_trajectory_ == cluster_id, 0]
            Y = latent_states_pred_y[clustered_trajectory_ == cluster_id, 0]
            ax.set_xlabel(r"$\mathbf{z}_t$")
            ax.set_ylabel(r"$\mathbf{z}_{t+1}$")
            if len(X) > 0:
                utils.scatterDensityLatentDynamicsPlot(
                    X,
                    Y,
                    ax=ax,
                    cmap=LABEL_COLOR_MAPS[cluster_id],
                    bins=50,
                    with_colorbar=True,
                    log_norm=False)
            fig.tight_layout()
            plt.savefig(
                model.getFigureDir() +
                "/{:}_state_latent_scatter_{:}_clusters_{:}_C{:}.{:}".format(
                    testing_mode, set_name, cluster_type, cluster_id, FIGTYPE),
                bbox_inches="tight",
                dpi=300)
            plt.close()

        fig, ax = plt.subplots()
        for cluster_id in range(n_clusters):
            # print(np.shape(clustered_trajectory_))
            X = latent_states_pred_x[clustered_trajectory_ == cluster_id, 0]
            Y = latent_states_pred_y[clustered_trajectory_ == cluster_id, 0]
            if len(X) > 0:
                utils.scatterDensityLatentDynamicsPlot(
                    X,
                    Y,
                    ax=ax,
                    cmap=LABEL_COLOR_MAPS[cluster_id],
                    bins=20,
                    with_colorbar=False,
                    log_norm=True)
            # color_ = LABEL_COLOR_MAPS[cluster_id](0.6)
            # if len(X) > 0: mp = ax.scatter(X, Y, c=color_,rasterized=True)
        ax.set_xlabel(r"$\mathbf{z}_t$")
        ax.set_ylabel(r"$\mathbf{z}_{t+1}$")
        fig.tight_layout()
        plt.savefig(
            model.getFigureDir() +
            "/{:}_state_latent_scatter_{:}_clusters_{:}_all.{:}".format(
                testing_mode, set_name, cluster_type, FIGTYPE),
            bbox_inches="tight",
            dpi=300)
        plt.close()


def getClusteredTrajectory(trajectory_pred,
                           trajectory_target,
                           latent_states_pred,
                           clustering_type="GaussianMixture",
                           n_clusters=4):
    if clustering_type == "GaussianMixture":
        # Subsampling up to at most 500 points
        MAX_POINTS = 10000
        SUBSAMPLE = 1 if len(latent_states_pred) < MAX_POINTS else int(
            len(latent_states_pred) / MAX_POINTS)
        trajectory_pred = trajectory_pred[::SUBSAMPLE]
        trajectory_target = trajectory_target[::SUBSAMPLE]
        latent_states_pred = latent_states_pred[::SUBSAMPLE]
        print("GaussianMixture clustering...")
        clustering = GaussianMixture(
            n_components=n_clusters).fit(latent_states_pred)
        print("GaussianMixture clustering finished!")
        clustered_trajectory = clustering.predict(latent_states_pred)
        clustered_trajectory = np.array(clustered_trajectory)
    elif clustering_type == "SpectralClustering":
        # Subsampling up to at most 500 points
        MAX_POINTS = 10000
        SUBSAMPLE = 1 if len(latent_states_pred) < MAX_POINTS else int(
            len(latent_states_pred) / MAX_POINTS)
        trajectory_pred = trajectory_pred[::SUBSAMPLE]
        trajectory_target = trajectory_target[::SUBSAMPLE]
        latent_states_pred = latent_states_pred[::SUBSAMPLE]
        print("Spectral clustering...")
        clustering = SpectralClustering(n_clusters=n_clusters,
                                        assign_labels="discretize",
                                        random_state=0).fit(latent_states_pred)
        print("Spectral clustering finished!")
        clustered_trajectory = clustering.labels_
    elif clustering_type == "DBSCAN":
        # Subsampling up to at most 500 points
        MAX_POINTS = 10000
        SUBSAMPLE = 1 if len(latent_states_pred) < MAX_POINTS else int(
            len(latent_states_pred) / MAX_POINTS)
        trajectory_pred = trajectory_pred[::SUBSAMPLE]
        trajectory_target = trajectory_target[::SUBSAMPLE]
        latent_states_pred = latent_states_pred[::SUBSAMPLE]
        print("OPTICS clustering...")
        clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=.01)
        print("OPTICS clustering finished!")
        clust.fit(latent_states_pred)
        clustered_trajectory = cluster_optics_dbscan(
            reachability=clust.reachability_,
            core_distances=clust.core_distances_,
            ordering=clust.ordering_,
            eps=0.1)
    else:
        raise ValueError(
            "Unknown clustering type {:}.".format(clustering_type))
    return clustered_trajectory, trajectory_pred, trajectory_target, latent_states_pred


def makeRamaPlots(model, set_name, testing_mode, results):

    rama_density_target = results["rama_density_target"]
    rama_density_predicted = results["rama_density_predicted"]
    rama_bin_centers = results["rama_bin_centers"]
    rama_l1_hist_error = results["rama_l1_hist_error"]

    rama_targets_all = results["rama_targets"]
    rama_predictions_all = results["rama_predictions"]

    rama_predictions = np.reshape(rama_predictions_all,
                                  (-1, *np.shape(rama_predictions_all)[2:]))
    rama_targets = np.reshape(rama_targets_all,
                              (-1, *np.shape(rama_targets_all)[2:]))
    print(np.shape(rama_predictions))
    print(np.shape(rama_targets))

    # rama_predictions = rama_predictions[:100]
    # rama_targets = rama_targets[:100]

    #################################################
    ## RAMA PLOTS FOR ALL THE ICS
    #################################################

    textfontsize = 12
    for norm_ in [False, True]:
        print(norm_)
        norm_str = "_normed" if norm_ else ""
        ncols = 2
        nrows = 1
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 figsize=(4.25 * ncols, 4.25 * 5/6 * nrows),
                                 squeeze=False)

        axes[0, 0].set_title("Target Density")
        phi = rama_targets[:, 0]
        psi = rama_targets[:, 1]
        axes[0, 0], mp = createRamachandranPlot(phi,
                                                psi,
                                                axes[0, 0],
                                                with_colorbar=False,
                                                log_norm=norm_,
                                                fontsize=textfontsize,
                                                )
        fig.colorbar(mp, ax=axes[0, 0])

        axes[0, 1].set_title("Predicted Density")
        phi = rama_predictions[:, 0]
        psi = rama_predictions[:, 1]
        axes[0, 1], mp = createRamachandranPlot(phi,
                                                psi,
                                                axes[0, 1],
                                                with_colorbar=False,
                                                log_norm=norm_,
                                                fontsize=textfontsize,
                                                )
        fig.colorbar(mp, ax=axes[0, 1])
        fig.tight_layout()
        fig_path = model.getFigureDir(
        ) + "/{:}_ramachandran_distr_scatter_{:}_ALL{:}.{:}".format(
            testing_mode, set_name, norm_str, FIGTYPE)
        plt.savefig(fig_path, dpi=300)
        plt.close()

    for norm_ in [False, True]:
        print(norm_)
        norm_str = "_normed" if norm_ else ""
        ncols = 1
        nrows = 1

        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 figsize=(4.25 * ncols, 4.25 * 5/6 * nrows),
                                 squeeze=False)
        # axes[0, 0].set_title("Target Density")
        phi = rama_targets[:, 0]
        psi = rama_targets[:, 1]
        axes[0, 0], mp = createRamachandranPlot(phi,
                                                psi,
                                                axes[0, 0],
                                                with_colorbar=True,
                                                log_norm=norm_,
                                                fontsize=textfontsize,
                                                )
        # fig.colorbar(mp, ax=axes[0, 0])
        fig.tight_layout()
        fig_path = model.getFigureDir(
        ) + "/{:}_ramachandran_distr_scatter_{:}_ALL{:}_target.{:}".format(
            testing_mode, set_name, norm_str, FIGTYPE)
        plt.savefig(fig_path, dpi=300)
        plt.close()

        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 figsize=(4.25 * ncols, 4.25 * 5/6 * nrows),
                                 squeeze=False)
        # axes[0, 0].set_title("Predicted Density")
        phi = rama_predictions[:, 0]
        psi = rama_predictions[:, 1]
        axes[0, 0], mp = createRamachandranPlot(phi,
                                                psi,
                                                axes[0, 0],
                                                with_colorbar=True,
                                                log_norm=norm_,
                                                fontsize=textfontsize,
                                                )
        # fig.colorbar(mp, ax=axes[0, 0])
        fig.tight_layout()
        fig_path = model.getFigureDir(
        ) + "/{:}_ramachandran_distr_scatter_{:}_ALL{:}_prediction.{:}".format(
            testing_mode, set_name, norm_str, FIGTYPE)
        plt.savefig(fig_path, dpi=300)
        plt.close()

    # num_ics, T, D = np.shape(rama_targets_all)

    # print(np.shape(rama_predictions_all))
    # print(np.shape(rama_targets_all))

    # max_ics_plot = 3
    # ics_plot = np.min([num_ics, max_ics_plot])
    # for ic in range(ics_plot):
    #     rama_predictions = rama_predictions_all[ic]
    #     rama_targets = rama_targets_all[ic]

    #     #################################################
    #     ## SCATTER PLOTS
    #     #################################################

    #     z_targ = interpn(rama_bin_centers, rama_density_target, rama_targets, method = "splinef2d", bounds_error = False, fill_value=0.0)
    #     z_pred = interpn(rama_bin_centers, rama_density_predicted, rama_predictions, method = "splinef2d", bounds_error = False, fill_value=0.0)
    #     print(np.shape(z_targ))
    #     print(np.shape(z_pred))

    #     # Sort the points by density, so that the densest points are plotted last
    #     idx = z_targ.argsort()
    #     rama_targets, z_targ = rama_targets[idx], z_targ[idx]

    #     idx = z_pred.argsort()
    #     rama_predictions, z_pred = rama_predictions[idx], z_pred[idx]

    #     for norm_ in [False, True]:
    #         print(norm_)
    #         norm_str = "_normed" if norm_ else ""
    #         ncols=2
    #         nrows=1

    #         # Densities cannot be smaller than zero (interpolation might though)
    #         z_targ_plot = np.where(z_targ>0,z_targ,np.nan)
    #         z_pred_plot = np.where(z_pred>0,z_pred,np.nan)
    #         vmin = np.nanmin(z_targ_plot)
    #         vmax = np.nanmax(z_targ_plot)

    #         if vmax > 0.0:

    #             fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(6*ncols, 5*nrows), squeeze=False)

    #             axes[0,0].set_title("Target Density")
    #             phi = rama_targets[:,0]
    #             psi = rama_targets[:,1]
    #             axes[0,0], mp = createRamachandranPlot(phi, psi, axes[0,0], with_colorbar=False, log_norm=norm_)
    #             fig.colorbar(mp, ax=axes[0,0])

    #             axes[0,1].set_title("Predicted Density")
    #             phi = rama_predictions[:,0]
    #             psi = rama_predictions[:,1]
    #             axes[0,1], mp = createRamachandranPlot(phi, psi, axes[0,1], with_colorbar=False, log_norm=norm_)
    #             fig.colorbar(mp, ax=axes[0,1])
    #             fig.tight_layout()
    #             fig_path = model.getFigureDir() + "/{:}_ramachandran_distr_scatter_{:}_IC{:}{:}.{:}".format(testing_mode,set_name,ic,norm_str,FIGTYPE)
    #             plt.savefig(fig_path, dpi=300)
    #             # plt.show()
    #             plt.close()

    #             for norm_ in [False, True]:
    #                 print(norm_)
    #                 norm_str = "_normed" if norm_ else ""
    #                 ncols=1
    #                 nrows=1

    #                 fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(6*ncols, 5*nrows), squeeze=False)
    #                 axes[0,0].set_title("Target Density")
    #                 phi = rama_targets[:,0]
    #                 psi = rama_targets[:,1]
    #                 axes[0,0], mp = createRamachandranPlot(phi, psi, axes[0,0], with_colorbar=False, log_norm=norm_)
    #                 fig.colorbar(mp, ax=axes[0,0])
    #                 fig.tight_layout()
    #                 fig_path = model.getFigureDir() + "/{:}_ramachandran_distr_scatter_{:}_IC{:}{:}_target.{:}".format(testing_mode,set_name,ic,norm_str,FIGTYPE)
    #                 plt.savefig(fig_path, dpi=300)
    #                 plt.close()

    #                 fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(6*ncols, 5*nrows), squeeze=False)
    #                 axes[0,0].set_title("Predicted Density")
    #                 phi = rama_predictions[:,0]
    #                 psi = rama_predictions[:,1]
    #                 axes[0,0], mp = createRamachandranPlot(phi, psi, axes[0,0], with_colorbar=False, log_norm=norm_)
    #                 fig.colorbar(mp, ax=axes[0,0])
    #                 fig.tight_layout()
    #                 fig_path = model.getFigureDir() + "/{:}_ramachandran_distr_scatter_{:}_IC{:}{:}_prediction.{:}".format(testing_mode,set_name,ic,norm_str,FIGTYPE)
    #                 plt.savefig(fig_path, dpi=300)
    #                 plt.close()


def createRamachandranPlot(phi,
                           psi,
                           ax=None,
                           cmap=plt.get_cmap("Reds"),
                           with_colorbar=True,
                           log_norm=True,
                           fontsize=None,
                           ):
    x, y = phi, psi
    sort = True
    bins = 72
    assert (np.max(phi) <= 180)
    assert (np.max(psi) <= 180)
    assert (np.min(phi) >= -180)
    assert (np.min(psi) >= -180)
    range_ = [[-180, 180], [-180, 180]]
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, range=range_)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
                data,
                np.vstack([x, y]).T,
                method="splinef2d",
                bounds_error=False,
                fill_value=0.0)
    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    # # Maximum plotting points
    # N_max = 10000
    # N_subsample = int(np.shape(x)/N_max)
    # x = x[::N_subsample]
    # y = y[::N_subsample]
    # z = z[::N_subsample]

    if ax is None:
        fig, ax = plt.subplots()
    if log_norm:
        mp = ax.scatter(x,
                        y,
                        c=z,
                        cmap=cmap,
                        norm=matplotlib.colors.LogNorm(),
                        rasterized=True)
    else:
        mp = ax.scatter(x, y, c=z, cmap=cmap, rasterized=True)

    # mp = ax.scatter( x, y, cmap=cmap, norm=matplotlib.colors.LogNorm())
    # if with_colorbar: plt.colorbar(mp)
    if with_colorbar: plt.colorbar(mp, fraction=0.046, pad=0.04)

    ax = addRamachandranPlotText(ax, fontsize=fontsize)
    return ax, mp


def plotRamachadranContourf(phi, psi):
    from scipy.stats import kde
    x = phi
    y = psi
    data = np.concatenate((np.reshape(x, (1, -1)), np.reshape(y, (1, -1))),
                          axis=0)
    nbins = 72
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde(data)
    # xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    xi, yi = np.mgrid[-180:180:nbins * 1j, -180:180:nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi_log = np.log10(zi)
    zi_plot = zi_log.reshape(xi.shape)
    # zi_plot = zi.reshape(xi.shape)
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(10, 8))

    z_min = -8.0
    zi_plot[zi_plot < z_min] = z_min
    vmin = np.nanmin(zi_plot[zi_plot != -np.inf])
    vmax = np.nanmax(zi_plot)
    levels = np.linspace(vmin, vmax, 20)
    # cmap = plt.get_cmap("RdGy")
    cmap = plt.get_cmap("Reds")
    contours = axes.contour(xi, yi, zi_plot, cmap=cmap, levels=levels)
    # axes.pcolormesh(xi, yi, zi_plot, cmap=cmap, shading='gouraud')
    # axes.pcolormesh(xi, yi, zi_plot, shading='gouraud', cmap=cmap)
    plt.contourf(xi, yi, zi_plot, cmap=cmap)

    plt.colorbar()
    plt.xlabel(r"Dihedral angle $\phi$")
    plt.ylabel(r"Dihedral angle $\psi$")

    for pathcoll in contours.collections:
        pathcoll.set_rasterized(True)

    addRamachandranPlotText(axes)
    return fig, axes


def plotRamachadranContour(phi, psi):
    from scipy.stats import kde
    x = phi
    y = psi
    data = np.concatenate((np.reshape(x, (1, -1)), np.reshape(y, (1, -1))),
                          axis=0)
    nbins = 72
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde(data)
    # xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    xi, yi = np.mgrid[-180:180:nbins * 1j, -180:180:nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi_log = np.log10(zi)
    zi_plot = zi_log.reshape(xi.shape)
    # zi_plot = zi.reshape(xi.shape)
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(10, 8))
    z_min = -8.0
    zi_plot[zi_plot < z_min] = z_min
    vmin = np.nanmin(zi_plot[zi_plot != -np.inf])
    vmax = np.nanmax(zi_plot)
    levels = np.linspace(vmin, vmax, 20)
    # cmap = plt.get_cmap("RdGy")
    cmap = plt.get_cmap("Reds")
    contours = axes.contour(xi, yi, zi_plot, cmap=cmap, levels=levels)
    plt.clabel(contours, inline=True, fontsize=10)
    # plt.imshow(zi_plot)
    # plt.colorbar()
    plt.xlabel(r"Dihedral angle $\phi$")
    plt.ylabel(r"Dihedral angle $\psi$")

    for pathcoll in contours.collections:
        pathcoll.set_rasterized(True)

    addRamachandranPlotText(axes)

    return fig, axes


def generateFiles(saving_dir, formated_str, traj_bad, ref_conf_file_path):
    # Write bad file
    filename_bad = saving_dir + '/{:}_bad_file.txt'.format(formated_str)
    np.savetxt(filename_bad, traj_bad, fmt="%15.10f")

    conffile = saving_dir + '/{:}_conf_file.xyz'.format(formated_str)
    lammps_file = saving_dir + '/{:}_lammps_file.xyz'.format(formated_str)

    utils_processing_alanine.generateXYZfileFromABDnoRotTr(
        filename_bad, ref_conf_file_path, conffile, lammps_file)
    return 0


def generateXYZfromABDtrajectoriesAlanine(model, results, set_name,
                                          testing_mode):

    print("# generateXYZfromABDtrajectoriesAlanine() #")

    if "autoencoder" in testing_mode:
        targets_all = results["input_sequence_all"]
        predictions_all = results["input_decoded_all"]
    else:
        targets_all = results["targets_all"]
        predictions_all = results["predictions_all"]

    ref_conf_file_path = model.params[
        "project_path"] + "/Methods/LED/Systems/Alanine/alphaR_COMP.txt"
    print(
        "# Looking for reference conf. file:\n{:}".format(ref_conf_file_path))

    if not os.path.isfile(ref_conf_file_path):
        print("# Error: reference conf. file:\n{:}\nnot found.".format(
            ref_conf_file_path))

    saving_dir = model.getFigureDir() + "/Trajectories"
    os.makedirs(saving_dir, exist_ok=True)

    cluster_labels = utils_processing_alanine.getRamaClusterLabelsAlanineSimpleFormat(
    )

    # num_ics = 10
    # # Generate BAD file from targets
    # for ic in range(num_ics):
    #     for type_ in ["target", "prediction"]:
    #         if type_ == "target":
    #             traj = targets_all[ic]
    #         elif type_ == "prediction":
    #             traj = predictions_all[ic]
    #         else:
    #             raise ValueError("Error.")

    #         # # Write bad file
    #         # filename_bad = model.saving_path + model.results_dir + model.model_name + '/{:}_{:}_ic{:}_{:}_bad_file.txt'.format(testing_mode, set_name, ic, type_)
    #         # np.savetxt(filename_bad, traj, fmt="%15.10f")
    #         # conffile = model.saving_path + model.results_dir + model.model_name + '/{:}_{:}_ic{:}_{:}_conf_file.xyz'.format(testing_mode, set_name, ic, type_)
    #         # lammps_file = model.saving_path + model.results_dir + model.model_name + '/{:}_{:}_ic{:}_{:}_lammps_file.xyz'.format(testing_mode, set_name, ic, type_)
    #         # generateXYZfileFromABDnoRotTr(filename_bad, ref_conf_file_path, conffile, lammps_file)

    #         # Cluster trajectory
    #         clustered_trj = utils_processing_alanine.clusterTrajectoryAlanine(
    #             traj)
    #         cluster_id = 0
    #         for cluster_id in [0, 1, 2, 3, 4]:
    #             idx = np.where(clustered_trj == cluster_id)[0]
    #             if len(idx) > 0:
    #                 cluster_label = cluster_labels[cluster_id]
    #                 print("# Cluster {:} / {:} found in trajectories.".format(
    #                     cluster_id, cluster_label))
    #                 index = np.random.choice(idx)
    #                 if index > 100:

    #                     formated_str = "{:}_{:}_{:}_ic{:}_{:}".format(
    #                         testing_mode, set_name, cluster_label, ic, type_)

    #                     # print(index)
    #                     # min_idx = np.min([0, index-10])
    #                     # max_idx = np.max([np.shape(traj)[0], index+1000])
    #                     max_idx = np.min([np.shape(traj)[0], index + 500])
    #                     traj_cluster = traj[index:max_idx].copy()

    #                     # Write bad file
    #                     filename_bad = saving_dir + '/{:}_bad_file.txt'.format(
    #                         formated_str)
    #                     np.savetxt(filename_bad, traj_cluster, fmt="%15.10f")

    #                     conffile = saving_dir + '/{:}_conf_file.xyz'.format(
    #                         formated_str)
    #                     lammps_file = saving_dir + '/{:}_lammps_file.xyz'.format(
    #                         formated_str)

    #                     utils_processing_alanine.generateXYZfileFromABDnoRotTr(
    #                         filename_bad, ref_conf_file_path, conffile,
    #                         lammps_file)

    num_ics_plot = 20
    targets_all = targets_all[:num_ics_plot]
    predictions_all = predictions_all[:num_ics_plot]

    clustered_trajectories_target = [
        utils_processing_alanine.clusterTrajectoryAlanine(targets_all[ic])
        for ic in range(num_ics_plot)
    ]
    clustered_trajectories_target = np.array(clustered_trajectories_target)
    print(np.shape(clustered_trajectories_target))

    print(np.shape(targets_all))
    print(np.shape(predictions_all))

    def nearest_neighbors(values, all_values, nbr_neighbors=5):
        nn = NearestNeighbors(nbr_neighbors,
                              metric='cosine',
                              algorithm='brute').fit(all_values)
        dists, idxs = nn.kneighbors(values)
        return dists, idxs

    for cluster_id in [0, 1, 2, 3, 4]:
        # for cluster_id in [0]:
        cluster_label = cluster_labels[cluster_id]

        saving_dir = model.getFigureDir(
        ) + "/SamplesAndNeighbors_{:}_C{:}_{:}".format(
            testing_mode, cluster_id, cluster_label)
        os.makedirs(saving_dir, exist_ok=True)

        idx_row, idx_col = np.where(
            clustered_trajectories_target == cluster_id)
        print(idx_row)
        print(idx_col)

        if len(idx_row) > 0:

            idx = np.arange(len(idx_row))
            idx = np.random.choice(idx)

            idx_row = idx_row[idx]
            idx_col = idx_col[idx]

            target = targets_all[idx_row, idx_col]

            predictions_all_candidates = np.reshape(
                predictions_all, (-1, np.shape(predictions_all)[2]))
            target = np.reshape(target, (1, np.shape(target)[0]))

            print(np.shape(target))
            print(np.shape(predictions_all_candidates))

            dists, idxs = nearest_neighbors(target,
                                            predictions_all_candidates,
                                            nbr_neighbors=10)
            dists = dists[0]
            idxs = idxs[0]
            print(dists)
            print(idxs)

            formated_str = "{:}_{:}".format(set_name, "target")
            traj_bad = target
            generateFiles(saving_dir, formated_str, traj_bad,
                          ref_conf_file_path)

            for i in range(len(idxs)):
                idx_neighbor = idxs[i]
                formated_str = "{:}_{:}".format(set_name,
                                                "neighbor_{:}".format(i))
                traj_bad = predictions_all_candidates[
                    idx_neighbor:idx_neighbor + 1]
                generateFiles(saving_dir, formated_str, traj_bad,
                              ref_conf_file_path)

    return 0
