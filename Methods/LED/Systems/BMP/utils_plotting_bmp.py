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

# FIGTYPE="pdf"
FIGTYPE = "pdf"

from scipy.interpolate import interpn
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from .mueller_potential import *
from . import utils_processing_bmp

from ... import Utils as utils


def plotLatentDynamicsComparisonSystemBMP(model, set_name):
    print("# plotLatentDynamicsComparisonSystemBMP() #")
    tf_targets_all, tf_latent_states_all, latent_state_regions = utils.plotLatentDynamicsComparisonSystemCommon(
        model, set_name)
    print(ark)
    colormaps = [
        plt.get_cmap("Reds"),
        plt.get_cmap("Reds"),
        plt.get_cmap("Reds"),
        plt.get_cmap("Reds"),
    ]

    latent_states = np.reshape(tf_latent_states_all,
                               (-1, np.shape(tf_latent_states_all)[2]))
    targets_all = np.reshape(tf_targets_all, (-1, np.shape(tf_targets_all)[2]))
    for region_iter in range(len(latent_state_regions)):
        if region_iter > len(colormaps) - 1:
            break
        latent_region = latent_state_regions[region_iter]
        idx_region = np.bitwise_and(latent_states > latent_region[0],
                                    latent_states < latent_region[1])
        idx_region = idx_region[:, 0]
        idx_region = np.where(idx_region)[0]
        idx_region = list(idx_region)
        if 0 in idx_region: idx_region.remove(0)
        idx_region = np.array(idx_region) - 1

        if len(idx_region) > 0:
            targets_all_region = targets_all[idx_region]

            for second_axis in [True, False]:
                # for colorbar_ in [True, False]:
                colorbar_ = False
                colorbar_str = "" if not colorbar_ else "_colorbar"
                second_axis_str = "" if not second_axis else "_second_axis"
                fig, ax = plt.subplots(figsize=(12, 6))
                mbp = muellerBrownPotential()
                ax = mbp.plotPotential(save=False,
                                       ax=ax,
                                       cmap=plt.get_cmap("gist_gray"),
                                       colorbar=colorbar_)
                X = targets_all_region[:, 0]
                Y = targets_all_region[:, 1]
                createDensityScatterPlot(X,
                                         Y,
                                         ax=ax,
                                         fig=fig,
                                         with_colorbar=True,
                                         cmap=plt.get_cmap("Reds"),
                                         second_axis=second_axis)
                fig_path = model.getFigureDir(
                ) + "/Comparison_latent_dynamics_{:}_region_{:}{:}{:}.{:}".format(
                    set_name, latent_region, colorbar_str, second_axis_str,
                    FIGTYPE)
                plt.savefig(fig_path, bbox_inches='tight', dpi=100)
                plt.close()
    return 0


def plotLatentMetaStableStatesBMP(model, results, set_name, testing_mode):

    if "autoencoder" in testing_mode:
        data_all = results["input_sequence_all"]
        latent_states_all = results["latent_states_all"]

    elif "iterative" in testing_mode:
        data_all = results["predictions_all"][:, :-1]
        latent_states_all = results["latent_states_all"][:, 1:]

    elif "teacher_forcing" in testing_mode:
        data_all = results["targets_all"][:, :-1]
        latent_states_all = results["latent_states_all"][:, 1:]

    else:
        raise ValueError(
            "testing_mode = {:} not recognized.".format(testing_mode))

    print("np.shape(data_all) = {:}".format(np.shape(data_all)))
    print("np.shape(latent_states_all) = {:}".format(
        np.shape(latent_states_all)))

    latent_state_regions = results["free_energy_latent_state_regions"]
    latent_cluster_labels = results["free_energy_latent_cluster_labels"]
    latent_state_freenergy = results["latent_state_freenergy"]
    latent_state_free_energy_grid = results["latent_state_free_energy_grid"]

    colormaps = [
        plt.get_cmap("Reds"),
        plt.get_cmap("Reds"),
        plt.get_cmap("Reds"),
        plt.get_cmap("Reds"),
        plt.get_cmap("Reds"),
        plt.get_cmap("Reds"),
    ]

    latent_states = np.reshape(latent_states_all,
                               (-1, np.shape(latent_states_all)[2]))
    data_all = np.reshape(data_all, (-1, np.shape(data_all)[2]))

    for region_iter in range(len(latent_state_regions)):

        if region_iter > len(colormaps) - 1:
            break

        latent_region = latent_state_regions[region_iter]
        idx_region = np.bitwise_and(latent_states > latent_region[0],
                                    latent_states < latent_region[1])
        idx_region = idx_region[:, 0]
        idx_region = np.where(idx_region)[0]
        idx_region = list(idx_region)

        if 0 in idx_region: idx_region.remove(0)

        idx_region = np.array(idx_region)

        if len(idx_region) > 0:
            data_all_region = data_all[idx_region]

            for colorbar_ in [True, False]:
                if colorbar_ in [False]:
                    second_axis_array = [True, False]
                else:
                    second_axis_array = [True]

                for second_axis in second_axis_array:
                    colorbar_str = "" if not colorbar_ else "_colorbar"
                    second_axis_str = "" if not second_axis else "_second_axis"

                    fig, ax = plt.subplots(figsize=(12, 6))
                    mbp = muellerBrownPotential()
                    # ax = utils_processing_bmp.plotClusters(ax, plot1=True, plot2=True, color1="k", color2="k")
                    X = data_all_region[:, 0]
                    Y = data_all_region[:, 1]
                    ax = mbp.plotPotential(save=False,
                                           ax=ax,
                                           cmap=plt.get_cmap("gist_gray"),
                                           colorbar=colorbar_)
                    createDensityScatterPlot(X,
                                             Y,
                                             ax=ax,
                                             fig=fig,
                                             with_colorbar=True,
                                             cmap=plt.get_cmap("Reds"),
                                             second_axis=second_axis,
                                             log=False)
                    ax.set_xlim([-2, 2])
                    ax.set_ylim([-1, 3])
                    plt.xlabel(r'$x_1$')
                    plt.ylabel(r'$x_2$')
                    fig_path = model.getFigureDir(
                    ) + "/{:}_{:}_latent_meta_stable_states_region_{:}{:}{:}.{:}".format(
                        testing_mode, set_name, latent_region, colorbar_str,
                        second_axis_str, FIGTYPE)
                    plt.savefig(fig_path, bbox_inches='tight', dpi=100)
                    plt.close()

    return 0


def createDensityScatterPlot(x,
                             y,
                             ax=None,
                             fig=None,
                             cmap=plt.get_cmap("gist_yarg"),
                             with_colorbar=True,
                             second_axis=False,
                             log=True):
    sort = True
    bins = 40
    range_ = [[-2, 2], [-1, 3]]
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

    if log: z = np.log(z)
    mp = ax.scatter(x, y, c=z, cmap=cmap, rasterized=True)

    if with_colorbar and not second_axis:
        cbar = plt.colorbar(mp, ax=ax)
        if log:
            cbar.set_label('$\log \, p(\mathbf{x})$', rotation=0, labelpad=30)
        else:
            cbar.set_label('$p(\mathbf{x})$', rotation=0, labelpad=30)
    elif with_colorbar and second_axis:
        cbar = plt.colorbar(mp, ax=[ax], location='left', pad=0.15)
        if log:
            cbar.set_label('$\log \, p(\mathbf{x})$', rotation=0, labelpad=30)
        else:
            cbar.set_label('$p(\mathbf{x})$', rotation=0, labelpad=30)
    return ax, mp


def plotLatentTransitionTimesBMP(model, results, set_name, testing_mode):
    cluster_labels = results["free_energy_latent_cluster_labels"]
    times_pred = results["latent_cluster_mean_passage_times_dt_msm_small"]
    file_path = model.getFigureDir(
    ) + "/{:}_{:}_latent_meta_stable_states_mean_passage_times_dt_msm_small.{:}".format(
        testing_mode, set_name, "txt")
    utils.writeLatentTransitionTimesToFile(model, file_path, cluster_labels,
                                           times_pred)


def plotTransitionTimesBMP(model, results, set_name, testing_mode):
    cluster_transition_times_target = results[
        "cluster_transition_times_target"]
    cluster_transition_times_pred = results["cluster_transition_times_pred"]
    dt = results["dt"]
    for key in cluster_transition_times_target:
        times_targ = cluster_transition_times_target[key]
        times_pred = cluster_transition_times_pred[key]
        i, j = key

        num_transitions_targ = len(times_targ)
        num_transitions_pred = len(times_pred)

        if num_transitions_targ > 0:
            max_time_target = np.max(times_targ)
            mean_time_target = np.mean(times_targ)
        else:
            max_time_target = 0.0
            mean_time_target = 0.0

        if num_transitions_pred > 0:
            max_time_pred = np.max(times_pred)
            mean_time_pred = np.mean(times_pred)
        else:
            max_time_pred = 0.0
            mean_time_pred = 0.0

        label_pred = 'Predicted'
        label_pred += " $\mathrm{max}(T)=" + "{:.0f}".format(
            max_time_pred) + "$"
        label_pred += ", $\mathrm{mean}(T)=" + "{:.0f}".format(
            mean_time_pred) + "$"
        label_pred += ", N={:}".format(num_transitions_pred)

        label_targ = 'Target'
        label_targ += " $\mathrm{max}(T)=" + "{:.0f}".format(
            max_time_target) + "$"
        label_targ += ", $\mathrm{mean}(T)=" + "{:.0f}".format(
            mean_time_target) + "$"
        label_targ += ", N={:}".format(num_transitions_targ)

        if num_transitions_targ > 0:

            LL_ = np.max(times_targ) - np.min(times_targ)
            N_samples = np.shape(times_targ)[0]
            nbins = utils.getNumberOfBins(N_samples, rule="rice")

            bins = np.linspace(np.min(times_targ), np.max(times_targ), nbins)

            fig, axes = plt.subplots()
            n, bins, patches = plt.hist([times_targ, times_pred], bins, alpha=0.75, \
                label=[r"{:}".format(label_targ), r"{:}".format(label_pred)], \
                color=['green', 'blue'])

            # n, bins, patches = plt.hist(times_targ, bins, facecolor='green', alpha=0.75, label=r"{:}".format(label_targ))
            # n, bins, patches = plt.hist(times_pred, bins, facecolor='blue', alpha=0.75, label=r"{:}".format(label_pred))
            # lgd = axes.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0., frameon=False)
            lgd = axes.legend(loc="upper center",
                              bbox_to_anchor=(0.5, 1.225),
                              borderaxespad=0.,
                              frameon=False)

            plt.xlabel('Transition times')
            plt.ylabel('Counts')

            plt_title = '$\mathrm{Histogram\ of}\ T_{' + '{:}'.format(
                i) + '\\to ' + '{:}'.format(j) + '}$'

            # plt_title += " $\overline{T}=" + "{:.0f}".format(max_time_target) +"$"
            # plt_title += ", \tilde{N}={:}".format(num_transitions_pred)
            # plt_title += ", $ \Delta t={:.3f}".format(dt) +"$"
            plt_title += ", $ \Delta t={:.1f}".format(dt) + "$"
            plt.title(r"{:}".format(plt_title), pad=68)
            plt.grid(True)
            # plt.tight_layout()

            fig_path = model.getFigureDir(
            ) + "/{:}_{:}_BMP_trans_time_from_{:}_to_{:}.{:}".format(
                testing_mode, set_name, i, j, FIGTYPE)
            plt.savefig(fig_path,
                        bbox_extra_artists=(lgd, ),
                        bbox_inches='tight')
            plt.close()

    return 0


def plotStateDistributionsSystemBMP(model, results, set_name, testing_mode):
    print(testing_mode)
    if "autoencoder" in testing_mode:
        trajectories_target = results["input_sequence_all"]
        trajectories_pred = results["input_decoded_all"]
        latent_states_all = results["latent_states_all"]
    else:
        trajectories_target = results["targets_all"]
        trajectories_pred = results["predictions_all"]
        latent_states_all = results["latent_states_all"]

    # for key in results:
    #     print(key)

    print(np.shape(trajectories_pred))
    print(np.shape(trajectories_target))

    #####################################################################
    #    Plotting samples trajectories in the Potential space
    ####################################################################

    # fig, ax = plt.subplots()
    # mbp = muellerBrownPotential()
    # ax = mbp.plotPotential(save=False, ax=ax)
    # for i in range(len(trajectories_target)):
    #     traj = trajectories_target[i]
    #     ax.plot(traj[:, 0], traj[:, 1], 'x-', rasterized=True)
    # # ax.set_title("BMP trajectories")
    # ax.set_xlim([-2,2])
    # ax.set_ylim([-1,3])
    # fig_path = model.getFigureDir() + "/{:}_{:}_BMP_TARGET_contour_lines.{:}".format(testing_mode,set_name,FIGTYPE)
    # plt.savefig(fig_path, bbox_inches='tight')
    # plt.close()

    # fig, ax = plt.subplots()
    # mbp = muellerBrownPotential()
    # ax = mbp.plotPotential(save=False, ax=ax)
    # for i in range(len(trajectories_pred)):
    #     traj = trajectories_pred[i]
    #     ax.plot(traj[:, 0], traj[:, 1], 'x-', rasterized=True)
    # # ax.set_title("Predictions")
    # ax.set_xlim([-2,2])
    # ax.set_ylim([-1,3])
    # fig_path = model.getFigureDir() + "/{:}_{:}_BMP_PREDICTIONS_contour_lines.{:}".format(testing_mode,set_name,FIGTYPE)
    # plt.savefig(fig_path, bbox_inches='tight')
    # plt.close()

    for second_axis_ in [True, False]:
        second_axis_str = "second_axis" if second_axis_ else ""
        if second_axis_:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig, ax = plt.subplots()

        mbp = muellerBrownPotential()
        ax = mbp.plotPotential(save=False,
                               ax=ax,
                               cmap=plt.get_cmap("gist_gray"),
                               title=False,
                               colorbar=second_axis_)
        trajectories_target_plot = np.reshape(trajectories_target, (-1, 2))
        SUBSAMPLE = 1
        X = trajectories_target_plot[::SUBSAMPLE, 0]
        Y = trajectories_target_plot[::SUBSAMPLE, 1]
        createDensityScatterPlot(X,
                                 Y,
                                 ax=ax,
                                 fig=fig,
                                 with_colorbar=True,
                                 cmap=plt.get_cmap("Reds"),
                                 second_axis=second_axis_)
        # ax.set_title("BMP trajectories")
        ax.set_xlim([-2, 2])
        ax.set_ylim([-1, 3])
        plt.title('BMP Data')
        fig_path = model.getFigureDir(
        ) + "/{:}_{:}_BMP_TARGET_potential_density{:}.{:}".format(
            testing_mode, set_name, second_axis_str, FIGTYPE)
        plt.savefig(fig_path, bbox_inches='tight', dpi=100)
        plt.close()

        if second_axis_:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig, ax = plt.subplots()
        mbp = muellerBrownPotential()
        ax = mbp.plotPotential(save=False,
                               ax=ax,
                               cmap=plt.get_cmap("gist_gray"),
                               title=False,
                               colorbar=second_axis_)
        trajectories_pred_plot = np.reshape(trajectories_pred, (-1, 2))
        SUBSAMPLE = 1
        X = trajectories_pred_plot[::SUBSAMPLE, 0]
        Y = trajectories_pred_plot[::SUBSAMPLE, 1]
        createDensityScatterPlot(X,
                                 Y,
                                 ax=ax,
                                 fig=fig,
                                 with_colorbar=True,
                                 cmap=plt.get_cmap("Reds"),
                                 second_axis=second_axis_)
        ax.set_xlim([-2, 2])
        ax.set_ylim([-1, 3])
        plt.title('LED')
        fig_path = model.getFigureDir(
        ) + "/{:}_{:}_BMP_PREDICTIONS_potential_density{:}.{:}".format(
            testing_mode, set_name, second_axis_str, FIGTYPE)
        plt.savefig(fig_path, bbox_inches='tight', dpi=100)
        plt.close()

    # #####################################################################
    # #    Plotting samples trajectories in the Potential space
    # ####################################################################

    # fig, ax = plt.subplots()
    # mbp = muellerBrownPotential()
    # ax = mbp.plotPotential(save=False, ax=ax)
    # for i in range(len(trajectories_target)):
    #     traj = trajectories_target[i]
    #     ax.plot(traj[:, 0], traj[:, 1], 'x', rasterized=True)
    # ax.set_xlim([-2,2])
    # ax.set_ylim([-1,3])
    # fig_path = model.getFigureDir() + "/{:}_{:}_BMP_TARGET_contour.{:}".format(testing_mode,set_name,FIGTYPE)
    # plt.savefig(fig_path, bbox_inches='tight')
    # plt.close()

    # fig, ax = plt.subplots()
    # mbp = muellerBrownPotential()
    # ax = mbp.plotPotential(save=False, ax=ax)
    # for i in range(len(trajectories_pred)):
    #     traj = trajectories_pred[i]
    #     ax.plot(traj[:, 0], traj[:, 1], 'x', rasterized=True)
    # ax.set_xlim([-2,2])
    # ax.set_ylim([-1,3])
    # fig_path = model.getFigureDir() + "/{:}_{:}_BMP_PREDICTIONS_contour.{:}".format(testing_mode,set_name,FIGTYPE)
    # plt.savefig(fig_path, bbox_inches='tight')
    # plt.close()

    # #####################################################################
    # #    Plotting the clustering of the latent space
    # #    and the clustering in the state space along with the Potential
    # ####################################################################

    # if np.shape(latent_states_all)[2]==1:

    #     num_ics = np.min([np.shape(latent_states_all)[0], 3])
    #     for IC in range(num_ics):
    #         print("Plotting the clustering of the latent space tesing mode {:}, IC={:}".format(testing_mode, IC))
    #         latent_states_pred = latent_states_all[IC]
    #         trajectory_pred = trajectories_pred[IC]
    #         trajectory_target = trajectories_target[IC]

    #         n_clusters = 2

    #         # data = np.reshape(latent_states_all, (-1, *np.shape(latent_states_all)[2:]))
    #         data =latent_states_pred
    #         print(np.shape(data))
    #         # Subsampling up to at most 500 points
    #         MAX_POINTS=4000

    #         SUBSAMPLE = 1 if len(data)<MAX_POINTS else int(len(data)/MAX_POINTS)
    #         data = data[::SUBSAMPLE]
    #         trajectory_pred = trajectory_pred[::SUBSAMPLE]
    #         trajectory_target = trajectory_target[::SUBSAMPLE]
    #         latent_states_pred = latent_states_pred[::SUBSAMPLE]

    #         # data = data[:MAX_POINTS]
    #         # trajectory_pred = trajectory_pred[:MAX_POINTS]
    #         # trajectory_target = trajectory_target[:MAX_POINTS]
    #         # latent_states_pred = latent_states_pred[:MAX_POINTS]

    #         print("Spectral clustering...")
    #         clustering = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0).fit(data)
    #         print("Spectral clustering finished!")

    #         cluster_idx = clustering.labels_
    #         colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    #         colors = np.hstack([colors] * 20)

    #         X = np.reshape(np.array(latent_states_pred[:-1]), (-1))
    #         Y = np.reshape(np.array(latent_states_pred[1:]), (-1))
    #         # cluster_idx = cluster_idx[:-1]
    #         cluster_idx = cluster_idx[1:]

    #         cluster_idx = np.array(cluster_idx)
    #         fig, axes = plt.subplots()
    #         LABEL_COLOR_MAP = {
    #         0:'tab:blue',
    #         1:'tab:green',
    #         3:'tab:orange',
    #         4:'tab:olive',
    #         5:'tab:gray',
    #         2:'tab:brown',
    #         6:'tab:red',
    #         }
    #         cluster_idx_colors = [LABEL_COLOR_MAP[l] for l in cluster_idx]
    #         MARKER_MAP = {
    #         0:'x',
    #         1:'o',
    #         2:"*",
    #         3:"P",
    #         4:"v",
    #         5:"D",
    #         6:"s",
    #         }
    #         # plt.scatter(X,Y,s=40,c=cluster_idx_colors, marker=cluster_idx_markers,rasterized=True)
    #         cluster_idx_colors = np.array(cluster_idx_colors)
    #         for num, cluster in enumerate(np.unique(cluster_idx)):
    #             plt.scatter(X[cluster_idx==cluster],Y[cluster_idx==cluster],c=cluster_idx_colors[cluster_idx==cluster], marker=MARKER_MAP[cluster],s=40,rasterized=True)
    #         plt.xlabel(r"$\mathbf{z}_{t}$")
    #         plt.ylabel(r"$\mathbf{z}_{t+1}$")
    #         plt.title("Latent dynamics in {:}".format(set_name))
    #         fig_path = model.getFigureDir() + "/{:}_{:}_BMP_clusters_latent_IC{:}.{:}".format(testing_mode,set_name,IC,FIGTYPE)
    #         plt.savefig(fig_path, bbox_inches='tight')
    #         plt.close()

    #         for num, cluster in enumerate(np.unique(cluster_idx)):
    #             fig, ax = plt.subplots()
    #             mbp = muellerBrownPotential()
    #             ax = mbp.plotPotential(save=False, ax=ax)
    #             X = np.reshape(np.array(trajectory_pred[1:,0]), (-1))
    #             Y = np.reshape(np.array(trajectory_pred[1:,1]), (-1))
    #             plt.scatter(X[cluster_idx==cluster],Y[cluster_idx==cluster],c=cluster_idx_colors[cluster_idx==cluster], marker=MARKER_MAP[cluster],s=40,rasterized=True)
    #             fig_path = model.getFigureDir() + "/{:}_{:}_BMP_clusters_pred_traj_IC{:}_CULSTER_{:}.{:}".format(testing_mode,set_name,IC,num,FIGTYPE)
    #             plt.savefig(fig_path, bbox_inches='tight')
    #             plt.close()

    #         fig, ax = plt.subplots()
    #         mbp = muellerBrownPotential()
    #         ax = mbp.plotPotential(save=False, ax=ax)
    #         X = np.reshape(np.array(trajectory_pred[1:,0]), (-1))
    #         Y = np.reshape(np.array(trajectory_pred[1:,1]), (-1))
    #         for num, cluster in enumerate(np.unique(cluster_idx)):
    #             plt.scatter(X[cluster_idx==cluster],Y[cluster_idx==cluster],c=cluster_idx_colors[cluster_idx==cluster], marker=MARKER_MAP[cluster],s=40,rasterized=True)
    #         # ax.set_title("Predicted trajectories clustered according to latent space")
    #         fig_path = model.getFigureDir() + "/{:}_{:}_BMP_clusters_pred_traj_IC{:}.{:}".format(testing_mode,set_name,IC,FIGTYPE)
    #         plt.savefig(fig_path, bbox_inches='tight')
    #         plt.close()

    #         if "autoencoder" in testing_mode:
    #             fig, ax = plt.subplots()
    #             mbp = muellerBrownPotential()
    #             ax = mbp.plotPotential(save=False, ax=ax)
    #             X = np.reshape(np.array(trajectory_target[:-1,0]), (-1))
    #             Y = np.reshape(np.array(trajectory_target[:-1,1]), (-1))
    #             for num, cluster in enumerate(np.unique(cluster_idx)):
    #                 plt.scatter(X[cluster_idx==cluster],Y[cluster_idx==cluster],c=cluster_idx_colors[cluster_idx==cluster], marker=MARKER_MAP[cluster],s=40,rasterized=True)
    #             # ax.set_title("Target trajectories clustered according to latent space")
    #             fig_path = model.getFigureDir() + "/{:}_{:}_BMP_clusters_target_traj_IC{:}.{:}".format(testing_mode,set_name,IC,FIGTYPE)
    #             plt.savefig(fig_path, bbox_inches='tight')
    #             plt.close()
