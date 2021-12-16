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

FIGTYPE = "pdf"
# FIGTYPE="pdf"

from scipy.interpolate import interpn
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ... import Utils as utils


def scatterDensityLatentDynamicsPlotColor(ax,
                                          x,
                                          y,
                                          colormap,
                                          sort=True,
                                          bins=20):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interpn
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None:
        fig, ax = plt.subplots()
    data, x_e, y_e = np.histogram2d(x, y, bins=bins)
    # z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False )
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

    # mp = ax.scatter( x, y, c=z, cmap=plt.get_cmap("Reds"),rasterized=True)
    # mp = ax.scatter( x, y, c=z, cmap=plt.get_cmap("Reds"), norm=matplotlib.colors.LogNorm(),rasterized=True)
    mp = ax.scatter(x,
                    y,
                    c=z,
                    cmap=plt.get_cmap(colormap),
                    norm=matplotlib.colors.LogNorm(),
                    rasterized=True)
    # plt.colorbar(mp)
    return ax, idx


def plotStateDistributionsSystemAdvectionDiffusion3D(model, results, set_name,
                                                     testing_mode):
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
    #    Plotting the clustering of the latent space
    ####################################################################

    num_ics = np.min([np.shape(latent_states_all)[0], 3])
    for IC in range(num_ics):
        print(
            "Plotting the clustering of the latent space tesing mode {:}, IC={:}"
            .format(testing_mode, IC))
        latent_states_pred = latent_states_all[IC]
        trajectory_pred = trajectories_pred[IC]
        trajectory_target = trajectories_target[IC]

        latent_dim = np.shape(latent_states_pred)[1]
        n_clusters = 6

        # data = np.reshape(latent_states_all, (-1, *np.shape(latent_states_all)[2:]))
        data = latent_states_pred
        print(np.shape(data))
        # Subsampling up to at most 500 points
        MAX_POINTS = 4000

        SUBSAMPLE = 1 if np.shape(data)[0] < MAX_POINTS else int(
            len(data) / MAX_POINTS)
        data = data[::SUBSAMPLE]
        trajectory_pred = trajectory_pred[::SUBSAMPLE]
        trajectory_target = trajectory_target[::SUBSAMPLE]
        latent_states_pred = latent_states_pred[::SUBSAMPLE]

        # data = data[:MAX_POINTS]
        # trajectory_pred = trajectory_pred[:MAX_POINTS]
        # trajectory_target = trajectory_target[:MAX_POINTS]
        # latent_states_pred = latent_states_pred[:MAX_POINTS]

        print(np.shape(trajectory_pred))
        print(np.shape(trajectory_target))
        print(np.shape(latent_states_pred))
        # print(ark)
        # (1000, 400, 3)
        # (1000, 400, 3)
        # (1000, 16)

        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        latent_tsne_results = tsne.fit_transform(latent_states_pred)
        print(np.shape(latent_tsne_results))

        print("Spectral clustering...")
        clustering = SpectralClustering(n_clusters=n_clusters,
                                        assign_labels="discretize",
                                        random_state=0).fit(latent_states_pred)
        print("Spectral clustering finished!")

        cluster_idx = clustering.labels_
        colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
        colors = np.hstack([colors] * 20)

        X = latent_tsne_results[:, 0]
        Y = latent_tsne_results[:, 1]
        # cluster_idx = cluster_idx

        cluster_idx = np.array(cluster_idx)
        LABEL_COLOR_MAP = {
            0: 'tab:blue',
            1: 'tab:green',
            2: 'tab:orange',
            3: 'tab:purple',
            4: 'tab:gray',
            5: 'tab:red',
        }
        LABEL_COLORMAPS = {
            0: 'Blues',
            1: 'Greens',
            2: 'Oranges',
            3: 'Purples',
            4: 'Greys',
            5: 'Reds',
        }
        cluster_idx_colors = [LABEL_COLOR_MAP[l] for l in cluster_idx]
        MARKER_MAP = {
            0: 'x',
            1: 'o',
            2: "*",
            3: "P",
            4: "v",
            5: "D",
            6: "s",
        }
        # plt.scatter(X,Y,s=40,c=cluster_idx_colors, marker=cluster_idx_markers,rasterized=True)
        fig, ax = plt.subplots()
        cluster_idx_colors = np.array(cluster_idx_colors)
        cluster_centers_idx = {}
        for num, cluster in enumerate(np.unique(cluster_idx)):
            _, idx = scatterDensityLatentDynamicsPlotColor(
                ax,
                X[cluster_idx == cluster],
                Y[cluster_idx == cluster],
                colormap=LABEL_COLORMAPS[cluster])
            # plt.scatter(X[cluster_idx==cluster],Y[cluster_idx==cluster],c=cluster_idx_colors[cluster_idx==cluster], marker=MARKER_MAP[cluster],s=40,rasterized=True)
            idx_temp = np.where(np.array(cluster_idx) == cluster)[0]
            cluster_center_idx = idx_temp[idx[-1]]
            cluster_centers_idx.update({cluster: cluster_center_idx})
            plt.scatter(X[cluster_centers_idx[cluster]],
                        Y[cluster_centers_idx[cluster]],
                        c=LABEL_COLOR_MAP[cluster],
                        marker=MARKER_MAP[cluster],
                        s=200,
                        linewidth=4,
                        rasterized=True)
        plt.xlabel("TSNE mode {:}".format(0 + 1))
        plt.ylabel("TSNE mode {:}".format(1 + 1))
        plt.title("Latent dynamics in {:}".format(set_name))
        fig_path = model.getFigureDir(
        ) + "/{:}_{:}_clusters_latent_TSNE_IC{:}.{:}".format(
            testing_mode, set_name, IC, FIGTYPE)
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        plt.close()

        latent_dim_max_comp = 3
        latent_dim_max_comp = np.min([latent_dim_max_comp, latent_dim])
        # print(np.shape(latent_states_pred))
        pca = PCA(n_components=latent_dim_max_comp)
        pca.fit(latent_states_pred)
        latent_states_pca = pca.transform(latent_states_pred)
        print(np.shape(latent_states_pred))
        for idx1 in range(latent_dim_max_comp):
            for idx2 in range(idx1 + 1, latent_dim_max_comp):
                fig, ax = plt.subplots()
                plt.title("Latent dynamics in {:}".format(set_name))
                X = latent_states_pca[:, idx1]
                Y = latent_states_pca[:, idx2]
                cluster_centers_idx = {}
                for num, cluster in enumerate(np.unique(cluster_idx)):
                    _, idx = scatterDensityLatentDynamicsPlotColor(
                        ax,
                        X[cluster_idx == cluster],
                        Y[cluster_idx == cluster],
                        colormap=LABEL_COLORMAPS[cluster])
                    idx_temp = np.where(np.array(cluster_idx) == cluster)[0]
                    cluster_center_idx = idx_temp[idx[-1]]
                    cluster_centers_idx.update({cluster: cluster_center_idx})
                    # plt.scatter(X[cluster_idx==cluster],Y[cluster_idx==cluster],c=cluster_idx_colors[cluster_idx==cluster], marker=MARKER_MAP[cluster],s=40,rasterized=True)
                    plt.scatter(X[cluster_centers_idx[cluster]],
                                Y[cluster_centers_idx[cluster]],
                                c=LABEL_COLOR_MAP[cluster],
                                marker=MARKER_MAP[cluster],
                                s=200,
                                linewidth=4,
                                rasterized=True)
                plt.xlabel("PCA mode {:}".format(idx1 + 1))
                plt.ylabel("PCA mode {:}".format(idx2 + 1))
                fig_path = model.getFigureDir(
                ) + "/{:}_clusters_latent_PCA_{:}_IC{:}_{:}_{:}.{:}".format(
                    testing_mode, set_name, IC, idx1, idx2, FIGTYPE)
                plt.savefig(fig_path, bbox_inches='tight', dpi=300)
                plt.close()

        for cluster_id in range(n_clusters):
            cluster_center = trajectory_pred[cluster_centers_idx[cluster_id]]
            data_plot = cluster_center
            max_target = np.amax(data_plot)
            min_target = np.amin(data_plot)
            fig = plt.figure(figsize=plt.figaspect(0.75) * 1.4)
            axes = fig.gca(projection='3d')
            axes.scatter3D(data_plot[:, 0],
                           data_plot[:, 1],
                           data_plot[:, 2],
                           "o",
                           color=LABEL_COLOR_MAP[cluster_id],
                           alpha=0.9)
            axes.set_xlim([min_target, max_target])
            axes.set_ylim([min_target, max_target])
            axes.set_zlim([min_target, max_target])
            axes.set_xlabel(r"$X_1$", labelpad=20)
            axes.set_ylabel(r"$X_2$", labelpad=20)
            axes.set_zlabel(r"$X_3$", labelpad=20)
            plt.tight_layout()
            fig_path = model.getFigureDir(
            ) + "/{:}_{:}_clusters_STATE_{:}_IC{:}.{:}".format(
                testing_mode, set_name, cluster_id, IC, FIGTYPE)
            plt.savefig(fig_path, bbox_inches='tight', dpi=300)
            plt.close()


def plotStateDistributionsSystemAdvectionDiffusion(model, results, set_name,
                                                   testing_mode):
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
    #    Plotting the clustering of the latent space
    ####################################################################

    num_ics = np.min([np.shape(latent_states_all)[0], 3])
    for IC in range(num_ics):
        print(
            "Plotting the clustering of the latent space tesing mode {:}, IC={:}"
            .format(testing_mode, IC))
        latent_states_pred = latent_states_all[IC]
        trajectory_pred = trajectories_pred[IC]
        trajectory_target = trajectories_target[IC]

        latent_dim = np.shape(latent_states_pred)[1]

        n_clusters = 2

        # data = np.reshape(latent_states_all, (-1, *np.shape(latent_states_all)[2:]))
        data = latent_states_pred
        print(np.shape(data))
        # Subsampling up to at most 500 points
        MAX_POINTS = 4000

        SUBSAMPLE = 1 if np.shape(data)[0] < MAX_POINTS else int(
            len(data) / MAX_POINTS)
        data = data[::SUBSAMPLE]
        trajectory_pred = trajectory_pred[::SUBSAMPLE]
        trajectory_target = trajectory_target[::SUBSAMPLE]
        latent_states_pred = latent_states_pred[::SUBSAMPLE]
        print(np.shape(trajectory_pred))
        print(np.shape(trajectory_target))
        print(np.shape(latent_states_pred))

        # print(ark)
        # (1000, 400, 3)
        # (1000, 400, 3)
        # (1000, 16)

        print("Spectral clustering...")
        clustering = SpectralClustering(n_clusters=n_clusters,
                                        assign_labels="discretize",
                                        random_state=0).fit(latent_states_pred)
        print("Spectral clustering finished!")

        cluster_idx = clustering.labels_
        colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
        colors = np.hstack([colors] * 20)

        X = np.array(latent_states_pred[:-1])
        Y = np.array(latent_states_pred[1:])
        cluster_idx = np.array(cluster_idx)
        cluster_idx = cluster_idx[:-1]
        cluster_idx = np.array(cluster_idx)
        X = np.reshape(X, (-1))
        Y = np.reshape(Y, (-1))

        print(np.shape(X))
        print(np.shape(Y))
        print(np.shape(cluster_idx))

        LABEL_COLOR_MAP = {
            0: 'tab:blue',
            1: 'tab:green',
            2: 'tab:orange',
            3: 'tab:purple',
            4: 'tab:gray',
            5: 'tab:red',
        }
        LABEL_COLORMAPS = {
            0: 'Blues',
            1: 'Greens',
            2: 'Oranges',
            3: 'Purples',
            4: 'Greys',
            5: 'Reds',
        }
        cluster_idx_colors = [LABEL_COLOR_MAP[l] for l in cluster_idx]
        MARKER_MAP = {
            0: 'x',
            1: 'o',
            2: "*",
            3: "P",
            4: "v",
            5: "D",
            6: "s",
        }

        fig, ax = plt.subplots()
        plt.title("Latent dynamics in {:}".format(set_name))
        cluster_centers_idx = {}
        for num, cluster in enumerate(np.unique(cluster_idx)):
            print(np.shape(X))
            print(np.shape(Y))
            X_plot = np.array(X[cluster_idx == cluster])
            Y_plot = np.array(Y[cluster_idx == cluster])
            _, idx = scatterDensityLatentDynamicsPlotColor(
                ax, X_plot, Y_plot, colormap=LABEL_COLORMAPS[cluster])
            idx_temp = np.where(np.array(cluster_idx) == cluster)[0]
            cluster_center_idx = idx_temp[idx[-1]]
            cluster_centers_idx.update({cluster: cluster_center_idx})
            plt.scatter(X[cluster_centers_idx[cluster]],
                        Y[cluster_centers_idx[cluster]],
                        c=LABEL_COLOR_MAP[cluster],
                        marker=MARKER_MAP[cluster],
                        s=200,
                        linewidth=4,
                        rasterized=True)
        plt.xlabel(r"$\mathbf{z}_{t}$")
        plt.ylabel(r"$\mathbf{z}_{t+1}$")
        fig_path = model.getFigureDir(
        ) + "/{:}_clusters_latent_{:}_IC{:}.{:}".format(
            testing_mode, set_name, IC, FIGTYPE)
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        plt.close()
