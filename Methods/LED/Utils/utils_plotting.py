#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
import socket
import os
import subprocess

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
    'tab:blue',
    'tab:red',
    'tab:green',
    'tab:brown',
    'tab:orange',
    'tab:cyan',
    'tab:olive',
    'tab:pink',
    'tab:gray',
    'tab:purple',
]

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

from scipy.stats.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scipy.interpolate import interpn

from . import utils_processing
from . import utils_statistics
from . import utils_data
from . import utils_time

FIGTYPE = "pdf"
# FIGTYPE="pdf"


def plotSchedule(model, ifp_train_vec, ifp_val_vec):
    color_labels = [
        'blue',
        'green',
        'brown',
        'darkcyan',
        'purple',
        'orange',
        'darkorange',
        'cadetblue',
        'honeydew',
    ]

    fig_path = model.getFigureDir() + "/schedule." + FIGTYPE
    fig, ax = plt.subplots(figsize=(20, 10))

    # Removing the initial epoch:
    ifp_train_vec = ifp_train_vec[1:]
    ifp_val_vec = ifp_val_vec[1:]

    plt.plot(
        np.arange(len(ifp_train_vec)),
        ifp_train_vec,
        "o-",
        color=color_dict['blue'],
        label="train",
        linewidth=3,
    )
    plt.plot(
        np.arange(len(ifp_val_vec)),
        ifp_val_vec,
        "x-",
        color=color_dict['red'],
        label="val",
        linewidth=3,
    )
    ax.set_ylabel(r"Iterative forecasting propability")
    ax.set_xlabel(r"Epoch")
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


def plotAllLosses(model,
                  losses_train,
                  time_train,
                  losses_val,
                  time_val,
                  min_val_error,
                  name_str=""):
    loss_labels = model.losses_labels
    idx1 = np.nonzero(losses_train[0])[0]
    idx2 = np.nonzero(losses_train[-1])[0]
    idx = np.union1d(idx1, idx2)
    losses_val = np.array(losses_val)
    losses_train = np.array(losses_train)
    color_labels = [
        'blue',
        'green',
        'brown',
        'darkcyan',
        'purple',
        'orange',
        'darkorange',
        'cadetblue',
        'honeydew',
    ]
    min_val_epoch = np.argmin(
        np.abs(np.array(losses_val[:, 0]) - min_val_error))
    min_val_time = time_val[min_val_epoch]

    losses_train = losses_train[:, idx]
    losses_val = losses_val[:, idx]
    loss_labels = [loss_labels[i] for i in idx]
    color_labels = [color_labels[i] for i in idx]

    time_train = np.array(time_train)
    time_val = np.array(time_val)

    if np.all(np.array(losses_train) > 0.0) and np.all(
            np.array(losses_val) > 0.0):
        losses_train = np.log10(losses_train)
        losses_val = np.log10(losses_val)
        min_val_error_log = np.log10(min_val_error)
        if len(time_train) > 1:
            for time_str in ["", "_time"]:
                fig_path = model.getFigureDir(
                ) + "/losses_all_log" + time_str + name_str + "." + FIGTYPE
                fig, ax = plt.subplots(figsize=(20, 10))
                title = "MIN LOSS-VAL={:.4f}".format(min_val_error)
                plt.title(title)
                max_i = np.min([np.shape(losses_train)[1], len(loss_labels)])
                for i in range(max_i):
                    if time_str != "_time":
                        x_axis_train = np.arange(
                            np.shape(losses_train[:, i])[0])
                        x_axis_val = np.arange(np.shape(losses_val[:, i])[0])
                        min_val_axis = min_val_epoch
                        ax.set_xlabel(r"Epoch")
                    else:
                        dt = time_train[1] - time_train[0]
                        x_axis_train = time_train + i * dt
                        x_axis_val = time_val + i * dt
                        min_val_axis = min_val_time
                        ax.set_xlabel(r"Time")
                    plt.plot(x_axis_train,
                             losses_train[:, i],
                             color=color_dict[color_labels[i]],
                             label=loss_labels[i] + " Train")
                    plt.plot(x_axis_val,
                             losses_val[:, i],
                             color=color_dict[color_labels[i]],
                             label=loss_labels[i] + " Val",
                             linestyle="--")
                plt.plot(min_val_axis,
                         min_val_error_log,
                         "o",
                         color=color_dict['red'],
                         label="optimal")
                ax.set_ylabel(r"Log${}_{10}$(Loss)")
                plt.legend(loc="upper left",
                           bbox_to_anchor=(1.05, 1),
                           borderaxespad=0.)
                plt.tight_layout()
                plt.savefig(fig_path, dpi=300)
                plt.close()
    else:
        if len(time_train) > 1:
            for time_str in ["", "_time"]:
                fig_path = model.getFigureDir(
                ) + "/losses_all" + time_str + name_str + "." + FIGTYPE
                fig, ax = plt.subplots(figsize=(20, 10))
                title = "MIN LOSS-VAL={:.4f}".format(min_val_error)
                plt.title(title)
                max_i = np.min([np.shape(losses_train)[1], len(loss_labels)])
                for i in range(max_i):
                    if time_str != "_time":
                        x_axis_train = np.arange(
                            np.shape(losses_train[:, i])[0])
                        x_axis_val = np.arange(np.shape(losses_val[:, i])[0])
                        min_val_axis = min_val_epoch
                        ax.set_xlabel(r"Epoch")
                    else:
                        dt = time_train[1] - time_train[0]
                        x_axis_train = time_train + i * dt
                        x_axis_val = time_val + i * dt
                        min_val_axis = min_val_time
                        ax.set_xlabel(r"Time")
                    plt.plot(x_axis_train,
                             losses_train[:, i],
                             color=color_dict[color_labels[i]],
                             label=loss_labels[i] + " Train")
                    plt.plot(x_axis_val,
                             losses_val[:, i],
                             color=color_dict[color_labels[i]],
                             label=loss_labels[i] + " Val",
                             linestyle="--")
                plt.plot(min_val_axis,
                         min_val_error,
                         "o",
                         color=color_dict['red'],
                         label="optimal")
                ax.set_ylabel(r"Loss")
                plt.legend(loc="upper left",
                           bbox_to_anchor=(1.05, 1),
                           borderaxespad=0.)
                plt.tight_layout()
                plt.savefig(fig_path, dpi=300)
                plt.close()


def plotTrainingLosses(model,
                       loss_train,
                       loss_val,
                       min_val_error,
                       name_str=""):
    if (len(loss_train) != 0) and (len(loss_val) != 0):
        min_val_epoch = np.argmin(np.abs(np.array(loss_val) - min_val_error))
        fig_path = model.getFigureDir(
        ) + "/loss_total" + name_str + "." + FIGTYPE
        fig, ax = plt.subplots()
        plt.title("Validation error {:.10f}".format(min_val_error))
        plt.plot(np.arange(np.shape(loss_train)[0]),
                 loss_train,
                 color=color_dict['green'],
                 label="Train RMSE")
        plt.plot(np.arange(np.shape(loss_val)[0]),
                 loss_val,
                 color=color_dict['blue'],
                 label="Validation RMSE")
        plt.plot(min_val_epoch,
                 min_val_error,
                 "o",
                 color=color_dict['red'],
                 label="optimal")
        ax.set_xlabel(r"Epoch")
        ax.set_ylabel(r"Loss")
        plt.legend(loc="upper left",
                   bbox_to_anchor=(1.05, 1),
                   borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()

        loss_train = np.array(loss_train)
        loss_val = np.array(loss_val)
        if (np.all(loss_train[~np.isnan(loss_train)] > 0.0)
                and np.all(loss_val[~np.isnan(loss_val)] > 0.0)):
            fig_path = model.getFigureDir(
            ) + "/loss_total_log" + name_str + "." + FIGTYPE
            fig, ax = plt.subplots()
            plt.title("Validation error {:.10f}".format(min_val_error))
            plt.plot(np.arange(np.shape(loss_train)[0]),
                     np.log10(loss_train),
                     color=color_dict['green'],
                     label="Train RMSE")
            plt.plot(np.arange(np.shape(loss_val)[0]),
                     np.log10(loss_val),
                     color=color_dict['blue'],
                     label="Validation RMSE")
            plt.plot(min_val_epoch,
                     np.log10(min_val_error),
                     "o",
                     color=color_dict['red'],
                     label="optimal")
            ax.set_xlabel(r"Epoch")
            ax.set_ylabel(r"Log${}_{10}$(Loss)")
            plt.legend(loc="upper left",
                       bbox_to_anchor=(1.05, 1),
                       borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300)
            plt.close()
    else:
        print("## Empty losses. Not printing... ##")


def plotLatentDynamicsOfSingleTrajectory(model,
                                         set_name,
                                         latent_states,
                                         ic_idx,
                                         testing_mode,
                                         dt=1.0):
    # if np.shape(latent_states)[1] >= 2:
    #     latent_dim = np.shape(latent_states)[1]
    #     latent_dim_max_comp = 3
    #     latent_dim_max_comp = np.min([latent_dim_max_comp, latent_dim])

    #     for idx1 in range(latent_dim_max_comp):
    #         for idx2 in range(idx1+1, latent_dim_max_comp):
    #             fig, ax = plt.subplots()
    #             plt.title("Latent dynamics in {:}".format(set_name))
    #             X = latent_states[:, idx1]
    #             Y = latent_states[:, idx2]
    #             # arrowplot(ax, X, Y, nArrs=100)
    #             scatterDensityLatentDynamicsPlot(X, Y, ax=ax)
    #             plt.xlabel("State {:}".format(idx1+1))
    #             plt.ylabel("State {:}".format(idx2+1))
    #             plt.tight_layout()
    #             fig_path = model.getFigureDir() + "/{:}_latent_dynamics_{:}_{:}_{:}_{:}.{:}".format(testing_mode, set_name, ic_idx, idx1, idx2, FIGTYPE)
    #             plt.savefig(fig_path, dpi=300)
    #             plt.close()

    #     tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    #     latent_tsne_results = tsne.fit_transform(latent_states)
    #     print(np.shape(latent_tsne_results))
    #     fig, ax = plt.subplots()
    #     plt.title("Latent dynamics in {:}".format(set_name))
    #     X = latent_tsne_results[:, 0]
    #     Y = latent_tsne_results[:, 1]
    #     # arrowplot(ax, X, Y, nArrs=100)
    #     scatterDensityLatentDynamicsPlot(X, Y, ax=ax)
    #     plt.xlabel("TSNE mode {:}".format(0+1))
    #     plt.ylabel("TSNE mode {:}".format(1+1))
    #     plt.tight_layout()
    #     fig_path = model.getFigureDir() + "/{:}_latent_dynamics_TSNE_{:}_{:}_{:}_{:}.{:}".format(testing_mode, set_name, ic_idx, 0, 1, FIGTYPE)
    #     plt.savefig(fig_path, dpi=300)
    #     plt.close()

    #     # print(np.shape(latent_states_plot))
    #     pca = PCA(n_components=latent_dim_max_comp)
    #     pca.fit(latent_states[:5000])
    #     latent_states_pca = pca.transform(latent_states)
    #     print(np.shape(latent_states))
    #     for idx1 in range(latent_dim_max_comp):
    #         for idx2 in range(idx1+1, latent_dim_max_comp):
    #             fig, ax = plt.subplots()
    #             plt.title("Latent dynamics in {:}".format(set_name))
    #             X = latent_states_pca[:, idx1]
    #             Y = latent_states_pca[:, idx2]
    #             # arrowplot(ax, X, Y, nArrs=100)
    #             scatterDensityLatentDynamicsPlot(X, Y, ax=ax)
    #             plt.xlabel("PCA mode {:}".format(idx1+1))
    #             plt.ylabel("PCA mode {:}".format(idx2+1))
    #             plt.tight_layout()
    #             fig_path = model.getFigureDir() + "/{:}_latent_dynamics_PCA_{:}_{:}_{:}_{:}.{:}".format(testing_mode, set_name, ic_idx, idx1, idx2, FIGTYPE)
    #             plt.savefig(fig_path, dpi=300)
    #             plt.close()
    # else:

    fig, ax = plt.subplots()
    plt.title("Latent dynamics in {:}".format(set_name))
    latent_states_plot_x = np.reshape(np.array(latent_states[:-1]), (-1))
    latent_states_plot_y = np.reshape(np.array(latent_states[1:]), (-1))
    scatterDensityLatentDynamicsPlot(latent_states_plot_x,
                                     latent_states_plot_y,
                                     ax=ax)
    plt.xlabel(r"$\mathbf{z}_{t}$")
    plt.ylabel(r"$\mathbf{z}_{t+1}$")
    plt.tight_layout()
    fig_path = model.getFigureDir(
    ) + "/{:}_latent_dynamics_{:}_{:}.{:}".format(testing_mode, set_name,
                                                  ic_idx, FIGTYPE)
    plt.savefig(fig_path, dpi=300)
    plt.close()

    fig, ax = plt.subplots()
    plt.title("Latent dynamics in {:}".format(set_name))
    plt.plot(np.arange(np.shape(latent_states)[0]) * dt, latent_states)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\mathbf{z}_{t}$")
    plt.tight_layout()
    fig_path = model.getFigureDir(
    ) + "/{:}_latent_dynamics_time_{:}_{:}.{:}".format(testing_mode, set_name,
                                                       ic_idx, FIGTYPE)
    plt.savefig(fig_path, dpi=300)
    plt.close()



def createIterativePredictionPlots(model, target, prediction, dt, ic_idx, set_name, \
    testing_mode="", latent_states=None, hist_data=None, wasserstein_distance_data=None, \
    warm_up=None, target_augment=None, prediction_augment=None):
    print("# createIterativePredictionPlots() #")
    # if error is not None:
    #      fig_path = model.getFigureDir() + "/{:}_{:}_{:}_error.{:}".format(testing_mode, set_name, ic_idx, FIGTYPE)
    #      plt.plot(error, label='error')
    #      plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    #      plt.tight_layout()
    #      plt.savefig(fig_path, dpi=300)
    #      plt.close()

    #      fig_path = model.getFigureDir() + "/{:}_{:}_{:}_log_error.{:}".format(testing_mode, set_name, ic_idx, FIGTYPE)
    #      plt.plot(np.log10(np.arange(np.shape(error)[0])), np.log10(error), label='Log${}_{10}$(Loss)')
    #      plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    #      plt.tight_layout()
    #      plt.savefig(fig_path, dpi=300)
    #      plt.close()

    if len(np.shape(prediction)) == 2 or len(np.shape(prediction)) == 3:

        if ((target_augment is not None) and (prediction_augment is not None)):
            prediction_augment_plot = prediction_augment[:, 0] if len(
                np.shape(prediction_augment)
            ) == 2 else prediction_augment[:, 0, 0] if len(
                np.shape(prediction_augment)) == 3 else None
            target_augment_plot = target_augment[:, 0] if len(
                np.shape(
                    target_augment)) == 2 else target_augment[:, 0, 0] if len(
                        np.shape(target_augment)) == 3 else None

            fig_path = model.getFigureDir(
            ) + "/{:}_augmented_{:}_{:}.{:}".format(testing_mode, set_name,
                                                    ic_idx, FIGTYPE)
            plt.plot(np.arange(np.shape(prediction_augment_plot)[0]),
                     prediction_augment_plot,
                     'b',
                     linewidth=2.0,
                     label='output')
            plt.plot(np.arange(np.shape(target_augment_plot)[0]),
                     target_augment_plot,
                     'r',
                     linewidth=2.0,
                     label='target')
            plt.plot(np.ones((100, 1)) * warm_up,
                     np.linspace(np.min(target_augment_plot),
                                 np.max(target_augment_plot), 100),
                     'g--',
                     linewidth=2.0,
                     label='warm-up')
            plt.legend(loc="upper left",
                       bbox_to_anchor=(1.05, 1),
                       borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=100)
            plt.close()

            prediction_plot = prediction[:, 0] if len(
                np.shape(prediction)) == 2 else prediction[:, 0, 0] if len(
                    np.shape(prediction)) == 3 else None
            target_plot = target[:, 0] if len(
                np.shape(target)) == 2 else target[:, 0, 0] if len(
                    np.shape(target)) == 3 else None

            fig_path = model.getFigureDir() + "/{:}_{:}_{:}.{:}".format(
                testing_mode, set_name, ic_idx, FIGTYPE)
            plt.plot(prediction_plot, 'r--', label='prediction')
            plt.plot(target_plot, 'g--', label='target')
            plt.legend(loc="upper left",
                       bbox_to_anchor=(1.05, 1),
                       borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=100)
            plt.close()

        if model.input_dim >= 2:
            plotTestingContours(
                model,
                target,
                prediction,
                dt,
                ic_idx,
                set_name,
                latent_states=latent_states,
                testing_mode=testing_mode,
                hist_data=hist_data,
                wasserstein_distance_data=wasserstein_distance_data,
                with_multiscale_bar=isMultiscale(testing_mode))

    elif len(np.shape(prediction)) == 4:
        createIterativePredictionPlotsForImageData(model, target, prediction,
                                                   dt, ic_idx, set_name,
                                                   testing_mode)


def isMultiscale(testing_mode):
    if "multiscale" in testing_mode:
        return True
    else:
        return False


def createIterativePredictionPlotsForImageData(model, target, prediction, dt,
                                               ic_idx, set_name, testing_mode):
    # IMAGE DATA
    assert (len(np.shape(prediction)) == 4)
    T_ = np.shape(prediction)[0]

    # vmin = np.min([target.min(), prediction.min()])
    # vmax = np.min([target.max(), prediction.max()])
    vmin = target.min()
    vmax = target.max()
    RGB_CHANNEL = 0
    T_MAX = np.min([5, T_])
    fig_path = model.getFigureDir() + "/{:}_{:}_{:}.{:}".format(
        testing_mode, set_name, ic_idx, FIGTYPE)
    fig, axes = plt.subplots(nrows=3,
                             ncols=T_MAX,
                             figsize=(16, 8),
                             sharey=True)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for t in range(T_MAX):
        abserror = np.abs(target[t, RGB_CHANNEL] - prediction[t, RGB_CHANNEL])
        mp1 = axes[0, t].imshow(target[t, RGB_CHANNEL],
                                vmin=vmin,
                                vmax=vmax,
                                cmap=plt.get_cmap("seismic"),
                                aspect=1.0,
                                interpolation='lanczos')
        mp2 = axes[1, t].imshow(prediction[t, RGB_CHANNEL],
                                vmin=vmin,
                                vmax=vmax,
                                cmap=plt.get_cmap("seismic"),
                                aspect=1.0,
                                interpolation='lanczos')
        mp3 = axes[2, t].imshow(abserror,
                                vmin=0,
                                vmax=vmax,
                                cmap=plt.get_cmap("Reds"),
                                aspect=1.0,
                                interpolation='lanczos')
        axes[0, t].set_title("Target")
        axes[1, t].set_title("Prediction")
        axes[2, t].set_title("Absolute error")
    cbar_ax = fig.add_axes([0.93, 0.68, 0.015,
                            0.2])  #[left, bottom, width, height]
    cbar = fig.colorbar(mp1, cax=cbar_ax, format='%.0f')
    cbar_ax = fig.add_axes([0.93, 0.4, 0.015,
                            0.2])  #[left, bottom, width, height]
    cbar = fig.colorbar(mp2, cax=cbar_ax, format='%.0f')
    cbar_ax = fig.add_axes([0.93, 0.11, 0.015,
                            0.2])  #[left, bottom, width, height]
    cbar = fig.colorbar(mp3, cax=cbar_ax, format='%.0f')
    plt.savefig(fig_path, dpi=100)
    plt.close()

    if model.params["make_videos"]:

        video_folder = "{:}_image_data_video_{:}_IC{:}".format(
            testing_mode, set_name, ic_idx)
        n_frames_max, frame_path_python, frame_path_bash, video_path = makeVideoPaths(
            model, video_folder)

        n_frames = np.min([n_frames_max, T_])
        for t in range(n_frames):
            fig_path = frame_path_python.format(t)
            fig, axes = plt.subplots(figsize=(13, 3), ncols=3)
            mp1 = axes[0].imshow(target[t, RGB_CHANNEL],
                                 vmin=vmin,
                                 vmax=vmax,
                                 cmap=plt.get_cmap("seismic"),
                                 aspect=1.0,
                                 interpolation='lanczos')
            mp2 = axes[1].imshow(prediction[t, RGB_CHANNEL],
                                 vmin=vmin,
                                 vmax=vmax,
                                 cmap=plt.get_cmap("seismic"),
                                 aspect=1.0,
                                 interpolation='lanczos')
            mp3 = axes[2].imshow(np.abs(target[t, RGB_CHANNEL] -
                                        prediction[t, RGB_CHANNEL]),
                                 vmin=0,
                                 vmax=vmax,
                                 cmap=plt.get_cmap("Reds"),
                                 aspect=1.0,
                                 interpolation='lanczos')
            axes[0].set_title("Target")
            axes[1].set_title("Prediction")
            axes[2].set_title("Absolute error")
            fig.colorbar(mp1, ax=axes[0])
            fig.colorbar(mp2, ax=axes[1])
            fig.colorbar(mp3, ax=axes[2])

            fig.subplots_adjust(hspace=0.4, wspace=0.4)
            plt.savefig(fig_path, dpi=100)
            plt.close()

        makeVideo(model, video_path, frame_path_bash, n_frames_max)


def makeVideoPaths(model, video_folder):
    n_frames_max = 10
    # n_frames_max = 1000
    video_base_dir = model.getFigureDir()
    video_path = "{:}/{:}".format(video_base_dir, video_folder)
    os.makedirs(video_path + "/", exist_ok=True)
    frame_path_python = video_path + "/frame_N{:04d}.png"
    frame_path_bash = video_path + "/frame_N%04d.png"
    return n_frames_max, frame_path_python, frame_path_bash, video_path


def makeVideo(model, video_path, frame_path_bash, n_frames_max):
    # MAKING VIDEO
    command_str = "ffmpeg -y -r 5 -f image2 -s 1342x830 -i {:} -vcodec libx264 -crf 1  -pix_fmt yuv420p -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' {:}.mp4".format(
        frame_path_bash, video_path)
    print("COMMAND TO MAKE VIDEO:")
    print(command_str)

    # Write video command to a .sh file in the Figures folder
    print("Writing the command to file...")
    with open(model.getFigureDir() + "/video_commands_abs.sh", "a+") as file:
        file.write(command_str)
        file.write("\n")

    temp = frame_path_bash.split("/")
    temp = temp[-2:]
    frame_path_bash_rel = "./" + temp[0] + "/" + temp[1]

    temp = video_path.split("/")
    video_path_rel = "./" + temp[-1]
    command_str_rel = "ffmpeg -y -r 5 -f image2 -s 1342x830 -i {:} -vcodec libx264 -crf 1  -pix_fmt yuv420p -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' {:}.mp4".format(
        frame_path_bash_rel, video_path_rel)
    print("COMMAND TO MAKE VIDEO (RELATIVE):")
    print(command_str_rel)
    print("Writing the command (with relative paths) to file...")
    with open(model.getFigureDir() + "/video_commands_rel.sh", "a+") as file:
        file.write(command_str_rel)
        file.write("\n")

    # os.system(command_str)
    # ffmpeg -y -r 5 -f image2 -s 1342x830 -i ./Iterative_Prediction_Video_TEST_IC108479/frame_N%04d.{:} -vcodec libx264 -crf 1  -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" video_TEST_IC108479.mp4

    if not CLUSTER:
        return_value = subprocess.call([
            'ffmpeg',
            '-y',
            '-r',
            '20',
            '-f',
            'image2',
            '-s',
            '1342x830',
            '-i',
            '{:}'.format(frame_path_bash),
            '-vcodec',
            'libx264',
            '-crf',
            '1',
            '-pix_fmt',
            'yuv420p',
            '-vf',
            'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-frames:v',
            '{:}'.format(n_frames_max),
            '-shortest',
            "{:}.mp4".format(video_path),
        ])
        if return_value:
            print("Failure: FFMPEG probably not installed.")
        else:
            print("Sucess: Video ready!")


def plotTestingContourEvolution(model,
                                target,
                                output,
                                dt,
                                ic_idx,
                                set_name,
                                latent_states=None,
                                testing_mode="",
                                with_multiscale_bar=False):
    print("# plotTestingContourEvolution() #")
    error = np.abs(target - output)
    vmin = target.min()
    vmax = target.max()
    vmin_error = 0.0
    vmax_error = target.max()
    if latent_states is None:
        fig, axes = plt.subplots(nrows=1,
                                 ncols=4,
                                 figsize=(14, 6),
                                 sharey=True)
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        axes[0].set_ylabel(r"Time $t$")
        contours_vec = []
        mp = createContour_(fig,
                            axes[0],
                            target,
                            "Target",
                            vmin,
                            vmax,
                            plt.get_cmap("seismic"),
                            dt,
                            xlabel="State")
        contours_vec.append(mp)
        mp = createContour_(fig,
                            axes[1],
                            output,
                            "Output",
                            vmin,
                            vmax,
                            plt.get_cmap("seismic"),
                            dt,
                            xlabel="State")
        contours_vec.append(mp)
        mp = createContour_(fig,
                            axes[2],
                            error,
                            "Error",
                            vmin_error,
                            vmax_error,
                            plt.get_cmap("Reds"),
                            dt,
                            xlabel="State")
        contours_vec.append(mp)
        corr = [pearsonr(target[i], output[i])[0] for i in range(len(target))]
        time_vector = np.arange(target.shape[0]) * dt
        axes[3].plot(corr, time_vector)
        axes[3].set_title("Correlation")
        axes[3].set_xlabel(r"Correlation")
        axes[3].set_xlim((-1, 1))
        axes[3].set_ylim((time_vector.min(), time_vector.max()))
        for contours in contours_vec:
            for pathcoll in contours.collections:
                pathcoll.set_rasterized(True)
        fig_path = model.getFigureDir() + "/{:}_{:}_{:}_contour.{:}".format(
            testing_mode, set_name, ic_idx, FIGTYPE)
        plt.savefig(fig_path, dpi=100)
        plt.close()
    elif len(np.shape(latent_states)) == 2:
        # Plotting the contour plot
        ncols = 6 if with_multiscale_bar else 5
        fig, axes = plt.subplots(nrows=1,
                                 ncols=ncols,
                                 figsize=(3.6 * ncols, 6),
                                 sharey=True)
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        axes[0].set_ylabel(r"Time $t$")
        contours_vec = []
        mp = createContour_(fig,
                            axes[0],
                            target,
                            "Target",
                            vmin,
                            vmax,
                            plt.get_cmap("seismic"),
                            dt,
                            xlabel="State")
        contours_vec.append(mp)
        time_vector = np.arange(target.shape[0]) * dt
        vmin_latent = np.min(latent_states)
        vmax_latent = np.max(latent_states)
        if np.shape(latent_states)[1] > 1:
            mp = createContour_(fig,
                                axes[1],
                                latent_states,
                                None,
                                vmin_latent,
                                vmax_latent,
                                plt.get_cmap("seismic"),
                                dt,
                                xlabel="Latent state")
            contours_vec.append(mp)
        else:
            axes[1].plot(latent_states[:, 0], time_vector)
            axes[1].set_xlabel(r"Latent state")
            axes[1].set_ylim((time_vector.min(), time_vector.max()))

        mp = createContour_(fig,
                            axes[2],
                            output,
                            "Output",
                            vmin,
                            vmax,
                            plt.get_cmap("seismic"),
                            dt,
                            xlabel="State")
        contours_vec.append(mp)
        mp = createContour_(fig,
                            axes[3],
                            error,
                            "Error",
                            vmin_error,
                            vmax_error,
                            plt.get_cmap("Reds"),
                            dt,
                            xlabel="State")
        contours_vec.append(mp)

        corr = [pearsonr(target[i], output[i])[0] for i in range(len(target))]
        axes[4].plot(corr, time_vector)
        axes[4].set_xlabel(r"Correlation")
        axes[4].set_xlim((-1, 1))
        axes[4].set_ylim((time_vector.min(), time_vector.max()))
        axes[4].set_title("Correlation")
        if with_multiscale_bar:
            # Add a bar plot demonstrating where it is with_multiscale_bar and where not
            axes[5].set_title("Multiscale?")
            multiscale_rounds, macro_steps_per_round, micro_steps_per_round = model.getMultiscaleParams(
                testing_mode)

            start_idx = 0
            for round_ in range(multiscale_rounds):
                end_idx = start_idx + macro_steps_per_round[round_]
                start_t = start_idx * dt
                end_t = end_idx * dt
                axes[5].axhspan(start_t,
                                end_t,
                                color="orange",
                                alpha=0.7,
                                label=None if round_ == 0 else None)

                start_idx = end_idx
                if round_ < len(micro_steps_per_round):
                    end_idx = start_idx + micro_steps_per_round[round_]
                    start_t = start_idx * dt
                    end_t = end_idx * dt
                    axes[5].axhspan(start_t,
                                    end_t,
                                    color="green",
                                    alpha=0.7,
                                    label=None if round_ == 0 else None)
                    start_idx = end_idx

                plt.legend(loc="upper left",
                           bbox_to_anchor=(1.05, 1),
                           borderaxespad=0.)
                plt.axis('off')
            axes[5].set_ylim((time_vector.min(), time_vector.max()))

        for contours in contours_vec:
            for pathcoll in contours.collections:
                pathcoll.set_rasterized(True)

        plt.tight_layout()
        fig_path = model.getFigureDir() + "/{:}_{:}_{:}_contour.{:}".format(
            testing_mode, set_name, ic_idx, FIGTYPE)
        plt.savefig(fig_path, dpi=100)
        plt.close()


def plotTestingContourDensity(model,
                              target,
                              output,
                              dt,
                              ic_idx,
                              set_name,
                              latent_states=None,
                              testing_mode="",
                              with_multiscale_bar=False,
                              quantity="Positions",
                              xlabel="Particles",
                              hist_data=None,
                              wasserstein_distance_data=None):
    # Creating two density plots, one with all evolutions (latent and state space) and one with the evoltions only on propagation
    error = np.abs(target - output)
    # vmin = np.array([target.min(), output.min()]).min()
    # vmax = np.array([target.max(), output.max()]).max()
    vmin = target.min()
    vmax = target.max()
    vmin_error = 0.0
    vmax_error = target.max()

    print("TARGET VMIN: {:} \nVMAX: {:} \n".format(vmin, vmax))
    # Plotting the contour plot
    vmin_o = output.min()
    vmax_o = output.max()
    print("OUTPUT VMIN: {:} \nVMAX: {:} \n".format(vmin_o, vmax_o))
    print(np.shape(output))
    print(np.shape(target))

    # Plotting the contour plot
    if model.Dx == 1:
        ncols = 6 if with_multiscale_bar else 5
        ncols = ncols if latent_states is not None else ncols - 1
        nrows = 2
    elif model.Dx == 2 or model.Dx == 3:
        ncols = 6 if with_multiscale_bar else 5
        ncols = ncols if latent_states is not None else ncols - 1
        nrows = 1
    else:
        raise ValueError("Invalid Dx")
    # REMOVING THE L1-NORM
    if model.Dx == 2 or model.Dx == 3: ncols = ncols - 1

    # real_dist, real_part, latent, pred_part, pred_dist, error_dist, multiscale
    fig, axes_temp = plt.subplots(nrows=nrows,
                                  ncols=ncols,
                                  figsize=(4.2 * ncols, 6 * nrows),
                                  sharey=True)
    axes = np.reshape(axes_temp, (-1))

    # fig.subplots_adjust(hspace=0.2, wspace = 0.2)
    axes[0].set_ylabel(r"Time $t$")

    N = np.shape(target)[1]
    max_target = np.amax(target)
    min_target = np.amin(target)
    bounds = [min_target, max_target]
    LL = bounds[1] - bounds[0]
    nbins = utils_statistics.getNumberOfBins(N, dim=model.Dx, rule="rice")

    # error_density_total =  np.array([temp[0] for temp in hist_data])
    # error_density =  np.array([temp[1] for temp in hist_data])
    # density_target =  np.array([temp[2] for temp in hist_data])
    # density_output =  np.array([temp[3] for temp in hist_data])
    # bin_centers =  np.array([temp[4] for temp in hist_data])[0]

    error_density_total = []
    error_density = []
    density_target = []
    density_output = []
    bin_centers = []
    for temp in hist_data:
        error_density_total.append(temp[0])
        error_density.append(temp[1])
        density_target.append(temp[2])
        density_output.append(temp[3])
        bin_centers.append(temp[4])
    error_density_total = np.array(error_density_total)
    error_density = np.array(error_density)
    density_target = np.array(density_target)
    density_output = np.array(density_output)
    bin_centers = np.array(bin_centers)[0]

    if model.channels == 0:
        bin_width = bin_centers[1] - bin_centers[0]
    elif model.channels == 1:
        bin_centers_ = bin_centers[0]
        bin_width = bin_centers_[1] - bin_centers_[0]

    # print("bin_width: ", bin_width)
    vmin_density = 0.0
    # Maximum density, depending if it is 2-D or 1-D diffusion
    vmax_density = 1.0 / ((bin_width)**model.Dx)

    vmin_error_density = 0.0
    vmax_error_density = density_target.max()
    time_vector = np.arange(target.shape[0]) * dt

    vmin = target.min()
    vmax = target.max()

    axes_iter = -1

    contours_vec = []

    if model.Dx == 1 and len(np.shape(density_target)) == 3:
        assert (np.shape(density_target)[1] == 1)
        density_target = density_target[:, 0, :]
        density_output = density_output[:, 0, :]
        error_density = error_density[:, 0, :]

        assert (np.shape(target)[2] == 1)
        target = target[:, :, 0]
        output = output[:, :, 0]

    if model.Dx == 1:
        axes_iter = axes_iter + 1
        mp = createDensityContour_(fig,
                                   axes[axes_iter],
                                   density_target,
                                   bin_centers,
                                   "Target Density $f_X(x)$",
                                   vmin_density,
                                   vmax_density,
                                   plt.get_cmap("Blues"),
                                   dt,
                                   xlabel=r"$x$",
                                   scale="log")
        contours_vec.append(mp)

        axes_iter = axes_iter + 1
        mp = createContour_(fig,
                            axes[axes_iter],
                            target,
                            "Target {:}".format(quantity),
                            vmin,
                            vmax,
                            plt.get_cmap("seismic"),
                            dt,
                            xlabel="{:}".format(xlabel))
        contours_vec.append(mp)

    if latent_states is not None:
        axes_iter = axes_iter + 1
        vmin_latent = np.min(latent_states)
        vmax_latent = np.max(latent_states)
        if np.shape(latent_states)[1] > 1:
            mp = createContour_(fig,
                                axes[axes_iter],
                                latent_states,
                                None,
                                vmin_latent,
                                vmax_latent,
                                plt.get_cmap("seismic"),
                                dt,
                                xlabel="Latent state")
            contours_vec.append(mp)
        else:
            axes[axes_iter].plot(latent_states[:, 0], time_vector)
            axes[axes_iter].set_xlabel(r"Latent state")
            axes[axes_iter].set_ylim((time_vector.min(), time_vector.max()))

    if model.Dx == 1:
        axes_iter = axes_iter + 1
        mp = createContour_(fig,
                            axes[axes_iter],
                            output,
                            "Output {:}".format(quantity),
                            vmin,
                            vmax,
                            plt.get_cmap("seismic"),
                            dt,
                            xlabel="{:}".format(xlabel))
        contours_vec.append(mp)
        axes_iter = axes_iter + 1
        mp = createDensityContour_(fig,
                                   axes[axes_iter],
                                   density_output,
                                   bin_centers,
                                   "Predicted Density $\\tilde{f}_X(x)$",
                                   vmin_density,
                                   vmax_density,
                                   plt.get_cmap("Blues"),
                                   dt,
                                   xlabel="$x$",
                                   scale="log")
        contours_vec.append(mp)
        axes[axes_iter].set_xticks(
            (np.min(bin_centers), 0, np.max(bin_centers)),
            (bounds[0], 0, bounds[1]))

        if with_multiscale_bar:
            axes_iter = axes_iter + 1
            # Add a bar plot demonstrating where it is multiscale and where not
            multiscale_rounds, macro_steps_per_round, micro_steps_per_round = model.getMultiscaleParams(
                testing_mode)
            start_idx = 0
            for round_ in range(multiscale_rounds):
                end_idx = start_idx + macro_steps_per_round[round_]
                start_t = start_idx * dt
                end_t = end_idx * dt
                axes[axes_iter].axhspan(
                    start_t,
                    end_t,
                    color="orange",
                    alpha=0.7,
                    label="Latent Space" if round_ == 0 else None)

                start_idx = end_idx
                if round_ < len(micro_steps_per_round):
                    end_idx = start_idx + micro_steps_per_round[round_]
                    start_t = start_idx * dt
                    end_t = end_idx * dt
                    axes[axes_iter].axhspan(
                        start_t,
                        end_t,
                        color="green",
                        alpha=0.7,
                        label="State Space" if round_ == 0 else None)
                    start_idx = end_idx
            axes[axes_iter].legend(loc="upper left",
                                   bbox_to_anchor=(1.05, 1),
                                   borderaxespad=0.)
            axes[axes_iter].set_ylim((time_vector.min(), time_vector.max()))
            axes[axes_iter].set_title(r"Multiscale?")
            axes[axes_iter].axis('off')

    if model.Dx == 1:
        axes_iter = axes_iter + 1
        mp = createDensityContour_(fig,
                                   axes[axes_iter],
                                   error_density,
                                   bin_centers,
                                   "Error Density",
                                   vmin_error_density,
                                   vmax_error_density,
                                   plt.get_cmap("Reds"),
                                   dt,
                                   xlabel="$x$")
        contours_vec.append(mp)
        axes[axes_iter].set_xticks(
            (np.min(bin_centers), 0, np.max(bin_centers)),
            (bounds[0], 0, bounds[1]))

    if nrows == 2: axes[axes_iter].set_ylabel(r"Time $t$")

    axes_iter = axes_iter + 1
    axes[axes_iter].plot(error_density_total, time_vector)
    # axes[axes_iter].set_xlabel(r"L1-NHD"))
    axes[axes_iter].set_xlim((0, 2))
    axes[axes_iter].set_ylim((time_vector.min(), time_vector.max()))
    axes[axes_iter].set_title(r"L1-NHD")

    if model.Dx == 1:
        axes_iter = axes_iter + 1
        axes[axes_iter].plot(wasserstein_distance_data, time_vector)
        # axes[axes_iter].set_xlabel(r"WD")
        axes[axes_iter].set_xlim((0, LL))
        axes[axes_iter].set_ylim((time_vector.min(), time_vector.max()))
        axes[axes_iter].set_title(r"WD")

    target_m1 = np.mean(target, 1)
    output_m1 = np.mean(output, 1)
    error_m1 = np.abs(output_m1 - target_m1)
    if model.Dx == 2 or model.Dx == 3: error_m1 = np.mean(error_m1, axis=-1)

    target_m2 = np.var(target, 1)
    output_m2 = np.var(output, 1)
    error_m2 = np.abs(output_m2 - target_m2)
    if model.Dx == 2 or model.Dx == 3: error_m2 = np.mean(error_m2, axis=-1)

    axes_iter = axes_iter + 1
    axes[axes_iter].plot(error_m1, time_vector)
    # axes[axes_iter].set_xlabel(r"M1 Error"))
    axes[axes_iter].set_xlim((0, LL))
    axes[axes_iter].set_ylim((time_vector.min(), time_vector.max()))
    axes[axes_iter].set_title(r"M1 Error")

    axes_iter = axes_iter + 1
    axes[axes_iter].plot(error_m2, time_vector)
    # axes[axes_iter].set_xlabel(r"M2 Error")
    axes[axes_iter].set_xlim((0, LL))
    axes[axes_iter].set_ylim((time_vector.min(), time_vector.max()))
    axes[axes_iter].set_title(r"M2 Error")

    if with_multiscale_bar:
        axes_iter = axes_iter + 1
        # Add a bar plot demonstrating where it is multiscale and where not
        multiscale_rounds, macro_steps_per_round, micro_steps_per_round = model.getMultiscaleParams(
            testing_mode)
        start_idx = 0
        for round_ in range(multiscale_rounds):
            end_idx = start_idx + macro_steps_per_round[round_]
            start_t = start_idx * dt
            end_t = end_idx * dt
            axes[axes_iter].axhspan(
                start_t,
                end_t,
                color="orange",
                alpha=0.7,
                label="Latent Space" if round_ == 0 else None)

            start_idx = end_idx
            if round_ < len(micro_steps_per_round):
                end_idx = start_idx + micro_steps_per_round[round_]
                start_t = start_idx * dt
                end_t = end_idx * dt
                axes[axes_iter].axhspan(
                    start_t,
                    end_t,
                    color="green",
                    alpha=0.7,
                    label="State Space" if round_ == 0 else None)
                start_idx = end_idx
        axes[axes_iter].legend(loc="upper left",
                               bbox_to_anchor=(1.05, 1),
                               borderaxespad=0.)
        axes[axes_iter].set_ylim((time_vector.min(), time_vector.max()))
        axes[axes_iter].set_title(r"Multiscale?")
        axes[axes_iter].axis('off')

    for contours in contours_vec:
        for pathcoll in contours.collections:
            pathcoll.set_rasterized(True)

    plt.tight_layout()
    fig_path = model.getFigureDir(
    ) + "/{:}_{:}_{:}_density_contour.{:}".format(testing_mode, set_name,
                                                  ic_idx, FIGTYPE)
    # plt.savefig(fig_path, dpi=300)
    plt.savefig(fig_path, dpi=100)
    plt.close()

    # if with_multiscale_bar:

    #     fig, axes = plt.subplots(nrows=1,
    #                              ncols=ncols,
    #                              figsize=(3.6 * ncols, 6),
    #                              sharey=True)
    #     fig.subplots_adjust(hspace=0.4, wspace=0.4)
    #     axes[0].set_ylabel(r"Time $t$")
    #     contours_vec = []

    #     multiscale_rounds, macro_steps_per_round, micro_steps_per_round = model.getMultiscaleParams(
    #         testing_mode)

    #     # Mulstiscale array
    #     indexes = np.array([])
    #     start_idx = 0
    #     for round_ in range(multiscale_rounds):
    #         end_idx = start_idx + macro_steps_per_round[round_] - 1
    #         indexes = np.concatenate((indexes, np.arange(start_idx, end_idx)),
    #                                  axis=None)
    #         if round_ < len(micro_steps_per_round):
    #             start_idx = end_idx + micro_steps_per_round[round_] + 1
    #         indexes = np.array(indexes).astype(int)

    #     where_latent_prop = np.zeros((latent_states.shape[0]))
    #     where_latent_prop[indexes] = 1.0
    #     where_state_prop = 1 - where_latent_prop
    #     where_latent_prop[0] = 1
    #     where_state_prop[0] = 0

    #     axis_iter = -1
    #     if model.Dx == 1:
    #         axis_iter = axis_iter + 1
    #         mp = createDensityContour_(fig,
    #                                    axes[axis_iter],
    #                                    density_target,
    #                                    bin_centers,
    #                                    "Target Density",
    #                                    vmin_density,
    #                                    vmax_density,
    #                                    plt.get_cmap("seismic"),
    #                                    dt,
    #                                    xlabel="Value")
    #         contours_vec.append(mp)
    #         axis_iter = axis_iter + 1
    #         mp = createContour_(fig,
    #                             axes[axis_iter],
    #                             target,
    #                             "Target {:}".format(quantity),
    #                             vmin,
    #                             vmax,
    #                             plt.get_cmap("seismic"),
    #                             dt,
    #                             xlabel="{:}".format(xlabel))
    #         contours_vec.append(mp)

    #     if np.shape(latent_states)[1] > 1:
    #         axis_iter = axis_iter + 1
    #         # mp = createContour_(fig, axes[axis_iter], latent_states, "Latent space", vmin_latent, vmax_latent, plt.get_cmap("seismic"), dt, mask_where=where_state_prop, xlabel="Latent state$")
    #         mp = createContour_(fig,
    #                             axes[axis_iter],
    #                             latent_states,
    #                             None,
    #                             vmin_latent,
    #                             vmax_latent,
    #                             plt.get_cmap("seismic"),
    #                             dt,
    #                             mask_where=where_state_prop,
    #                             xlabel="Latent state")
    #         contours_vec.append(mp)
    #     else:
    #         axis_iter = axis_iter + 1
    #         axes[axis_iter].plot(latent_states[:, 0], time_vector)
    #         axes[axis_iter].set_xlabel(r"Latent state")
    #         axes[axis_iter].set_ylim((time_vector.min(), time_vector.max()))

    #     if model.Dx == 1:
    #         axis_iter = axis_iter + 1
    #         mp = createContour_(fig,
    #                             axes[axis_iter],
    #                             output,
    #                             "Output {:}".format(quantity),
    #                             vmin,
    #                             vmax,
    #                             plt.get_cmap("seismic"),
    #                             dt,
    #                             mask_where=where_latent_prop,
    #                             xlabel="{:}".format(xlabel))
    #         contours_vec.append(mp)
    #         axis_iter = axis_iter + 1
    #         mp = createDensityContour_(fig,
    #                                    axes[axis_iter],
    #                                    density_output,
    #                                    bin_centers,
    #                                    "Predicted Density",
    #                                    vmin_density,
    #                                    vmax_density,
    #                                    plt.get_cmap("seismic"),
    #                                    dt,
    #                                    xlabel="Value")
    #         contours_vec.append(mp)
    #         axis_iter = axis_iter + 1
    #         mp = createDensityContour_(fig,
    #                                    axes[axis_iter],
    #                                    error_density,
    #                                    bin_centers,
    #                                    "Error Density",
    #                                    vmin_error_density,
    #                                    vmax_error_density,
    #                                    plt.get_cmap("Reds"),
    #                                    dt,
    #                                    xlabel="Value")
    #         contours_vec.append(mp)
    #         if ncols == 2: axes[axes_iter].set_ylabel(r"Time $t$")

    #         axis_iter = axis_iter + 1
    #         axes[axis_iter].plot(error_density_total, time_vector)
    #         axes[axis_iter].set_xlabel(r"L1-NHD Error")
    #         axes[axis_iter].set_xlim((0, 2))
    #         axes[axis_iter].set_ylim((time_vector.min(), time_vector.max()))
    #         # axes[axis_iter].set_title("L1-NHD Error")

    #     axis_iter = axis_iter + 1
    #     axes[axis_iter].plot(wasserstein_distance_data, time_vector)
    #     axes[axis_iter].set_xlabel(r"WD")
    #     axes[axis_iter].set_xlim((0, LL))
    #     axes[axis_iter].set_ylim((time_vector.min(), time_vector.max()))
    #     # axes[axis_iter].set_title("WD")

    #     axis_iter = axis_iter + 1
    #     axes[axis_iter].plot(error_m1, time_vector)
    #     axes[axis_iter].set_xlabel(r"M1 Error")
    #     axes[axis_iter].set_xlim((0, LL))
    #     axes[axis_iter].set_ylim((time_vector.min(), time_vector.max()))
    #     # axes[axis_iter].set_title("M1 Error")

    #     axis_iter = axis_iter + 1
    #     axes[axis_iter].plot(error_m2, time_vector)
    #     axes[axis_iter].set_xlabel(r"M2 Error")
    #     axes[axis_iter].set_xlim((0, LL))
    #     axes[axis_iter].set_ylim((time_vector.min(), time_vector.max()))
    #     # axes[axis_iter].set_title("M2 Error")

    #     axis_iter = axis_iter + 1
    #     # Add a bar plot demonstrating where it is multiscale and where not
    #     # axes[axis_iter].set_title("Multiscale?")
    #     axes[axis_iter].set_xlabel(r"Multiscale?")

    #     multiscale_rounds, macro_steps_per_round, micro_steps_per_round = model.getMultiscaleParams(
    #         testing_mode)

    #     start_idx = 0
    #     for round_ in range(multiscale_rounds):
    #         end_idx = start_idx + macro_steps_per_round[round_]
    #         start_t = start_idx * dt
    #         end_t = end_idx * dt
    #         axes[axis_iter].axhspan(
    #             start_t,
    #             end_t,
    #             color="orange",
    #             alpha=0.7,
    #             label="Latent Space" if round_ == 0 else None)

    #         start_idx = end_idx
    #         if round_ < len(micro_steps_per_round):
    #             end_idx = start_idx + micro_steps_per_round[round_]
    #             start_t = start_idx * dt
    #             end_t = end_idx * dt
    #             axes[axes_iter].axhspan(
    #                 end_t,
    #                 end_t,
    #                 color="green",
    #                 alpha=0.7,
    #                 label="State Space" if round_ == 0 else None)
    #             start_idx = end_idx

    #     for contours in contours_vec:
    #         for pathcoll in contours.collections:
    #             pathcoll.set_rasterized(True)
    #     plt.legend(loc="upper left",
    #                bbox_to_anchor=(1.05, 1),
    #                borderaxespad=0.)
    #     plt.axis('off')
    #     axes[axis_iter].set_ylim((time_vector.min(), time_vector.max()))
    #     fig_path = model.getFigureDir(
    #     ) + "/{:}_{:}_{:}_density_contour_propagations.{:}".format(
    #         testing_mode, set_name, ic_idx, FIGTYPE)
    #     plt.tight_layout()
    #     plt.savefig(fig_path, dpi=100)
    #     # plt.show()
    #     plt.close()

    if model.params["make_videos"]:

        video_folder = "{:}_density_video_{:}_IC{:}".format(
            testing_mode, set_name, ic_idx)
        n_frames_max, frame_path_python, frame_path_bash, video_path = makeVideoPaths(
            model, video_folder)

        n_frames = np.min([n_frames_max, np.shape(density_output)[0]])
        SUBSAMPLE = 1 if np.shape(density_output)[0] < n_frames else int(
            np.shape(density_output)[0] / n_frames)
        bin_size = bin_centers[1] - bin_centers[0]
        # ONLY_SAMPLES = True
        ONLY_SAMPLES = False
        # T_vec = range(n_frames) if not ONLY_SAMPLES else [0, 1000, 1970]
        STEP = int(np.shape(density_output)[0] / n_frames)
        T_vec = np.arange(0,
                          np.shape(density_output)[0],
                          SUBSAMPLE) if not ONLY_SAMPLES else [
                              0, 250, 500, 750, 1000, 1250, 1500, 1750, 1970
                          ]
        for t in T_vec:
            fig_path = frame_path_python.format(
                t) if not ONLY_SAMPLES else model.getFigureDir(
                ) + "/{:}_{:}_{:}_video_frame_{:}_propagations.{:}".format(
                    testing_mode, set_name, ic_idx, t, FIGTYPE)
            if model.Dx == 1:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.bar(bin_centers,
                       density_target[t],
                       width=bin_size,
                       color=color_dict["green"],
                       alpha=0.5,
                       label="Target Density")
                ax.bar(bin_centers,
                       density_output[t],
                       width=bin_size,
                       color=color_dict["blue"],
                       alpha=0.5,
                       label="Predicted Density")
                ax.legend(loc="upper left",
                          bbox_to_anchor=(1.05, 1),
                          borderaxespad=0.)
                ax.set_ylim((vmin_density, vmax_density))
                ax.set_xlim((min_target, max_target))
            elif model.Dx == 2:
                ncols = 4
                nrows = 1
                fig, axes = plt.subplots(nrows=nrows,
                                         ncols=ncols,
                                         figsize=(5 * ncols, 4 * nrows),
                                         squeeze=False)
                x, y = np.meshgrid(bin_centers[0], bin_centers[1])
                xmin = x.min()
                xmax = x.max()

                # print(ark)
                mp = axes[0, 0].contourf(x,
                                         y,
                                         density_target[t],
                                         15,
                                         cmap=plt.get_cmap("Reds"),
                                         levels=np.linspace(
                                             vmin_density, vmax_density, 60),
                                         extend="both")
                fig.colorbar(mp, ax=axes[0, 0])
                axes[0, 0].set_title("Target")
                axes[0, 0].set_xlim((xmin, xmax))
                axes[0, 0].set_ylim((xmin, xmax))

                mp = axes[0, 1].contourf(x,
                                         y,
                                         density_output[t],
                                         15,
                                         cmap=plt.get_cmap("Reds"),
                                         levels=np.linspace(
                                             vmin_density, vmax_density, 60),
                                         extend="both")
                fig.colorbar(mp, ax=axes[0, 1])
                axes[0, 1].set_title("Prediction")
                axes[0, 1].set_xlim((xmin, xmax))
                axes[0, 1].set_ylim((xmin, xmax))

                mp = axes[0, 2].contourf(
                    x,
                    y,
                    np.abs(density_output[t] - density_target[t]),
                    15,
                    cmap=plt.get_cmap("Reds"),
                    levels=np.linspace(vmin_density, vmax_density, 60),
                    extend="both")
                fig.colorbar(mp, ax=axes[0, 2])
                axes[0, 2].set_title("Absolute Error")
                axes[0, 2].set_xlim((xmin, xmax))
                axes[0, 2].set_ylim((xmin, xmax))

                axes[0, 3].plot(target[t, :, 0],
                                target[t, :, 1],
                                "x",
                                label="Target")
                axes[0, 3].plot(output[t, :, 0],
                                output[t, :, 1],
                                "o",
                                label="Prediction")
                axes[0, 3].legend(loc="upper left",
                                  bbox_to_anchor=(1.05, 1),
                                  borderaxespad=0.)
                axes[0, 3].set_xlim((min_target, max_target))
                axes[0, 3].set_ylim((min_target, max_target))
                axes[0, 3].set_title("Particle positions")
            elif model.Dx == 3:
                fig = plt.figure(figsize=plt.figaspect(0.75) * 1.4)
                axes = fig.gca(projection='3d')
                if ("iterative" in testing_mode) or ("autoencoder"
                                                     in testing_mode):
                    color = "tab:red"
                    label = "T_{\mu}=0 "
                    label = "$" + label + "$"
                    label = "LED, " + label
                elif "multiscale" in testing_mode:
                    temp = testing_mode.split("_")
                    macro_steps = int(float(temp[-1]))
                    micro_steps = int(float(temp[-3]))
                    color = "tab:green"
                    if macro_steps == 900: color = "tab:blue"
                    if macro_steps == 500: color = "tab:green"
                    label = "T_{m}=" + "{:.0f}".format(float(macro_steps) * dt)
                    label = label + ", \, \\rho=" + "{:.0f}".format(
                        float(macro_steps / micro_steps))
                    label = "$" + label + "$"
                    label = "LED, " + label
                else:
                    raise ValueError("I do not know which color to pick.")
                # markersize = 80
                # axes.scatter3D(target[t,:,0], target[t,:,1], target[t,:,2], marker="P", label="Groundtruth", color="green", alpha=0.9)
                axes.scatter3D(target[t, :, 0],
                               target[t, :, 1],
                               target[t, :, 2],
                               marker="P",
                               label="Groundtruth",
                               color="k",
                               alpha=0.9)
                axes.scatter3D(output[t, :, 0],
                               output[t, :, 1],
                               output[t, :, 2],
                               marker="o",
                               label=r"{:}".format(label),
                               color=color,
                               alpha=0.9)
                axes.set_xlim([min_target, max_target])
                axes.set_ylim([min_target, max_target])
                axes.set_zlim([min_target, max_target])
                axes.set_xlabel(r"$X_1$", labelpad=20)
                axes.set_ylabel(r"$X_2$", labelpad=20)
                axes.set_zlabel(r"$X_3$", labelpad=20)
                # plt.title("Particle positions", y=1.08)
                # axes.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
                axes.legend(loc="upper center",
                            bbox_to_anchor=(0.5, 1.1),
                            borderaxespad=0.,
                            ncol=2,
                            frameon=False,
                            fontsize=24,
                            markerscale=3.,
                            columnspacing=0.5)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=100)
            plt.close()
        # makeVideo(model, video_path, frame_path_bash, n_frames_max)


def plotTestingContours(model,
                        target,
                        output,
                        dt,
                        ic_idx,
                        set_name,
                        latent_states=None,
                        testing_mode="",
                        with_multiscale_bar=False,
                        quantity="Positions",
                        xlabel="Positions",
                        hist_data=None,
                        wasserstein_distance_data=None):

    print("# plotTestingContours() # - {:}, {:}, Multiscale bar? {:}".format(
        testing_mode, set_name, with_multiscale_bar))

    if model.data_info_dict["contour_plots"]:
        plotTestingContourEvolution(model,
                                    target,
                                    output,
                                    dt,
                                    ic_idx,
                                    set_name,
                                    latent_states=latent_states,
                                    testing_mode=testing_mode,
                                    with_multiscale_bar=with_multiscale_bar)

    if model.data_info_dict["density_plots"]:
        plotTestingContourDensity(
            model,
            target,
            output,
            dt,
            ic_idx,
            set_name,
            latent_states=latent_states,
            testing_mode=testing_mode,
            with_multiscale_bar=with_multiscale_bar,
            quantity=quantity,
            xlabel=xlabel,
            hist_data=hist_data,
            wasserstein_distance_data=wasserstein_distance_data)

    # if len(np.shape(target)) == 2:
    #     N_PLOT_MAX = 1000
    #     target = target[:N_PLOT_MAX]
    #     output = output[:N_PLOT_MAX]
    #     # PLOTTING 10 SIGNALS FOR REFERENCE
    #     plot_max = np.min([np.shape(target)[1], 10])
    #     fig_path = model.getFigureDir() + "/{:}_{:}_{:}_signals.{:}".format(
    #         testing_mode, set_name, ic_idx, FIGTYPE)
    #     for idx in range(plot_max):
    #         plt.plot(np.arange(np.shape(output)[0]),
    #                  output[:, idx],
    #                  color='blue',
    #                  linewidth=1.0,
    #                  label='Output' if idx == 0 else None)
    #         plt.plot(np.arange(np.shape(target)[0]),
    #                  target[:, idx],
    #                  color='red',
    #                  linewidth=1.0,
    #                  label='Target' if idx == 0 else None)
    #     plt.legend(loc="upper left",
    #                bbox_to_anchor=(1.05, 1),
    #                borderaxespad=0.)
    #     plt.tight_layout()
    #     plt.savefig(fig_path, dpi=300)
    #     plt.close()

    #     fig_path = model.getFigureDir(
    #     ) + "/{:}_{:}_{:}_signals_target.{:}".format(testing_mode, set_name,
    #                                                  ic_idx, FIGTYPE)
    #     for idx in range(plot_max):
    #         plt.plot(np.arange(np.shape(target)[0]),
    #                  target[:, idx],
    #                  linewidth=1.0,
    #                  label='Target' if idx == 0 else None)
    #     plt.legend(loc="upper left",
    #                bbox_to_anchor=(1.05, 1),
    #                borderaxespad=0.)
    #     plt.tight_layout()
    #     plt.savefig(fig_path, dpi=300)
    #     plt.close()

    #     fig_path = model.getFigureDir(
    #     ) + "/{:}_{:}_{:}_signals_output.{:}".format(testing_mode, set_name,
    #                                                  ic_idx, FIGTYPE)
    #     for idx in range(plot_max):
    #         plt.plot(np.arange(np.shape(output)[0]),
    #                  output[:, idx],
    #                  linewidth=1.0,
    #                  label='Output' if idx == 0 else None)
    #     plt.legend(loc="upper left",
    #                bbox_to_anchor=(1.05, 1),
    #                borderaxespad=0.)
    #     plt.tight_layout()
    #     plt.savefig(fig_path, dpi=300)
    #     plt.close()

    elif len(np.shape(target)) == 4:
        createIterativePredictionPlotsForImageData(model, target, output, dt,
                                                   ic_idx, set_name,
                                                   testing_mode)


def createDensityContour_(fig,
                          ax,
                          density,
                          bins,
                          title,
                          vmin,
                          vmax,
                          cmap,
                          dt,
                          xlabel="Value",
                          scale=None):
    ax.set_title(title)
    t, s = np.meshgrid(np.arange(density.shape[0]) * dt, bins)
    if scale is None:
        mp = ax.contourf(s,
                         t,
                         np.transpose(density),
                         cmap=cmap,
                         levels=np.linspace(vmin, vmax, 60),
                         extend="both")
    elif scale == "log":
        from matplotlib import ticker
        mp = ax.contourf(s,
                         t,
                         np.transpose(density),
                         cmap=cmap,
                         locator=ticker.LogLocator(),
                         extend="both")
    fig.colorbar(mp, ax=ax)
    ax.set_xlabel(r"{:}".format(xlabel))
    return mp


def createContour_(fig,
                   ax,
                   data,
                   title,
                   vmin,
                   vmax,
                   cmap,
                   dt,
                   mask_where=None,
                   xlabel=None):
    ax.set_title(title)
    time_vec = np.arange(data.shape[0]) * dt
    state_vec = np.arange(data.shape[1])
    if mask_where is not None:
        # print(mask_where)
        mask = [
            mask_where[i] * np.ones(data.shape[1])
            for i in range(np.shape(mask_where)[0])
        ]
        mask = np.array(mask)
        data = np.ma.array(data, mask=mask)

    t, s = np.meshgrid(time_vec, state_vec)
    mp = ax.contourf(s,
                     t,
                     np.transpose(data),
                     15,
                     cmap=cmap,
                     levels=np.linspace(vmin, vmax, 60),
                     extend="both")
    fig.colorbar(mp, ax=ax)
    ax.set_xlabel(r"{:}".format(xlabel))
    return mp


def plotStateDistributions(model, results, set_name, testing_mode):
    keys_ = model.data_info_dict.keys()
    if ('statistics_cummulative'
            in keys_) or ('statistics_per_state'
                          in keys_) or ('statistics_per_channel' in keys_):

        print("# plotStateDistributions() #")

        if "state_dist_hist_data" in results:

            hist_data = results["state_dist_hist_data"]
            LL = results["state_dist_LL"]
            if ('statistics_per_state'
                    in keys_) and not ('statistics_per_timestep' in keys_):

                assert (type(LL) is list)

                # Iterating over all states
                N = len(LL)
                if N < 30:
                    for n in range(N):
                        _, _, density_target, density_pred, bin_centers = hist_data[
                            n]
                        bounds = results["state_dist_bounds"][n]
                        figdir = model.getFigureDir()
                        plotStateDistribution(figdir, density_target,
                                              density_pred, bin_centers,
                                              bounds, n, set_name,
                                              testing_mode)
                else:
                    figdir = model.getFigureDir()
                    plotStateDistributionLarge(figdir, results, set_name,
                                               testing_mode)

            elif ('statistics_per_state'
                  in keys_) and ('statistics_per_timestep' in keys_):
                print("No plotting of statistics for this case.")

            else:
                # CUMMULATIVE STATE DISTRIBUTION
                _, _, density_target, density_pred, bin_centers = hist_data
                bounds = results["state_dist_bounds"]
                figdir = model.getFigureDir()
                plotStateDistribution(figdir, density_target, density_pred,
                                      bin_centers, bounds, 0, set_name,
                                      testing_mode)
        else:
            print(
                "# No computation of state statistics. state_dist_hist_data not found in results."
            )


def plotStateDistributionLarge(
    figdir,
    results,
    set_name,
    testing_mode,
    label1="Target Density",
    label2="Predicted Density",
):
    print("# plotStateDistributionLarge() #")
    if "state_dist_hist_data" in results:

        hist_data = results["state_dist_hist_data"]

        N = len(hist_data)
        nrows = 5
        ncols = 5

        num_plots = int(np.ceil(N / (ncols * nrows)))
        print("num_plots = {:}".format(num_plots))

        state_num_start = 0
        state_num = 0
        for plot in range(num_plots):
            fig, axes = plt.subplots(nrows, ncols, figsize=(20, 20))
            for i in range(nrows):
                for j in range(ncols):
                    if state_num >= N:
                        break
                    ax = axes[i, j]
                    _, _, density_target, density_pred, bin_centers = hist_data[
                        state_num]
                    bounds = results["state_dist_bounds"][state_num]

                    bin_width = bin_centers[1] - bin_centers[0]
                    vmax_density = np.max(
                        [np.max(density_target),
                         np.max(density_pred)])
                    vmin_density = 0.0
                    ax.bar(bin_centers,
                           density_target,
                           width=bin_width,
                           color=color_dict["green"],
                           alpha=0.5,
                           label=label1)
                    ax.bar(bin_centers,
                           density_pred,
                           width=bin_width,
                           color=color_dict["blue"],
                           alpha=0.5,
                           label=label2)
                    if j == ncols - 1:
                        ax.legend(loc="upper left",
                                  bbox_to_anchor=(1.05, 1),
                                  borderaxespad=0.,
                                  ncol=1,
                                  frameon=False)

                    ax.set_ylim((vmin_density, vmax_density))
                    ax.set_xlim(bounds)
                    # state_label = "s_{" + "{:}".format(state_num+1) + "}"
                    state_label = "x_{" + "{:}".format(state_num + 1) + "}"
                    xlabel = "$" + state_label + "$"
                    ax.set_xlabel(r"{:}".format(xlabel))
                    ylabel = "$" + "f_{" + state_label + "}(" + state_label + ")$"
                    ax.set_ylabel(r"{:}".format(ylabel))
                    state_num += 1
                if state_num >= N:
                    break
            state_num_end = state_num - 1
            fig_path = figdir + "/{:}_state_dist_bar_S{:}-{:}_{:}.{:}".format(
                testing_mode, state_num_start, state_num_end, set_name,
                FIGTYPE)
            # plt.show()
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300)
            plt.close()
            state_num_start = state_num
        else:
            print("No state_dist_hist_data found.")


def plotStateDistribution(figdir, density_target, density_pred, bin_centers,
                          bounds, state_num, set_name, testing_mode):

    ################################
    ### BAR PLOTS - (.bar())
    ################################

    fig_path = figdir + "/{:}_state_dist_bar_S{:}_{:}.{:}".format(
        testing_mode, state_num, set_name, FIGTYPE)
    bin_width = bin_centers[1] - bin_centers[0]
    vmax_density = np.max([np.max(density_target), np.max(density_pred)])
    vmin_density = 0.0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(bin_centers,
           density_target,
           width=bin_width,
           color=color_dict["green"],
           alpha=0.5,
           label="Target Density")
    ax.bar(bin_centers,
           density_pred,
           width=bin_width,
           color=color_dict["blue"],
           alpha=0.5,
           label="Predicted Density")
    ax.legend(loc="upper center",
              bbox_to_anchor=(0.5, 1.1),
              borderaxespad=0.,
              ncol=2,
              frameon=False)
    ax.set_ylim((vmin_density, vmax_density))
    ax.set_xlim(bounds)
    # state_label = "s_{" + "{:}".format(state_num+1) + "}"
    state_label = "x_{" + "{:}".format(state_num + 1) + "}"
    xlabel = "$" + state_label + "$"
    ax.set_xlabel(r"{:}".format(xlabel))
    ylabel = "$" + "f_{" + state_label + "}(" + state_label + ")$"
    ax.set_ylabel(r"{:}".format(ylabel))
    # plt.show()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    ################################
    ### PLOTS - (.plt())
    ################################

    # fig_path = figdir + "/{:}_state_dist_S{:}_{:}.{:}".format(testing_mode, state_num, set_name, FIGTYPE)
    # bin_width = bin_centers[1] - bin_centers[0]
    # vmax_density = np.max([np.max(density_target), np.max(density_pred)])
    # vmin_density = 0.0
    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.plot(bin_centers, density_target, color=color_dict["green"],label="Target Density")
    # ax.plot(bin_centers, density_pred, color=color_dict["blue"], label="Predicted Density")
    # ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), borderaxespad=0., ncol=2, frameon=False)
    # ax.set_ylim((vmin_density, vmax_density))
    # ax.set_xlim(bounds)
    # state_label = "s_{" + "{:}".format(state_num) + "}"
    # xlabel = "$" + state_label + "$"
    # ax.set_xlabel(r"{:}".format(xlabel))
    # ylabel = "$" + "f_{" + state_label + "}(" + state_label +")$"
    # ax.set_ylabel(r"{:}".format(ylabel))
    # # plt.show()
    # plt.tight_layout()
    # plt.savefig(fig_path, dpi=300)
    # plt.close()


def plotSpectrum(model, results, set_name, testing_mode=""):
    assert ("sp_true" in results)
    assert ("sp_pred" in results)
    assert ("freq_true" in results)
    assert ("freq_pred" in results)
    sp_true = results["sp_true"]
    sp_pred = results["sp_pred"]
    freq_true = results["freq_true"]
    freq_pred = results["freq_pred"]
    fig_path = model.getFigureDir() + "/{:}_{:}_frequencies.{:}".format(
        testing_mode, set_name, FIGTYPE)
    spatial_dims = len(np.shape(sp_pred))
    if spatial_dims == 1:
        # plt.title("Frequency error={:.4f}".format(np.mean(np.abs(sp_true-sp_pred))))
        plt.plot(freq_pred, sp_pred, '--', color="tab:red", label="prediction")
        plt.plot(freq_true, sp_true, '--', color="tab:green", label="target")
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power Spectrum [dB]')
        # plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.legend(loc="upper center",
                   bbox_to_anchor=(0.5, 1.1),
                   borderaxespad=0.,
                   ncol=2,
                   frameon=False)
    elif spatial_dims == 2:
        fig, axes = plt.subplots(figsize=(8, 8), ncols=2)
        plt.suptitle("Frequency error = {:.4f}".format(
            np.mean(np.abs(sp_true - sp_pred))))
        mp1 = axes[0].imshow(sp_true,
                             cmap=plt.get_cmap("plasma"),
                             aspect=1.0,
                             interpolation='lanczos')
        axes[0].set_title("True Spatial FFT2D")
        mp2 = axes[1].imshow(sp_pred,
                             cmap=plt.get_cmap("plasma"),
                             aspect=1.0,
                             interpolation='lanczos')
        axes[1].set_title("Predicted Spatial FFT2D")
        fig.colorbar(mp1, ax=axes[0])
        fig.colorbar(mp2, ax=axes[1])
    else:
        raise ValueError("Not implemented.")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


def plotLatentDynamicsComparison(model, latent_states_dict, set_name):
    max_plot = 3
    iter_ = 0
    for key1, value1 in latent_states_dict.items():
        for key2, value2 in latent_states_dict.items():
            if key1 != key2:
                plotLatentDynamicsComparison_(model, value1[0], value2[0],
                                              key1, key2, set_name)
                iter_ += 1
            if iter_ > max_plot:
                break


def plotLatentDynamicsComparison_(model,
                                  latent_states1,
                                  latent_states2,
                                  label1,
                                  label2,
                                  set_name,
                                  latent_states3=None,
                                  label3=None):
    shape_ = np.shape(latent_states1)
    if len(shape_) == 2:
        T, D = shape_
        if D >= 2:
            fig, ax = plt.subplots()
            plt.title("Latent dynamics in {:}".format(set_name))
            arrowplot(ax,
                      latent_states1[:, 0],
                      latent_states1[:, 1],
                      nArrs=100,
                      color="blue",
                      label=label1)
            arrowplot(ax,
                      latent_states2[:, 0],
                      latent_states2[:, 1],
                      nArrs=100,
                      color="green",
                      label=label2)
            if label3 is not None:
                arrowplot(ax,
                          latent_states3[:, 0],
                          latent_states3[:, 1],
                          nArrs=100,
                          color="tab:red",
                          label=label3)
            plt.legend(loc="upper left",
                       bbox_to_anchor=(1.05, 1),
                       borderaxespad=0.)
            fig_path = model.getFigureDir(
            ) + "/Comparison_latent_dynamics_{:}_{:}_{:}_{:}.{:}".format(
                set_name, label1, label2, label3, FIGTYPE)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300)
            plt.close()
        else:
            fig, ax = plt.subplots()
            plt.title("Latent dynamics in {:}".format(set_name))
            plt.plot(latent_states1[:-1, 0],
                     latent_states1[1:, 0],
                     'b',
                     linewidth=1.0,
                     label=label1)
            plt.plot(latent_states2[:-1, 0],
                     latent_states2[1:, 0],
                     'g',
                     linewidth=1.0,
                     label=label2)
            if label3 is not None:
                plt.plot(latent_states3[:-1, 0],
                         latent_states3[1:, 0],
                         'r',
                         linewidth=1.0,
                         label=label3)
            fig_path = model.getFigureDir(
            ) + "/Comparison_latent_dynamics_{:}_{:}_{:}_{:}.{:}".format(
                set_name, label1, label2, label3, FIGTYPE)
            plt.legend(loc="upper left",
                       bbox_to_anchor=(1.05, 1),
                       borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300)
            plt.close()


def scatterDensityLatentDynamicsPlot(x,
                                     y,
                                     ax=None,
                                     sort=True,
                                     bins=20,
                                     cmap=plt.get_cmap("Reds"),
                                     with_colorbar=True,
                                     log_norm=True):
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None:
        fig, ax = plt.subplots()
    data, x_e, y_e = np.histogram2d(x, y, bins=bins)
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
    if with_colorbar: plt.colorbar(mp)
    return ax
