#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

##############################################################
### Utilities for the analysis (meta stable clustering)
### on the latent space based on the free energy projection
### Common for BMP, Alanine and TRP
##############################################################

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

FIGTYPE = "pdf"

import io

from . import utils_latent_analysis
from . import utils_data
from . import utils_statistics


def plotLatentDynamicsComparisonSystemCommon(model, set_name):
    print("# plotLatentDynamicsComparisonSystemCommon() #")

    assert (model.params["latent_state_dim"] == 1)

    testing_mode = "teacher_forcing_forecasting"
    data_path = model.getResultsDir() + "/results_{:}_{:}".format(
        testing_mode, set_name)

    try:
        results = utils_data.loadData(data_path, model.save_format)
    except Exception as inst:
        print(inst)
        print(
            "# Something went wrong during loading of file {:}. Not plotting the latent comparison results."
            .format(data_path))
        return 0

    # Results of teacher forcing the MD data
    tf_latent_state_free_energy_grid = results["latent_state_free_energy_grid"]
    tf_latent_states_flatten_range = results["latent_states_flatten_range"]
    tf_latent_state_freenergy_ = results["latent_state_freenergy"]
    tf_free_energy_latent_state_regions = results[
        "free_energy_latent_state_regions"]
    tf_latent_states_all = results["latent_states_all"]
    tf_covariance_factor = results["covariance_factor"]
    tf_targets_all = results["targets_all"]

    # print(np.shape(utils_data.loadData(model.getResultsDir() + "/results_{:}_{:}".format("teacher_forcing_forecasting", "train"), model.save_format)["latent_states_all"]))
    # print(np.shape(utils_data.loadData(model.getResultsDir() + "/results_{:}_{:}".format("teacher_forcing_forecasting", "test"), model.save_format)["latent_states_all"]))
    # print(np.shape(utils_data.loadData(model.getResultsDir() + "/results_{:}_{:}".format("teacher_forcing_forecasting", "val"), model.save_format)["latent_states_all"]))

    # print(np.shape(utils_data.loadData(model.getResultsDir() + "/results_{:}_{:}".format("iterative_latent_forecasting", "train"), model.save_format)["latent_states_all"]))
    # print(np.shape(utils_data.loadData(model.getResultsDir() + "/results_{:}_{:}".format("iterative_latent_forecasting", "test"), model.save_format)["latent_states_all"]))
    # print(np.shape(utils_data.loadData(model.getResultsDir() + "/results_{:}_{:}".format("iterative_latent_forecasting", "val"), model.save_format)["latent_states_all"]))
    # print(ark)

    testing_mode = "iterative_latent_forecasting"
    data_path = model.getResultsDir() + "/results_{:}_{:}".format(
        testing_mode, set_name)
    try:
        results = utils_data.loadData(data_path, model.save_format)
    except Exception as inst:
        print(inst)
        print(
            "# Something went wrong during loading of file {:}. Not plotting the latent comparison results."
            .format(data_path))
        return 0

    # Results of iterative latent forecasting
    lf_latent_state_free_energy_grid = results["latent_state_free_energy_grid"]
    lf_latent_states_flatten_range = results["latent_states_flatten_range"]
    lf_latent_state_freenergy_ = results["latent_state_freenergy"]
    lf_free_energy_latent_state_regions = results[
        "free_energy_latent_state_regions"]
    lf_latent_states_all = results["latent_states_all"]
    lf_covariance_factor = results["covariance_factor"]

    # fig, ax = plt.subplots()
    # plt.plot(tf_latent_state_free_energy_grid,
    #          tf_latent_state_freenergy_,
    #          "g",
    #          label="MD",
    #          linewidth=2)
    # plt.plot(lf_latent_state_free_energy_grid,
    #          lf_latent_state_freenergy_,
    #          "b",
    #          label="LED Iterative",
    #          linewidth=2)

    # margin = 0.2 * (np.max(tf_latent_state_freenergy_) -
    #                 np.min(tf_latent_state_freenergy_))
    # freenergy_min_global_margin = np.min(tf_latent_state_freenergy_) - margin

    # for region in tf_free_energy_latent_state_regions:
    #     size = region[1] - region[0]
    #     idx_region = np.bitwise_and(
    #         tf_latent_state_free_energy_grid > region[0],
    #         tf_latent_state_free_energy_grid < region[1])
    #     idx_region = np.where(idx_region)[0]
    #     if len(idx_region) > 0:
    #         # freenergy_min = np.min(tf_latent_state_freenergy_[idx_region])
    #         idx_region = np.bitwise_and(
    #             lf_latent_state_free_energy_grid > region[0],
    #             lf_latent_state_free_energy_grid < region[1])
    #         idx_region = np.where(idx_region)[0]
    #         if len(idx_region) > 0:
    #             # freenergy_min2 = np.min(lf_latent_state_freenergy_[idx_region])
    #             # freenergy_min = np.min([freenergy_min, freenergy_min2])

    #             plt.errorbar((region[0] + region[1]) / 2,
    #                          freenergy_min_global_margin,
    #                          xerr=size,
    #                          color="k",
    #                          capsize=8,
    #                          capthick=4,
    #                          linewidth=4,
    #                          clip_on=False)
    #             plt.gca().set_ylim(bottom=freenergy_min_global_margin)

    # plt.ylabel(r'$F/\kappa_B T $')
    # plt.xlabel(r'$z$')
    # fig.tight_layout()
    # plt.savefig(
    #     model.getFigureDir() +
    #     "/Comparison_latent_dynamics_{:}.{:}".format(set_name, FIGTYPE),
    #     bbox_inches="tight",
    #     dpi=300)
    # plt.legend(loc="upper center",
    #            bbox_to_anchor=(0.5, 1.225),
    #            borderaxespad=0.,
    #            frameon=False)
    # plt.savefig(
    #     model.getFigureDir() +
    #     "/Comparison_latent_dynamics_{:}_legend.{:}".format(set_name, FIGTYPE),
    #     bbox_inches="tight",
    #     dpi=300)
    # plt.close()


    testing_mode = "teacher_forcing_forecasting"
    data_path = model.getResultsDir() + "/results_{:}_{:}".format(
        testing_mode, "train")
    try:
        results = utils_data.loadData(data_path, model.save_format)
    except Exception as inst:
        print(inst)
        print(
            "# Something went wrong during loading of file {:}. Not plotting the latent comparison results."
            .format(data_path))
        return 0

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
    try:
        results = utils_data.loadData(data_path, model.save_format)
    except Exception as inst:
        print(inst)
        print(
            "# Something went wrong during loading of file {:}. Not plotting the latent comparison results."
            .format(data_path))
        return 0

    # Results of iterative latent forecasting
    val_latent_state_free_energy_grid = results["latent_state_free_energy_grid"]
    val_latent_states_flatten_range = results["latent_states_flatten_range"]
    val_latent_state_freenergy_ = results["latent_state_freenergy"]
    val_free_energy_latent_state_regions = results[
        "free_energy_latent_state_regions"]
    val_latent_states_all = results["latent_states_all"]
    val_covariance_factor = results["covariance_factor"]

    train_latent_states_all = np.concatenate((train_latent_states_all, val_latent_states_all), axis=0)
    # print(np.shape(train_latent_states_all))
    # print(ark)

    max_ = np.max([np.max(tf_latent_states_all), np.max(lf_latent_states_all), np.max(train_latent_states_all)])
    min_ = np.min([np.min(tf_latent_states_all), np.min(lf_latent_states_all), np.min(train_latent_states_all)])
    latent_states_flatten_range = max_ - min_
    max_ += 0.05 * latent_states_flatten_range
    min_ -= 0.05 * latent_states_flatten_range
    latent_state_free_energy_grid = np.linspace(min_, max_, 400)

    # tf_latent_states_flatten = tf_latent_states_all.flatten()
    # tf_latent_state_freenergy = utils_latent_analysis.calculateLatentFreeEnergy(
    #     tf_latent_states_flatten, tf_covariance_factor,
    #     latent_state_free_energy_grid)

    # lf_latent_states_flatten = lf_latent_states_all.flatten()
    # lf_latent_state_freenergy = utils_latent_analysis.calculateLatentFreeEnergy(
    #     lf_latent_states_flatten, tf_covariance_factor,
    #     latent_state_free_energy_grid)

    # margin = 0.2 * (np.max(lf_latent_state_freenergy_) -
    #                 np.min(lf_latent_state_freenergy_))
    # freenergy_min_global_margin = np.min(lf_latent_state_freenergy_) - margin

    # fig, ax = plt.subplots()
    # plt.plot(latent_state_free_energy_grid,
    #          tf_latent_state_freenergy,
    #          "g",
    #          label="MD",
    #          linewidth=2)
    # plt.plot(latent_state_free_energy_grid,
    #          lf_latent_state_freenergy,
    #          "b",
    #          label="LED Iterative",
    #          linewidth=2)
    # for region in tf_free_energy_latent_state_regions:
    #     size = region[1] - region[0]
    #     idx_region = np.bitwise_and(
    #         tf_latent_state_free_energy_grid > region[0],
    #         tf_latent_state_free_energy_grid < region[1])
    #     idx_region = np.where(idx_region)[0]
    #     if len(idx_region) > 0:
    #         # freenergy_min = np.min(tf_latent_state_freenergy_[idx_region])

    #         idx_region = np.bitwise_and(
    #             lf_latent_state_free_energy_grid > region[0],
    #             lf_latent_state_free_energy_grid < region[1])
    #         idx_region = np.where(idx_region)[0]
    #         if len(idx_region) > 0:
    #             # freenergy_min2 = np.min(lf_latent_state_freenergy_[idx_region])
    #             # freenergy_min = np.min([freenergy_min, freenergy_min2])
    #             # margin = 0.05 * (np.max(tf_latent_state_freenergy_)-np.min(tf_latent_state_freenergy_))

    #             plt.errorbar((region[0] + region[1]) / 2,
    #                          freenergy_min_global_margin,
    #                          xerr=size,
    #                          color="k",
    #                          capsize=8,
    #                          capthick=4,
    #                          linewidth=4,
    #                          clip_on=False)
    #             plt.gca().set_ylim(bottom=freenergy_min_global_margin)

    # plt.ylabel(r'$F/\kappa_B T $')
    # plt.xlabel(r'$z$')
    # fig.tight_layout()
    # plt.savefig(model.getFigureDir() +
    #             "/Comparison_latent_dynamics_calibrated_{:}.{:}".format(
    #                 set_name, FIGTYPE),
    #             bbox_inches="tight",
    #             dpi=300)
    # plt.legend(loc="upper center",
    #            bbox_to_anchor=(0.5, 1.225),
    #            borderaxespad=0.,
    #            frameon=False)
    # plt.savefig(model.getFigureDir() +
    #             "/Comparison_latent_dynamics_calibrated_{:}_legend.{:}".format(
    #                 set_name, FIGTYPE),
    #             bbox_inches="tight",
    #             dpi=300)
    # plt.close()


    margin = 0.2 * (np.max(lf_latent_state_freenergy_) -
                    np.min(lf_latent_state_freenergy_))
    freenergy_min_global_margin = np.min(lf_latent_state_freenergy_) - margin

    print("Shapes of datasets for free energy comparison:")
    print("np.shape(train_latent_states_all) = {:}".format(np.shape(train_latent_states_all)))
    print("np.shape(tf_latent_states_all) = {:}".format(np.shape(tf_latent_states_all)))
    print("np.shape(lf_latent_states_all) = {:}".format(np.shape(lf_latent_states_all)))


    if model.system_name == "BMP":
        n_splits_1 = 64
        n_splits_2 = 96
        n_splits_3 = 1

    elif model.system_name == "Alanine":
        n_splits_1 = 96
        n_splits_2 = 112
        n_splits_3 = 4
        # n_splits_1 = 48
        # n_splits_2 = 56

        # n_splits_1 = 1
        # n_splits_2 = 1
        # n_splits_3 = 1

    else:
        n_splits_1 = 64
        n_splits_2 = 64
        n_splits_3 = 1

    train_latent_state_freenergy_bootstrap = utils_latent_analysis.calculateLatentFreeEnergyWithUncertainty(
        train_latent_states_all, n_splits_1, tf_covariance_factor,
        latent_state_free_energy_grid)
    train_latent_state_freenergy_mean = np.mean(train_latent_state_freenergy_bootstrap, axis=0)
    train_latent_state_freenergy_std = np.std(train_latent_state_freenergy_bootstrap, axis=0)
    train_latent_state_freenergy_std = train_latent_state_freenergy_std / np.sqrt(n_splits_1)

    lf_latent_state_freenergy_bootstrap = utils_latent_analysis.calculateLatentFreeEnergyWithUncertainty(
        lf_latent_states_all, n_splits_2, tf_covariance_factor,
        latent_state_free_energy_grid)
    lf_latent_state_freenergy_mean = np.mean(lf_latent_state_freenergy_bootstrap, axis=0)
    lf_latent_state_freenergy_std = np.std(lf_latent_state_freenergy_bootstrap, axis=0)
    lf_latent_state_freenergy_std = lf_latent_state_freenergy_std / np.sqrt(n_splits_2)

    tf_latent_state_freenergy_bootstrap = utils_latent_analysis.calculateLatentFreeEnergyWithUncertainty(
        tf_latent_states_all, n_splits_3, tf_covariance_factor,
        latent_state_free_energy_grid)
    tf_latent_state_freenergy_mean = np.mean(tf_latent_state_freenergy_bootstrap, axis=0)
    tf_latent_state_freenergy_std = np.std(tf_latent_state_freenergy_bootstrap, axis=0)
    tf_latent_state_freenergy_bootstrap = tf_latent_state_freenergy_bootstrap / np.sqrt(n_splits_3)


    fig, ax = plt.subplots()

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
    plt.fill_between(latent_state_free_energy_grid, tf_latent_state_freenergy_mean-tf_latent_state_freenergy_std, tf_latent_state_freenergy_mean+tf_latent_state_freenergy_std, alpha=0.4, color="g")

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
               bbox_to_anchor=(0.5, 1.425),
               borderaxespad=0.,
               frameon=False)
    plt.savefig(model.getFigureDir() +
                "/Comparison_latent_dynamics_calibrated_bootstrap_{:}_legend.{:}".format(
                    set_name, FIGTYPE),
                bbox_inches="tight",
                dpi=300)
    plt.close()

    latent_state_regions = tf_free_energy_latent_state_regions

    # tf_covariance_factor = 5.0
    # lf_covariance_factor = 5.0

    # print(tf_covariance_factor)
    # print(lf_covariance_factor)

    covariance_factor = tf_covariance_factor
    max_ = np.max([np.max(tf_latent_states_all), np.max(lf_latent_states_all)])
    min_ = np.min([np.min(tf_latent_states_all), np.min(lf_latent_states_all)])
    latent_states_flatten_range = max_ - min_
    latent_state_free_energy_grid = np.linspace(min_, max_, 100)

    tf_latent_states_flatten = tf_latent_states_all.flatten()
    tf_latent_state_freenergy = utils_latent_analysis.calculateLatentFreeEnergy(
        tf_latent_states_flatten, covariance_factor,
        latent_state_free_energy_grid)

    lf_latent_states_flatten = lf_latent_states_all.flatten()
    lf_latent_state_freenergy = utils_latent_analysis.calculateLatentFreeEnergy(
        lf_latent_states_flatten, covariance_factor,
        latent_state_free_energy_grid)

    error = np.abs(tf_latent_state_freenergy - lf_latent_state_freenergy)
    error = error * error
    total_error = np.trapz(error, x=latent_state_free_energy_grid)
    total_error = np.sqrt(total_error)
    print("FREE ENERGY MEAN SQUARE ROOT ERROR {:}".format(total_error))

    file_path = model.getFigureDir(
    ) + "/{:}_latent_free_energy_RMSE_ERROR.{:}".format(set_name, "txt")

    with io.open(file_path, "w") as f:
        f.write("FREE ENERGY MEAN SQUARE ROOT ERROR {:}".format(total_error))
        f.write("\n")

    return tf_targets_all, tf_latent_states_all, latent_state_regions


def plotLatentTransitionTimes(model, results, set_name, testing_mode):

    cluster_labels = results["free_energy_latent_cluster_labels"]

    times_pred = results["latent_cluster_mean_passage_times"]

    file_path = model.getFigureDir(
    ) + "/{:}_{:}_latent_meta_stable_states_mean_passage_times.{:}".format(
        testing_mode, set_name, "txt")

    writeLatentTransitionTimesToFile(model, file_path, cluster_labels,
                                     times_pred)


def plotLatentDynamicsFreeEnergy(model, results, set_name, testing_mode):
    print("# plotLatentDynamicsFreeEnergy() #")
    latent_states_all = results["latent_states_all"]
    if np.shape(latent_states_all)[2] == 1:

        latent_state_free_energy_grid = results[
            "latent_state_free_energy_grid"]
        latent_states_flatten_range = results["latent_states_flatten_range"]
        latent_states_flatten = results["latent_states_all"]
        latent_states_flatten = np.array(latent_states_flatten).flatten()
        freenergy = results["latent_state_freenergy"]

        free_energy_latent_state_regions = results[
            "free_energy_latent_state_regions"]





        # freenergy_bootstrap = results["latent_state_freenergy_bootstrap"]
        # freenergy_bootstrap_mean = np.mean(freenergy_bootstrap, axis=0)
        # freenergy_bootstrap_std = np.std(freenergy_bootstrap, axis=0)
        # freenergy_bootstrap_train = results["latent_state_freenergy_bootstrap_train"]
        # freenergy_bootstrap_mean_train = np.mean(freenergy_bootstrap_train, axis=0)
        # freenergy_bootstrap_std_train = np.std(freenergy_bootstrap_train, axis=0)
        # fig, ax = plt.subplots()
        # plt.plot(latent_state_free_energy_grid, freenergy_bootstrap_mean, "g")
        # plt.fill_between(latent_state_free_energy_grid, freenergy_bootstrap_mean-freenergy_bootstrap_std, freenergy_bootstrap_mean+freenergy_bootstrap_std, alpha=0.4, color="g")
        # plt.plot(latent_state_free_energy_grid, freenergy_bootstrap_mean_train, "b")
        # plt.fill_between(latent_state_free_energy_grid, freenergy_bootstrap_mean_train-freenergy_bootstrap_std_train, freenergy_bootstrap_mean_train+freenergy_bootstrap_std_train, alpha=0.4, color="b")
        # plt.ylabel(r'$F/\kappa_B T $')
        # plt.xlabel(r'$z$')
        # fig.tight_layout()
        # plt.savefig(model.getFigureDir() +
        #             "/{:}_{:}_latent_dynamics_free_energy_bootstrap.{:}".format(
        #                 testing_mode, set_name, FIGTYPE),
        #             bbox_inches="tight",
        #             dpi=300)
        # plt.close()


        # Free energy profiles projected into the (slowest) latent space coordinate F(ψ1)=−kBT ln(p(ψ1))
        # Plotting F/kbT =  - ln(p(ψ1))
        fig, ax = plt.subplots()
        plt.plot(latent_state_free_energy_grid, freenergy, "g")
        plt.ylabel(r'$F/\kappa_B T $')
        plt.xlabel(r'$z$')
        fig.tight_layout()
        plt.savefig(model.getFigureDir() +
                    "/{:}_{:}_latent_dynamics_free_energy.{:}".format(
                        testing_mode, set_name, FIGTYPE),
                    bbox_inches="tight",
                    dpi=300)
        plt.close()

        fig, ax = plt.subplots()
        plt.plot(latent_state_free_energy_grid, freenergy, "g", linewidth=3)
        for region in free_energy_latent_state_regions:
            size = region[1] - region[0]
            idx_region = np.bitwise_and(
                latent_state_free_energy_grid > region[0],
                latent_state_free_energy_grid < region[1])
            idx_region = np.where(idx_region)[0]
            freenergy_min = np.min(freenergy[idx_region])
            margin = 0.05 * (np.max(freenergy) - np.min(freenergy))
            plt.errorbar((region[0] + region[1]) / 2,
                         freenergy_min - margin,
                         xerr=size,
                         color="k",
                         capsize=6,
                         capthick=3,
                         linewidth=3)
        plt.ylabel(r'$F/\kappa_B T $')
        plt.xlabel(r'$z$')
        fig.tight_layout()
        plt.savefig(model.getFigureDir() +
                    "/{:}_{:}_latent_dynamics_free_energy_regions.{:}".format(
                        testing_mode, set_name, FIGTYPE),
                    bbox_inches="tight",
                    dpi=300)
        plt.close()

    else:

        # latent_state_freenergy             = results["latent_state_freenergy"]
        # latent_state_free_energy_grid     = results["latent_state_free_energy_grid"]
        # latent_states_flatten_range     = results["latent_states_flatten_range"]

        # fig, ax = plt.subplots()
        # vmin = np.nanmin(latent_state_freenergy[latent_state_freenergy != -np.inf])
        # vmax = np.nanmax(latent_state_freenergy[latent_state_freenergy != +np.inf])
        # mp = ax.contourf(latent_state_free_energy_grid[0], latent_state_free_energy_grid[1], latent_state_freenergy.T, 60, cmap=plt.get_cmap("Reds_r"), levels=np.linspace(vmin, vmax, 60))
        # fig.colorbar(mp, ax=ax)
        # # plt.show()
        # plt.ylabel(r'PCA${}_2$')
        # plt.xlabel(r'PCA${}_1$')
        # plt.savefig(model.getFigureDir() + "/{:}_{:}_latent_dynamics_free_energy_regions.{:}".format(testing_mode, set_name, FIGTYPE), bbox_inches="tight", dpi=300)
        # plt.close()

        # fig, ax = plt.subplots()
        # sns.jointplot(x=data[:,0], y=data[:,1])
        # # plt.ylabel(r'PCA${}_2$')
        # # plt.xlabel(r'PCA${}_1$')
        # # plt.savefig(model.getFigureDir() + "/{:}_{:}_latent_dynamics_free_energy_regions_joint.{:}".format(testing_mode, set_name, FIGTYPE), bbox_inches="tight", dpi=300)
        # fig_path = model.getFigureDir() + "/{:}_{:}_latent_dynamics_free_energy_regions_joint.png".format(testing_mode, set_name)
        # plt.savefig(fig_path, bbox_inches="tight", dpi=300)
        # plt.close()

        latent_states_flatten = results["latent_states_flatten"]

        makeFreeEnergyPlot(model, testing_mode, set_name,
                           latent_states_flatten)
        makejointDistributionPlot(model, testing_mode, set_name,
                                  latent_states_flatten)


def makeFreeEnergyPlot(
    model,
    testing_mode,
    set_name,
    data,
    data_bounds=None,
    add_str="",
    vmin=None,
    vmax=None,
    freenergyx_max=None,
    freenergyy_max=None,
    cmap="gist_rainbow",  # 'Blues_r' "gist_rainbow" "Reds_r"
    gridpoints=20,
    covariance_factor_scalex=80.0,
    covariance_factor_scaley=40.0,
):
    print("# makeFreeEnergyPlot() #")
    from matplotlib import gridspec
    from matplotlib.colorbar import Colorbar

    datax = data[:, 0]
    datay = data[:, 1]

    # covariance_factor_scalex     = 80.0
    # covariance_factor_scaley     = 40.0
    # gridpoints                    = 20

    margin_density = 0.1

    if data_bounds is None:
        boundsx = [datax.min(), datax.max()]
    else:
        print("# boundsx found #")
        boundsx = data_bounds[0]

    data_rangex = boundsx[1] - boundsx[0]
    covariance_factorx = data_rangex / covariance_factor_scalex
    min_x = boundsx[0] - margin_density * data_rangex
    max_x = boundsx[1] + margin_density * data_rangex
    gridx = np.linspace(min_x, max_x, gridpoints)
    densityx = utils_latent_analysis.calculateGaussianKernelDensityEstimate(
        datax, covariance_factorx, gridx)

    if data_bounds is None:
        boundsy = [datay.min(), datay.max()]
    else:
        print("# boundsy found #")
        boundsy = data_bounds[1]

    data_rangey = boundsy[1] - boundsy[0]
    covariance_factory = data_rangey / covariance_factor_scaley
    min_y = boundsy[0] - margin_density * data_rangey
    max_y = boundsy[1] + margin_density * data_rangey
    gridy = np.linspace(min_y, max_y, gridpoints)
    densityy = utils_latent_analysis.calculateGaussianKernelDensityEstimate(
        datay, covariance_factory, gridy)

    latent_states_flatten_range = np.max(data, axis=0) - np.min(data, axis=0)

    grid_min = np.min(data, axis=0) - 0.05 * latent_states_flatten_range
    grid_max = np.max(data, axis=0) + 0.05 * latent_states_flatten_range

    if data_bounds is None:
        bounds = []
        for i in range(len(grid_min)):
            min_var = grid_min[i]
            max_var = grid_max[i]
            bounds.append([min_var, max_var])
    else:
        print("# bounds found #")
        bounds = []
        for i in range(len(grid_min)):
            min_var = data_bounds[i][0]
            max_var = data_bounds[i][1]
            bounds.append([min_var, max_var])

    nbins = gridpoints
    density, grid_centers = utils_statistics.get_density(
        data, gridpoints, bounds)

    # density_norm = utils_statistics.evaluate2DIntegral(density, grid_centers[0], grid_centers[1])
    # print(density_norm)

    min_value = data.min()
    max_value = data.max()
    sigmax = (grid_max[0] - grid_min[0]) / covariance_factor_scalex
    sigmay = (grid_max[1] - grid_min[1]) / covariance_factor_scaley

    import scipy.ndimage as ndimage
    density = ndimage.gaussian_filter(density, sigma=(sigmax, sigmay), order=0)
    density_norm = utils_statistics.evaluate2DIntegral(density,
                                                       grid_centers[0],
                                                       grid_centers[1])

    latent_state_freenergy = -np.log(density)

    latent_state_free_energy_grid = np.meshgrid(grid_centers[0],
                                                grid_centers[1])

    freenergyx = -np.log(densityx)
    freenergyy = -np.log(densityy)

    fig = plt.figure(1, figsize=(8, 8))
    axgrid = gridspec.GridSpec(4,
                               2,
                               height_ratios=[0.2, 0.8, 0.01, 0.04],
                               width_ratios=[0.8, 0.2])
    axmain = plt.subplot(axgrid[1, 0])
    axfirst = plt.subplot(axgrid[0, 0])
    axsecond = plt.subplot(axgrid[1, 1])
    axcolorbar = plt.subplot(axgrid[3, 0])
    axgrid.update(left=0.15,
                  right=0.95,
                  bottom=0.1,
                  top=0.93,
                  wspace=0.2,
                  hspace=0.35)

    datax_hex = np.log(datax)
    datay_hex = np.log(datay)
    datax_hex = datax
    datay_hex = datay

    if (vmin is None) or (vmax is None):
        vmin = np.nanmin(
            latent_state_freenergy[latent_state_freenergy != -np.inf])
        vmax = np.nanmax(
            latent_state_freenergy[latent_state_freenergy != +np.inf])
    else:
        latent_state_freenergy[latent_state_freenergy < vmin] = vmin

    mp = axmain.contourf(latent_state_free_energy_grid[0],
                         latent_state_free_energy_grid[1],
                         latent_state_freenergy.T,
                         40,
                         cmap=plt.get_cmap(cmap),
                         levels=np.linspace(vmin, vmax, 40))

    cb = Colorbar(ax=axcolorbar,
                  mappable=mp,
                  orientation='horizontal',
                  ticklocation='bottom',
                  label="$F/ \kappa_B T$")
    from matplotlib import ticker
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()

    # axfirst.fill_between(gridx, densityx, np.min(densityx), facecolor='blue', alpha=0.5)
    axfirst.plot(gridx, freenergyx, "blue", linewidth=2)
    axfirst.set_xlim(boundsx)

    # axsecond.fill_betweenx(gridy, 0.0, densityy, facecolor='blue', alpha=0.5)
    axsecond.plot(freenergyy, gridy, color='blue', linewidth=2)
    axsecond.set_ylim(boundsy)

    axfirst.spines['right'].set_visible(False)
    axfirst.spines['top'].set_visible(False)
    # axfirst.spines['bottom'].set_visible(False)
    # axfirst.get_xaxis().set_visible(False)
    if freenergyx_max is None: freenergyx_max = 1.1 * np.max(freenergyx)
    axfirst.set_ylim([0, freenergyx_max])
    # axfirst.set_ylim(ymin=0)
    # axfirst.axis('off')
    if add_str:
        # PCA data
        ylabel = "$ - \log p \, ( \mathbf{\\tilde{z}}_1 )$"
    else:
        ylabel = "$- \log p \, ( \mathbf{z}_1 )$"
    axfirst.set_ylabel(r"{:}".format(ylabel))

    axsecond.spines['right'].set_visible(False)
    axsecond.spines['top'].set_visible(False)
    # axsecond.spines['left'].set_visible(False)
    # axsecond.get_yaxis().set_visible(False)
    if freenergyy_max is None: freenergyy_max = 1.1 * np.max(freenergyy)
    axsecond.set_xlim([0, freenergyy_max])
    # axsecond.set_xlim(xmin=0)
    # axsecond.axis('off')
    if add_str:
        # PCA data
        xlabel = "$- \log p \, ( \mathbf{\\tilde{z}}_2 )$"
    else:
        xlabel = "$- \log p \, ( \mathbf{z}_2 )$"
    axsecond.set_xlabel(r"{:}".format(xlabel))

    if add_str:
        xlabel_axmain = "$\mathbf{\\tilde{z}}_1$"
        ylabel_axmain = "$\mathbf{\\tilde{z}}_2$"
    else:
        xlabel_axmain = "$\mathbf{z}_1$"
        ylabel_axmain = "$\mathbf{z}_2$"

    axmain.set_xlabel(r"{:}".format(xlabel_axmain))
    axmain.set_ylabel(r"{:}".format(ylabel_axmain))

    fig_path = model.getFigureDir(
    ) + "/latent_dynamics{:}_free_energy_{:}_{:}.{:}".format(
        add_str, set_name, testing_mode, FIGTYPE)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return vmin, vmax, freenergyx_max, freenergyy_max


def makejointDistributionPlot(model,
                              testing_mode,
                              set_name,
                              data,
                              add_str="",
                              data_bounds=None):
    print("# makejointDistributionPlot() #")
    from matplotlib import gridspec
    from matplotlib.colorbar import Colorbar

    datax = data[:, 0]
    datay = data[:, 1]

    # covariance_factor_scalex     = 300.0
    # covariance_factor_scaley     = 300.0
    # gridpoints                    = 100
    # hexbins                        = 30

    # covariance_factor_scalex     = 200.0
    # covariance_factor_scaley     = 100.0
    # gridpoints                    = 100
    # hexbins                        = 25

    covariance_factor_scalex = 80.0
    covariance_factor_scaley = 40.0
    gridpoints = 20
    hexbins = 20

    margin_density = 0.0

    if data_bounds is None:
        boundsx = [datax.min(), datax.max()]
    else:
        boundsx = data_bounds[0]

    data_rangex = boundsx[1] - boundsx[0]
    covariance_factorx = data_rangex / covariance_factor_scalex
    min_x = boundsx[0] - margin_density * data_rangex
    max_x = boundsx[1] - margin_density * data_rangex
    gridx = np.linspace(min_x, max_x, gridpoints)
    densityx = utils_latent_analysis.calculateGaussianKernelDensityEstimate(
        datax, covariance_factorx, gridx)

    if data_bounds is None:
        boundsy = [datay.min(), datay.max()]
    else:
        boundsy = data_bounds[1]

    data_rangey = boundsy[1] - boundsy[0]
    covariance_factory = data_rangey / covariance_factor_scaley
    min_y = boundsy[0] - margin_density * data_rangey
    max_y = boundsy[1] - margin_density * data_rangey
    gridy = np.linspace(min_y, max_y, gridpoints)
    densityy = utils_latent_analysis.calculateGaussianKernelDensityEstimate(
        datay, covariance_factory, gridy)

    # fig = plt.figure(1, figsize=(15,15))
    fig = plt.figure(1, figsize=(8, 8))

    # axgrid         = gridspec.GridSpec(3, 2, height_ratios=[0.2,1,0.04], width_ratios=[1,0.2])
    axgrid = gridspec.GridSpec(4,
                               2,
                               height_ratios=[0.2, 0.8, 0.01, 0.04],
                               width_ratios=[0.8, 0.2])

    axmain = plt.subplot(axgrid[1, 0])
    axfirst = plt.subplot(axgrid[0, 0])
    axsecond = plt.subplot(axgrid[1, 1])
    axcolorbar = plt.subplot(axgrid[3, 0])

    # axgrid.update(left=0.05, right=0.95, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)
    # axgrid.update(left=0.1, right=0.95, bottom=0.1, top=0.93, wspace=0.02, hspace=0.4)
    axgrid.update(left=0.15,
                  right=0.95,
                  bottom=0.1,
                  top=0.93,
                  wspace=0.2,
                  hspace=0.35)

    # axgrid         = gridspec.GridSpec(2, 2, width_ratios=[3,1], height_ratios=[1,4])
    # axmain         = plt.subplot(axgrid[1,0])
    # axuppper     = plt.subplot(axgrid[0,0])
    # axsecond     = plt.subplot(axgrid[1,1])

    # datax_hex = np.log(datax)
    # datay_hex = np.log(datay)
    datax_hex = datax
    datay_hex = datay
    # mp = axmain.hexbin(x=datax_hex, y=datay_hex, gridsize=hexbins, cmap='Reds', bins='log')
    mp = axmain.hexbin(x=datax_hex, y=datay_hex, gridsize=hexbins, cmap='Reds')
    axmain.set_xlim(boundsx)
    axmain.set_ylim(boundsy)
    # print(boundsx)
    # print(boundsy)

    cb = Colorbar(ax=axcolorbar,
                  mappable=mp,
                  orientation='horizontal',
                  ticklocation='bottom',
                  label="Counts")

    axfirst.fill_between(gridx,
                         densityx,
                         np.min(densityx),
                         facecolor='blue',
                         alpha=0.5)
    axfirst.plot(gridx, densityx, "blue", linewidth=2)
    axfirst.set_xlim(boundsx)

    axsecond.fill_betweenx(gridy, 0.0, densityy, facecolor='blue', alpha=0.5)
    axsecond.plot(densityy, gridy, color='blue', linewidth=2)
    axsecond.set_ylim(boundsy)

    axfirst.spines['right'].set_visible(False)
    axfirst.spines['top'].set_visible(False)
    # axfirst.spines['bottom'].set_visible(False)
    # axfirst.get_xaxis().set_visible(False)
    axfirst.set_ylim(ymin=0)
    # axfirst.axis('off')
    if add_str:
        # PCA data
        ylabel = "$p \, ( \mathbf{\\tilde{z}}_1 )$"
    else:
        ylabel = "$p \, ( \mathbf{z}_1 )$"
    axfirst.set_ylabel(r"{:}".format(ylabel))

    axsecond.spines['right'].set_visible(False)
    axsecond.spines['top'].set_visible(False)
    # axsecond.spines['left'].set_visible(False)
    # axsecond.get_yaxis().set_visible(False)
    axsecond.set_xlim(xmin=0)
    # axsecond.axis('off')
    if add_str:
        # PCA data
        xlabel = "$p \, ( \mathbf{\\tilde{z}}_2 )$"
    else:
        xlabel = "$p \, ( \mathbf{z}_2 )$"
    axsecond.set_xlabel(r"{:}".format(xlabel))

    if add_str:
        xlabel_axmain = "$\mathbf{\\tilde{z}}_1$"
        ylabel_axmain = "$\mathbf{\\tilde{z}}_2$"
    else:
        xlabel_axmain = "$\mathbf{z}_1$"
        ylabel_axmain = "$\mathbf{z}_2$"

    axmain.set_xlabel(r"{:}".format(xlabel_axmain))
    axmain.set_ylabel(r"{:}".format(ylabel_axmain))

    fig_path = model.getFigureDir(
    ) + "/latent_dynamics{:}_distr_joint_{:}_{:}.{:}".format(
        add_str, set_name, testing_mode, FIGTYPE)
    plt.savefig(fig_path, dpi=300)
    plt.close()


def arrowed_spines(ax=None, arrow_length=20, labels=('', ''), arrowprops=None):
    xlabel, ylabel = labels
    if ax is None:
        ax = plt.gca()
    if arrowprops is None:
        # arrowprops = dict(arrowstyle='<|-', facecolor='black')
        arrowprops = dict(arrowstyle='-|>', facecolor='black')

    for i, spine in enumerate(['left', 'bottom']):
        # Set up the annotation parameters
        t = ax.spines[spine].get_transform()
        xy, xycoords = [1, 0], ('axes fraction', t)
        xytext, textcoords = [arrow_length, 0], ('offset points', t)
        ha, va = 'left', 'bottom'

        # If axis is reversed, draw the arrow the other way
        top, bottom = ax.spines[spine].axis.get_view_interval()
        if top < bottom:
            xy[0] = 0
            xytext[0] *= -1
            ha, va = 'right', 'top'

        if spine == 'bottom':
            xarrow = ax.annotate(xlabel,
                                 xy,
                                 xycoords=xycoords,
                                 xytext=xytext,
                                 textcoords=textcoords,
                                 ha=ha,
                                 va='center',
                                 arrowprops=arrowprops)
        else:
            yarrow = ax.annotate(ylabel,
                                 xy[::-1],
                                 xycoords=xycoords[::-1],
                                 xytext=xytext[::-1],
                                 textcoords=textcoords[::-1],
                                 ha='center',
                                 va=va,
                                 arrowprops=arrowprops)
    return xarrow, yarrow


def getMeanTransitionTimesFileLines(cluster_labels,
                                    transition_times,
                                    nstates,
                                    order=None):
    # print("# getMeanTransitionTimesFileLines()")
    lines = []
    if order is None: order = np.arange(nstates)
    lines.append("-" * 100)
    line_f = " " * 10 + "{:8s}, " * nstates
    line_f = line_f.format(
        *[cluster_labels[order[ii]] for ii in range(nstates)])
    # print(line_f)
    lines.append(line_f)
    line_title_base = "{:8s}, "
    line_inhalt_base = ""
    for ii in range(nstates):
        i = order[ii]
        jlist = range(nstates)
        # jlist = [order[jj] for jj in jlist if order[jj] != i]
        jlist = [order[jj] for jj in jlist]
        line_inhalt = line_inhalt_base
        for j in jlist:
            if i == j:
                mtt = 0.0
            else:
                mtt = np.mean(transition_times[tuple((i, j))])
            # line_inhalt += "{:04.4f}, ".format(mtt)
            line_inhalt += "{:0>8f}, ".format(mtt)

        line_title = line_title_base.format(cluster_labels[i])
        # line_inhalt = line_inhalt_base.format(*[np.mean(transition_times[tuple((i,j))]) for j in jlist])
        line_f = line_title + line_inhalt
        # print(line_f)
        lines.append(line_f)
    return lines


def writeLatentTransitionTimesToFile(
    model,
    file_path,
    cluster_labels,
    times_pred,
):
    nstates = len(cluster_labels)

    lines_pred = getMeanTransitionTimesFileLines(cluster_labels, times_pred,
                                                 nstates)
    with io.open(file_path, "w") as f:
        f.write("PREDICTION")
        f.write("\n")
        for line in lines_pred:
            f.write(line)
            f.write("\n")
        f.write("\n")
