#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import numpy as np
import os
import time
import pyemma

# UTILITIES
from ... import Utils as utils

import rmsd


def getRamaClusterLabelsAlanineSimpleFormat():
    cluster_labels = {
        0: "C_5",
        1: "P_II",
        2: "alpha_R",
        3: "alpha_L",
        4: "C_7ax",
    }
    return cluster_labels


def getRamaClusterLabelsAlanine():
    cluster_labels = {
        0: "C_5",
        1: "P_{II}",
        2: "\\alpha_R",
        3: "\\alpha_L",
        4: "C_7^{ax}",
        # 5:"\\alpha_R^{\prime \prime}",
        # None:"None",
    }
    return cluster_labels


def getRamaClustersOverWhichToComputeMFPTError():
    # return [0,1,2,3]
    return [0, 1, 2]


def getRamaClustersOverWhichToComputeMFPTErrorKeys():
    n_clusters_red = len(getRamaClustersOverWhichToComputeMFPTError())
    keys = []
    for i in range(n_clusters_red):
        for j in range(n_clusters_red):
            keys.append(tuple((i, j)))
    return keys


def getRamaClusterCentersAlanine():
    cluster_centers = {
        0: [np.array([-155, 155])],  # cluster_label = "C_5"
        1: [np.array([-75, 150])],  # cluster_label = "P_{II}"
        2: [np.array([-75, -20])],  # cluster_label = "\\alpha_R"
        3: [np.array([67, 5])],  # cluster_label = "\\alpha_L"
        4: [np.array([70, 160]),
            np.array([70, -160])],  # cluster_label = "C_7^{ax}"
        # 5:np.array([-150, -10]), # cluster_label = "\\alpha_R^{\prime \prime}"
        # None:"None",
    }
    return cluster_centers


def isInCircle(phi, psi, center, radius):
    possible_states = [
        np.array([phi, psi]),
        np.array([phi - 360, psi]),
        np.array([phi, psi - 360]),
        np.array([phi - 360, psi - 360]),
        np.array([phi + 360, psi]),
        np.array([phi, psi + 360]),
        np.array([phi + 360, psi + 360]),
    ]
    for state in possible_states:
        if np.linalg.norm(state - center) < radius:
            return True
    return False


def getRamaClusterAlanine(phi, psi, radius):
    cluster = None
    cluster_centers = getRamaClusterCentersAlanine()
    cluster_labels = getRamaClusterLabelsAlanine()
    for cluster_id, label in cluster_labels.items():
        centers = cluster_centers[cluster_id]
        for center in centers:
            if isInCircle(phi, psi, center, radius): cluster = cluster_id
    return cluster


def clusterTrajectoryAlanine(trajBAD, radius=10):
    traj_clustered = []
    # Convert radians to angles
    psi = trajBAD[:, 20] * 180 / np.pi
    phi = trajBAD[:, 21] * 180 / np.pi
    for t in range(np.shape(trajBAD)[0]):
        phi_t = phi[t]
        psi_t = psi[t]
        cluster = getRamaClusterAlanine(phi_t, psi_t, radius)
        traj_clustered.append(cluster)
    traj_clustered = np.array(traj_clustered)
    return traj_clustered


def estimateRamaClusterMeanPassageTimesAlanine(trajectories,
                                               dt,
                                               is_abd=False,
                                               radius=20,
                                               dt_msm=10 * 1e-12,
                                               dt_save=1e-9,
                                               clustered_trajectories=None,
                                               print_=True,
                                               ):
    ###############################################
    ####### Estimating the mean_passage times
    ###############################################
    print("# estimateRamaClusterMeanPassageTimesAlanine() #")
    time_start = time.time()

    cluster_labels = getRamaClusterLabelsAlanine()
    n_clusters = len(cluster_labels)

    print("n_clusters = {:}".format(n_clusters))
    cluster_mean_passage_times = {}
    for i in range(n_clusters):
        for j in range(n_clusters):
            # mean_passage time from cluster i to cluster j
            # if i != j: cluster_mean_passage_times.update({tuple((i,j)):[]})
            if i != j: cluster_mean_passage_times.update({tuple((i, j)): 0.0})

    if clustered_trajectories is None:
        # Estimating the mean_passage times
        clustered_trajectories = []
        for traj_num in range(len(trajectories)):
            traj = trajectories[traj_num]
            print("traj = {:}/{:}".format(traj_num, len(trajectories)))

            if not is_abd:
                raise ValueError("Not implemented.")
                print("# transformToBAD()")
                traj_BAD = transformToBAD(traj, "")
            else:
                traj_BAD = traj.copy()
            traj_clustered = clusterTrajectoryAlanine(traj_BAD, radius=radius)
            clustered_trajectories.append(traj_clustered)

    clustered_trajectories = np.array(clustered_trajectories)
    # Adding an auxiliary cluster (equal to None)
    clustered_trajectories[clustered_trajectories == None] = int(n_clusters)
    # print(np.shape(clustered_trajectories))
    # print(ark)

    # subsample = 10
    subsample = 1
    dt_traj = subsample * dt

    # Lag-time 10ps
    # dt_msm = 10*1e-12
    msm_lag = int(np.rint(dt_msm / dt_traj))
    print("Lagtime of the MSM = {:}".format(dt_msm))
    print("Lagtimesteps for fitting the MSM = {:}".format(msm_lag))

    traj_clustered_msm = []
    for clustered_traj in clustered_trajectories:
        temp = clustered_traj.copy()
        temp = temp[::subsample]
        traj_clustered_msm.append(temp.astype(int))

    # In which timestep to save the MTT (nanoseconds)
    # dt_save = 1e-9 # ns
    save_lag = int(dt_save / dt_traj)

    assert (int(dt_save % dt_traj) == 0)

    try:
        # Fitting the MSM
        msm = pyemma.msm.estimate_markov_model(traj_clustered_msm,
                                               msm_lag,
                                               count_mode='sample')
        # msm = pyemma.msm.bayesian_markov_model(traj_clustered_msm, msm_lag, count_mode='sample', reversible=False)

        msm_active_set = msm.active_set

        # print("##########################################")
        # print("transition_matrix")
        # T = msm.transition_matrix
        # print(T)
        # print("##########################################")

        # print("MSM active set:")
        # print(msm.active_set)
        # print("count_matrix_active:")
        # print(msm.count_matrix_active)

        # One auxiliary cluster for states that do not belong to any cluster
        for cluster_start in range(n_clusters):
            for cluster_end in range(n_clusters):
                if cluster_start != cluster_end:
                    # print("Estimating MPT between clusters: {:} -> {:}".format(cluster_start, cluster_end))
                    if (cluster_start
                            in msm_active_set) and (cluster_end
                                                    in msm_active_set):
                        cluster_start_index_in_active_set = np.where(
                            msm.active_set == cluster_start)[0][0]
                        cluster_end_index_in_active_set = np.where(
                            msm.active_set == cluster_end)[0][0]
                        mtt = msm.mfpt(cluster_start_index_in_active_set,
                                       cluster_end_index_in_active_set)
                    else:
                        mtt = 0.0

                    mtt_save = mtt / save_lag

                    cluster_mean_passage_times[tuple(
                        (cluster_start, cluster_end))] = mtt_save
    except Exception as inst:
        print(inst)
        for cluster_start in range(n_clusters):
            for cluster_end in range(n_clusters):
                if cluster_start != cluster_end:
                    mtt = 0.0
                    mtt_save = mtt / save_lag
                    cluster_mean_passage_times[tuple(
                        (cluster_start, cluster_end))] = mtt_save

    # print(cluster_mean_passage_times)
    if print_: utils.printTransitionTimes(cluster_labels, cluster_mean_passage_times,
                               n_clusters)

    time_end = time.time()
    utils.printTime(time_end - time_start)

    # print(cluster_mean_passage_times)
    # print(ark)

    # Multiplying by dt
    # for key in cluster_mean_passage_times:
    #     times_ = cluster_mean_passage_times[key]
    #     times_ = [float(time) * dt for time in times_]
    #     cluster_mean_passage_times[key] = times_

    clustered_trajectories = np.array(clustered_trajectories)
    # print(cluster_mean_passage_times)
    return cluster_mean_passage_times, clustered_trajectories


def computeStateDistributionStatisticsSystemAlanine(state_dist_statistics,
                                                    targets_all,
                                                    predictions_all):
    assert (len(np.shape(targets_all)) == 3)
    n_ics, T, D = np.shape(targets_all)
    errors = []
    targets = []
    density_targs = []
    predictions = []
    density_preds = []
    bin_centers = []
    for ic in range(n_ics):
        traj_targ = targets_all[ic]
        traj_pred = predictions_all[ic]

        # TODO: ABD

        # traj_targ = transformToBAD(traj_targ, "")
        # traj_pred = transformToBAD(traj_pred, "")

        phi_targ = traj_targ[:, 21] * 180 / np.pi
        psi_targ = traj_targ[:, 20] * 180 / np.pi

        # createRamachandranPlot(phi_targ, psi_targ)
        phi_targ = phi_targ[:, np.newaxis]
        psi_targ = psi_targ[:, np.newaxis]
        targets_ = np.concatenate((phi_targ, psi_targ), axis=1)

        phi_pred = traj_pred[:, 21] * 180 / np.pi
        psi_pred = traj_pred[:, 20] * 180 / np.pi
        # createRamachandranPlot(phi_pred, psi_pred)
        phi_pred = phi_pred[:, np.newaxis]
        psi_pred = psi_pred[:, np.newaxis]
        predictions_ = np.concatenate((phi_pred, psi_pred), axis=1)
        nbins = 50
        bounds = [[np.min(phi_targ), np.max(phi_targ)],
                  [np.min(psi_targ), np.max(psi_targ)]]
        error, error_vec, density_targ, density_pred, bin_centers_ = utils.evaluateL1HistErrorVector(
            targets_, predictions_, nbins, bounds)
        errors.append(error)
        predictions.append(predictions_)
        targets.append(targets_)
        density_targs.append(density_targ)
        density_preds.append(density_pred)
        bin_centers.append(bin_centers_)
    errors = np.array(errors)
    density_targs = np.array(density_targs)
    density_preds = np.array(density_preds)
    predictions = np.array(predictions)
    targets = np.array(targets)
    bin_centers = np.array(bin_centers)

    error = np.mean(errors)
    # MEAN OVER ICs
    density_targ = np.mean(density_targs, axis=0)
    density_pred = np.mean(density_preds, axis=0)
    bin_centers = np.mean(bin_centers, axis=0)
    assert ("rama_density_target" not in state_dist_statistics)
    assert ("rama_density_predicted" not in state_dist_statistics)
    assert ("rama_l1_hist_error" not in state_dist_statistics)
    assert ("rama_mesh" not in state_dist_statistics)
    assert ("rama_predictions" not in state_dist_statistics)
    assert ("rama_targets" not in state_dist_statistics)
    state_dist_statistics.update({
        "rama_predictions": predictions,
        "rama_targets": targets,
        "rama_density_target": density_targ,
        "rama_density_predicted": density_pred,
        "rama_bin_centers": bin_centers,
        "rama_l1_hist_error": error,
    })
    return state_dist_statistics


def transitionTimesTrain(model, n_splits=16):
    testing_mode = "teacher_forcing_forecasting"
    # Loading the results
    data_path = model.getResultsDir(
    ) + "/results_{:}_{:}".format(testing_mode, "train")
    results_train = utils.loadData(data_path, model.save_format)
    targets_all_train = results_train["targets_all"]
    dt = results_train["dt"]
    del results_train
    
    is_abd = True
    dt_msm = 10 * 1e-12
    dt_save = 1e-9
    # nstates = 5
    # dt_msm_small = 1 * 1e-12

    # targets_all_train = targets_all_train[:10]

    print("# TRAIN MFPT")
    print(np.shape(targets_all_train))
    # n_splits = 8
    # n_splits = 16
    # n_splits = 32
    cluster_transition_times_train_mean, cluster_transition_times_train_std = getTransitionTimesWithUncertainty(targets_all_train, n_splits, is_abd, dt, dt_msm, dt_save)

    # cluster_labels = getRamaClusterLabelsAlanine()
    # print("MEAN (TRAIN)")
    # utils.printTransitionTimes(cluster_labels, cluster_transition_times_train_mean, nstates)
    # print("STD (TRAIN)")
    # utils.printTransitionTimes(cluster_labels, cluster_transition_times_train_std, nstates)

    data = {
    "cluster_transition_times_train_mean":cluster_transition_times_train_mean,
    "cluster_transition_times_train_std":cluster_transition_times_train_std,
    }
    data_path = model.getResultsDir(
    ) + "/results_mfpt_train_{:}".format(n_splits)
    utils.saveData(data, data_path, "pickle")
    return 0


def transitionTimesTrainVal(model, n_splits=16):
    testing_mode = "teacher_forcing_forecasting"
    # Loading the results
    data_path = model.getResultsDir(
    ) + "/results_{:}_{:}".format(testing_mode, "train")
    results_train = utils.loadData(data_path, model.save_format)
    targets_all_train = results_train["targets_all"]
    dt = results_train["dt"]
    del results_train
    
    # Loading the results
    data_path = model.getResultsDir(
    ) + "/results_{:}_{:}".format(testing_mode, "val")
    results2 = utils.loadData(data_path, model.save_format)
    targets_all_val = results2["targets_all"]
    targets_all_train = np.concatenate((targets_all_train, targets_all_val), axis=0)
    del results2

    is_abd = True
    dt_msm = 10 * 1e-12
    dt_save = 1e-9
    # nstates = 5
    # dt_msm_small = 1 * 1e-12

    # targets_all_train = targets_all_train[:10]

    print("# TRAIN and VAL MFPT")
    print(np.shape(targets_all_train))
    # n_splits = 8
    # n_splits = 16
    # n_splits = 32
    cluster_transition_times_train_val_mean, cluster_transition_times_train_val_std = getTransitionTimesWithUncertainty(targets_all_train, n_splits, is_abd, dt, dt_msm, dt_save)

    # cluster_labels = getRamaClusterLabelsAlanine()
    # print("MEAN (TRAIN)")
    # utils.printTransitionTimes(cluster_labels, cluster_transition_times_train_mean, nstates)
    # print("STD (TRAIN)")
    # utils.printTransitionTimes(cluster_labels, cluster_transition_times_train_std, nstates)

    data = {
    "cluster_transition_times_train_val_mean":cluster_transition_times_train_val_mean,
    "cluster_transition_times_train_val_std":cluster_transition_times_train_val_std,
    }
    data_path = model.getResultsDir(
    ) + "/results_mfpt_train_and_val_{:}".format(n_splits)
    utils.saveData(data, data_path, "pickle")
    return 0

def transitionTimesPredTest(model, n_splits=16):

    testing_mode = "iterative_latent_forecasting"

    # Loading the results
    data_path = model.getResultsDir(
    ) + "/results_{:}_{:}".format(testing_mode, "test")
    results_lf = utils.loadData(data_path, model.save_format)
    predictions_all_lf = results_lf["predictions_all"]
    dt = results_lf["dt"]

    for random_seed in [10, 20, 30, 40]:
        # Loading the results
        data_path = model.getResultsDir(
        ) + "-RS_{:}".format(random_seed) + "/results_{:}_{:}".format(testing_mode, "test")
        results_lf = utils.loadData(data_path, model.save_format)
        predictions_all_lf_ = results_lf["predictions_all"]
        del results_lf
        predictions_all_lf = np.concatenate((predictions_all_lf, predictions_all_lf_), axis=0)

    # take into account only the first 1232 trajectories (has to be divisible by bootstrap=16)
    is_abd = True
    dt_msm = 10 * 1e-12
    dt_save = 1e-9

    # predictions_all_lf = predictions_all_lf[:16]

    # total: 1240
    # n_splits = 8
    # predictions_all_lf = predictions_all_lf[:1232]
    # n_splits = 16
    # predictions_all_lf = predictions_all_lf[:1216]
    # n_splits = 32
    # predictions_all_lf = predictions_all_lf[:1216]
    
    n_traj = np.shape(predictions_all_lf)[0]
    to_remove = n_traj % n_splits
    predictions_all_lf = predictions_all_lf[:-to_remove or None]
    print("# PREDICTION MFPT")
    print(np.shape(predictions_all_lf))
    cluster_transition_times_pred_mean, cluster_transition_times_pred_std = getTransitionTimesWithUncertainty(predictions_all_lf, n_splits, is_abd, dt, dt_msm, dt_save)

    # print("MEAN (TEST)")
    # utils.printTransitionTimes(cluster_labels, cluster_transition_times_pred_mean, nstates)
    # print("STD (TEST)")
    # utils.printTransitionTimes(cluster_labels, cluster_transition_times_pred_std, nstates)

    data = {
    "cluster_transition_times_pred_mean":cluster_transition_times_pred_mean,
    "cluster_transition_times_pred_std":cluster_transition_times_pred_std,
    }
    data_path = model.getResultsDir(
    ) + "/results_mfpt_pred_test_large_{:}".format(n_splits)
    utils.saveData(data, data_path, "pickle")
    return 0


def transitionTimesPredTrain(model, n_splits=16):

    testing_mode = "iterative_latent_forecasting"

    # Loading the results
    data_path = model.getResultsDir(
    ) + "/results_{:}_{:}".format(testing_mode, "test")
    results_lf = utils.loadData(data_path, model.save_format)
    predictions_all_lf = results_lf["predictions_all"]
    dt = results_lf["dt"]

    # take into account only the first 96 trajectories (training data)
    predictions_all_lf = predictions_all_lf[:96]
    is_abd = True
    dt_msm = 10 * 1e-12
    dt_save = 1e-9

    # predictions_all_lf = predictions_all_lf[:16]

    # n_splits = 8
    # n_splits = 16
    # n_splits = 32
    print("# PREDICTION on TRAIN MFPT")
    print(np.shape(predictions_all_lf))
    cluster_transition_times_pred_train_mean, cluster_transition_times_pred_train_std = getTransitionTimesWithUncertainty(predictions_all_lf, n_splits, is_abd, dt, dt_msm, dt_save)

    data = {
    "cluster_transition_times_pred_train_mean":cluster_transition_times_pred_train_mean,
    "cluster_transition_times_pred_train_std":cluster_transition_times_pred_train_std,
    }
    data_path = model.getResultsDir(
    ) + "/results_mfpt_pred_train_{:}".format(n_splits)
    utils.saveData(data, data_path, "pickle")
    return 0



def transitionTimesReference(model):
    testing_mode = "iterative_latent_forecasting"

    # Loading the results
    data_path = model.getResultsDir(
    ) + "/results_{:}_{:}".format(testing_mode, "test")
    results_lf = utils.loadData(data_path, model.save_format)
    targets_all_lf = results_lf["targets_all"]
    dt = results_lf["dt"]

    is_abd = True
    dt_msm = 10 * 1e-12
    dt_save = 1e-9

    n_splits = 1
    print("# TRUE MFPT")
    print(np.shape(targets_all_lf))
    cluster_transition_times_target_mean, cluster_transition_times_target_std = getTransitionTimesWithUncertainty(targets_all_lf, n_splits, is_abd, dt, dt_msm, dt_save)

    data = {
    "cluster_transition_times_target_mean":cluster_transition_times_target_mean,
    "cluster_transition_times_target_std":cluster_transition_times_target_std,
    }
    data_path = model.getResultsDir(
    ) + "/results_mfpt_reference"
    utils.saveData(data, data_path, "pickle")
    return 0

def transitionTimesReferenceSmallDt(model):
    testing_mode = "iterative_latent_forecasting"

    # Loading the results
    data_path = model.getResultsDir(
    ) + "/results_{:}_{:}".format(testing_mode, "test")
    results_lf = utils.loadData(data_path, model.save_format)
    targets_all_lf = results_lf["targets_all"]
    dt = results_lf["dt"]

    is_abd = True
    dt_msm_small = 1 * 1e-12
    dt_save = 1e-9

    n_splits = 1
    print("# TRUE MFPT (small dt)")
    print(np.shape(targets_all_lf))
    cluster_transition_times_smalldt_mean, cluster_transition_times_smalldt_std = getTransitionTimesWithUncertainty(targets_all_lf, n_splits, is_abd, dt, dt_msm_small, dt_save)

    data = {
    "cluster_transition_times_smalldt_mean":cluster_transition_times_smalldt_mean,
    "cluster_transition_times_smalldt_std":cluster_transition_times_smalldt_std,
    }
    data_path = model.getResultsDir() + "/results_mfpt_smalldt"
    utils.saveData(data, data_path, "pickle")
    return 0

def addResultsSystemAlanineTrainingDataAnalysis(model):

    data_path = model.getResultsDir() + "/results_mfpt_reference"
    results = utils.loadData(data_path, model.save_format)
    cluster_transition_times_target_mean = results["cluster_transition_times_target_mean"]
    cluster_transition_times_target_std = results["cluster_transition_times_target_std"]
    del results

    data_path = model.getResultsDir() + "/results_mfpt_smalldt"
    results = utils.loadData(data_path, model.save_format)
    cluster_transition_times_smalldt_mean = results["cluster_transition_times_smalldt_mean"]
    cluster_transition_times_smalldt_std = results["cluster_transition_times_smalldt_std"]
    del results

    n_splits = 8
    data_path = model.getResultsDir() + "/results_mfpt_train_{:}".format(n_splits)
    results = utils.loadData(data_path, model.save_format)
    cluster_transition_times_train_mean = results["cluster_transition_times_train_mean"]
    cluster_transition_times_train_std = results["cluster_transition_times_train_std"]
    del results

    # n_splits = 16
    # data_path = model.getResultsDir() + "/results_mfpt_train_and_val_{:}".format(n_splits)
    # results = utils.loadData(data_path, model.save_format)
    # cluster_transition_times_train_val_mean = results["cluster_transition_times_train_val_mean"]
    # cluster_transition_times_train_val_std = results["cluster_transition_times_train_val_std"]
    # del results


    n_splits = 32
    data_path = model.getResultsDir() + "/results_mfpt_pred_test_large_{:}".format(n_splits)
    results = utils.loadData(data_path, model.save_format)
    cluster_transition_times_pred_mean = results["cluster_transition_times_pred_mean"]
    cluster_transition_times_pred_std = results["cluster_transition_times_pred_std"]
    del results

    # n_splits = 16
    # data_path = model.getResultsDir() + "/results_mfpt_pred_train_{:}".format(n_splits)
    # results = utils.loadData(data_path, model.save_format)
    # cluster_transition_times_pred_train_mean = results["cluster_transition_times_pred_train_mean"]
    # cluster_transition_times_pred_train_std = results["cluster_transition_times_pred_train_std"]
    # del results

    cluster_transitions_considered_in_the_error = getRamaClustersOverWhichToComputeMFPTErrorKeys()

    cluster_transition_times_smalldt_errors, _ = utils.computeErrorOnTimes(cluster_transition_times_target_mean, cluster_transition_times_smalldt_mean, cluster_transitions_considered_in_the_error)
    cluster_transition_times_train_errors, _ = utils.computeErrorOnTimes(cluster_transition_times_target_mean, cluster_transition_times_train_mean, cluster_transitions_considered_in_the_error)

    # cluster_transition_times_train_val_errors, _ = utils.computeErrorOnTimes(cluster_transition_times_target_mean, cluster_transition_times_train_val_mean, cluster_transitions_considered_in_the_error)

    cluster_transition_times_pred_errors, _ = utils.computeErrorOnTimes(cluster_transition_times_target_mean, cluster_transition_times_pred_mean, cluster_transitions_considered_in_the_error)
    # cluster_transition_times_pred_train_errors, _ = utils.computeErrorOnTimes(cluster_transition_times_target_mean, cluster_transition_times_pred_train_mean, cluster_transitions_considered_in_the_error)



    cluster_labels = getRamaClusterLabelsAlanine()

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

    times_target = cluster_transition_times_target_mean

    times_smalldt = cluster_transition_times_smalldt_mean
    times_smalldt_std = cluster_transition_times_smalldt_std
    times_smalldt_errors = cluster_transition_times_smalldt_errors

    times_pred = cluster_transition_times_pred_mean
    times_pred_std = cluster_transition_times_pred_std
    times_pred_errors = cluster_transition_times_pred_errors

    # times_pred_train = cluster_transition_times_pred_train_mean
    # times_pred_train_std = cluster_transition_times_pred_train_std
    # times_pred_train_errors = cluster_transition_times_pred_train_errors

    time_train = cluster_transition_times_train_mean
    times_train_std = cluster_transition_times_train_std
    times_train_errors = cluster_transition_times_train_errors

    # time_train_val = cluster_transition_times_train_val_mean
    # times_train_val_std = cluster_transition_times_train_val_std
    # times_train_val_errors = cluster_transition_times_train_val_errors


    file_path = model.getFigureDir(
    ) + "/rama_clusters_mean_passage_times_ANALYSIS_TEST_LATEX.{:}".format("txt")
    writeLatentTransitionTimesToLatexTableFile(model,
                                               file_path,
                                               cluster_labels,
                                               transitions_to_print,
                                               times_target,
                                               times_smalldt,
                                               times_smalldt_std,
                                               times_smalldt_errors,
                                               times_pred,
                                               times_pred_std,
                                               times_pred_errors,
                                               # times_pred_train,
                                               # times_pred_train_std,
                                               # times_pred_train_errors,
                                               time_train,
                                               times_train_std,
                                               times_train_errors,
                                               # time_train_val,
                                               # times_train_val_std,
                                               # times_train_val_errors,
                                               )

    return 0

import io
def writeLatentTransitionTimesToLatexTableFile(
    model,
    file_path,
    cluster_labels,
    transitions_to_print,
    times_target,
    times_smalldt,
    times_smalldt_std,
    times_smalldt_error,
    times_pred,
    times_pred_std,
    times_pred_error,
    # times_pred_train,
    # times_pred_train_std,
    # times_pred_train_error,
    time_train,
    times_train_std,
    times_train_error,
    # time_train_val,
    # times_train_val_std,
    # times_train_val_error,
):
    nstates = len(cluster_labels)
    lines = []
    errors_relative_smalldt = []
    errors_relative_pred = []
    # errors_relative_pred_train = []
    errors_relative_train = []
    # errors_relative_train_val = []
    for i in range(nstates):
        for j in range(nstates):
            if i != j:
                if tuple((i, j)) in transitions_to_print:
                    transition_str_i = "{:}".format(cluster_labels[i])
                    transition_str_j = "{:}".format(cluster_labels[j])
                    transition_str = "$T_{" + transition_str_i + " \\to " + transition_str_j + "}$"
                    transition_str += " &"

                    value = times_target[tuple((i, j))]
                    value_str = " ${:.3f}$".format(value)
                    transition_str += str(value_str)
                    transition_str += " &"

                    value = times_smalldt[tuple((i, j))]
                    value_str = " ${:.3f}$".format(value)
                    transition_str += str(value_str)
                    transition_str += " &"

                    value = times_smalldt_error[tuple((i, j))]
                    errors_relative_smalldt.append(value)
                    value_str = " ${:2.0f}$".format(value * 100.0)
                    transition_str += str(value_str)
                    transition_str += " &"

                    value = time_train[tuple((i, j))]
                    value_str = " ${:.3f}".format(value)
                    transition_str += str(value_str)
                    value = times_train_std[tuple((i, j))]
                    value_str = " \pm {:.3f}$".format(value)
                    transition_str += str(value_str)
                    transition_str += " &"

                    value = times_train_error[tuple((i, j))]
                    errors_relative_train.append(value)
                    value_str = " ${:2.0f}$".format(value * 100.0)
                    transition_str += str(value_str)
                    transition_str += " &"

                    value = times_pred[tuple((i, j))]
                    value_str = " ${:.3f}".format(value)
                    transition_str += str(value_str)
                    value = times_pred_std[tuple((i, j))]
                    value_str = " \pm {:.3f}$".format(value)
                    value_str = "{\color{" + "green!50!black}{" + value_str[:-1] + "}}$"
                    value_str = "\\bm{" + value_str[:-1] + "}$"
                    transition_str += str(value_str)
                    transition_str += " &"

                    value = times_pred_error[tuple((i, j))]
                    errors_relative_pred.append(value)
                    value_str = " ${:2.0f}$".format(value * 100.0)
                    transition_str += str(value_str)
                    # transition_str += " &"



                    # value = times_pred_train[tuple((i, j))]
                    # value_str = " ${:.3f}".format(value)
                    # transition_str += str(value_str)
                    # value = times_pred_train_std[tuple((i, j))]
                    # value_str = " \pm {:.3f}$".format(value)
                    # transition_str += str(value_str)
                    # transition_str += " &"

                    # value = times_pred_train_error[tuple((i, j))]
                    # errors_relative_pred_train.append(value)
                    # value_str = " ${:2.0f}$".format(value * 100.0)
                    # transition_str += str(value_str)
                    # transition_str += " &"

                    # value = time_train_val[tuple((i, j))]
                    # value_str = " ${:.3f}".format(value)
                    # transition_str += str(value_str)
                    # value = times_train_val_std[tuple((i, j))]
                    # value_str = " \pm {:.3f}$".format(value)
                    # transition_str += str(value_str)
                    # transition_str += " &"

                    # value = times_train_val_error[tuple((i, j))]
                    # errors_relative_train_val.append(value)
                    # value_str = " ${:2.0f}$".format(value * 100.0)
                    # transition_str += str(value_str)
                    # # transition_str += " &"


                    transition_str += "  " + "\\" + "\\"
                    # print(transition_str)
                    lines.append(transition_str)
    error_relative_smalldt = np.mean(np.array(errors_relative_smalldt))
    error_relative_pred = np.mean(np.array(errors_relative_pred))
    # error_relative_pred_train = np.mean(np.array(errors_relative_pred_train))
    error_relative_train = np.mean(np.array(errors_relative_train))
    # error_relative_train_val = np.mean(np.array(errors_relative_train_val))

    value_str = "${:.2f} \\%$".format(error_relative_smalldt * 100.0)
    line = "\\hline \\hline \n"
    line += "\\multicolumn{3}{r|}{Average Relative Error} "
    line += "& " + value_str
    line += "& \n"

    value_str = "${:.2f} \\%$".format(error_relative_train * 100.0)
    line += "& " + value_str
    line += "& \n"

    value_str = "${:.2f} \\%$".format(error_relative_pred * 100.0)
    value_str = "$\\bm{{\color{" + "green!50!black}{" + value_str[1:-1] + "}}}$"
    line += "& " + value_str
    # line += "& \n"

    # value_str = "${:.2f} \\%$".format(error_relative_pred_train * 100.0)
    # line += "& " + value_str
    # line += "& \n"

    # value_str = "${:.2f} \\%$".format(error_relative_train_val * 100.0)
    # line += "& " + value_str
    # # line += "& \n"

    line += "\n"
    line += "\\" + "\\" + "\\hline"

    # print(line)
    lines.append(line)

    write_mode = "w"
    with io.open(file_path, write_mode) as f:
        for line in lines:
            f.write(line)
            f.write("\n")
        f.write("\n")
    return 0


def getTransitionTimesWithUncertainty(trajectories, n_splits, is_abd, dt, dt_msm, dt_save):
    assert np.shape(trajectories)[0] % n_splits == 0

    n_traj = int(np.shape(trajectories)[0] / n_splits)

    idx = np.random.permutation(np.arange(len(trajectories)))
    trajectories = trajectories[idx]
    
    for nn in range(n_splits):
        # print(nn*n_traj)
        # print((nn+1)*n_traj)
        trajectories_ = trajectories[nn*n_traj:(nn+1)*n_traj]
        cluster_transition_times_, _ = estimateRamaClusterMeanPassageTimesAlanine(trajectories_, dt, is_abd=is_abd, dt_msm=dt_msm, dt_save=dt_save, radius=20, print_=False)
        if nn==0:
            cluster_transition_times_all = cluster_transition_times_
        elif nn==1:
            for key in cluster_transition_times_all:
                cluster_transition_times_all[key] = list([cluster_transition_times_all[key]])
                cluster_transition_times_all[key].append(cluster_transition_times_[key])
        else:
            for key in cluster_transition_times_all:
                cluster_transition_times_all[key].append(cluster_transition_times_[key])

    cluster_transition_times_std = {}
    cluster_transition_times_mean = {}
    for key in cluster_transition_times_all:
        cluster_transition_times_mean[key] = np.mean(np.array(cluster_transition_times_all[key]))
        cluster_transition_times_std[key] = np.std(np.array(cluster_transition_times_all[key])) / np.sqrt(n_splits)
    return cluster_transition_times_mean, cluster_transition_times_std


def addResultsSystemAlanine(model, results, statistics, testing_mode):
    # assert("rama_density_target" not in results)
    # assert("rama_density_predicted" not in results)
    # assert("rama_bin_centers" not in results)
    # assert("rama_l1_hist_error" not in results)
    if statistics is not None:
        results.update({
            "rama_density_target":
            statistics["rama_density_target"],
            "rama_density_predicted":
            statistics["rama_density_predicted"],
            "rama_bin_centers":
            statistics["rama_bin_centers"],
            "rama_l1_hist_error":
            statistics["rama_l1_hist_error"],
        })
    if "rama_l1_hist_error" not in results["fields_2_save_2_logfile"]:
        results["fields_2_save_2_logfile"].append("rama_l1_hist_error")

    if "autoencoder" in testing_mode:
        targets_all = results["input_sequence_all"]
        predictions_all = results["input_decoded_all"]
    else:
        targets_all = results["targets_all"]
        predictions_all = results["predictions_all"]

    dt = results["dt"]
    is_abd = True
    dt_msm = 10 * 1e-12
    dt_save = 1e-9

    # targets_all = targets_all[:5]
    # predictions_all = predictions_all[:5]

    print("# Estimating Mean Passage Times (MPT) between Ramachadran clusters of TARGET trajectories.")
    cluster_transition_times_target, clustered_trajectories_target     = estimateRamaClusterMeanPassageTimesAlanine(targets_all, dt, is_abd=is_abd, dt_msm=dt_msm, dt_save=dt_save, radius=20)
    print("# Estimating Mean Passage Times (MPT) between Ramachadran clusters of PREDICTED trajectories.")
    cluster_transition_times_pred, clustered_trajectories_pred         = estimateRamaClusterMeanPassageTimesAlanine(predictions_all, dt, is_abd=is_abd, dt_msm=dt_msm, dt_save=dt_save, radius=20)

    # Computing the errors on the mean times
    cluster_transitions_considered_in_the_error = getRamaClustersOverWhichToComputeMFPTErrorKeys()
    cluster_transition_times_errors, transition_times_error_mean = utils.computeErrorOnTimes(cluster_transition_times_target, cluster_transition_times_pred, cluster_transitions_considered_in_the_error)

    results["transition_times_error_mean"]         = transition_times_error_mean

    results["cluster_transition_times_errors"]     = cluster_transition_times_errors
    results["cluster_transition_times_target"]     = cluster_transition_times_target
    results["cluster_transition_times_pred"]     = cluster_transition_times_pred

    results["clustered_trajectories_target"]     = clustered_trajectories_target
    results["clustered_trajectories_pred"]         = clustered_trajectories_pred

    if "transition_times_error_mean" not in results["fields_2_save_2_logfile"]: results["fields_2_save_2_logfile"].append("transition_times_error_mean")

    # Estimate free energy projection on the latent space
    covariance_factor_scale = 50.0
    gridpoints = 100
    results = utils.calculateFreeEnergyProjection(results,
                                                  covariance_factor_scale,
                                                  gridpoints)
    latent_range_state_percent = 0.01
    results = utils.caclulateFreeEnergyProjectionLatentClusters(
        results, latent_range_state_percent)

    # print(
    #     "# Estimating Mean Passage Times (MPT) between Latent space clusters.")

    # latent_cluster_mean_passage_times, latent_clustered_trajectories = utils.estimateLatentClusterMFPT(
    #     results, dt, dt_msm=dt_msm, dt_save=dt_save)
    # results[
    #     "latent_cluster_mean_passage_times"] = latent_cluster_mean_passage_times


    # # Calculating the free-energy projection with uncertainty
    # covariance_factor_scale = 50.0
    # gridpoints = 100
    # results = utils.calculateFreeEnergyProjectionWithUncertainty(results,
    #                                               covariance_factor_scale,
    #                                               gridpoints)

    # print(ark)
    # latent_range_state_percent = 0.01
    # results = utils.caclulateFreeEnergyProjectionLatentClusters(
    #     results, latent_range_state_percent)

    # print(
    #     "# Estimating Mean Passage Times (MPT) between Latent space clusters.")

    # latent_cluster_mean_passage_times, latent_clustered_trajectories = utils.estimateLatentClusterMFPT(
    #     results, dt, dt_msm=dt_msm, dt_save=dt_save)
    # results[
    #     "latent_cluster_mean_passage_times"] = latent_cluster_mean_passage_times
        
    return results


def addResultsSystemAlanineMSMAnalysis(model, results, statistics,
                                       testing_mode):

    if "autoencoder" in testing_mode:
        targets_all = results["input_sequence_all"]
        predictions_all = results["input_decoded_all"]
    else:
        targets_all = results["targets_all"]
        predictions_all = results["predictions_all"]

    dt = results["dt"]
    is_abd = True
    # dt of the MSM is 10ps
    dt_msm = 10 * 1e-12
    dt_save = 1e-9

    clustered_trajectories_target = results["clustered_trajectories_target"]

    print(
        "# Estimating Mean Passage Times (MPT) of MSMs fitted in MD data with small time-lags."
    )
    cluster_transition_times_target, _ = estimateRamaClusterMeanPassageTimesAlanine(
        targets_all,
        dt,
        is_abd=is_abd,
        radius=20,
        dt_msm=dt_msm,
        dt_save=dt_save,
        clustered_trajectories=clustered_trajectories_target)

    # dt of the MSM is 1 ps (testing the Markovianity)
    dt_msm_small = 1 * 1e-12
    cluster_transition_times_pred, _ = estimateRamaClusterMeanPassageTimesAlanine(
        targets_all,
        dt,
        is_abd=is_abd,
        radius=20,
        dt_msm=dt_msm_small,
        dt_save=dt_save,
        clustered_trajectories=clustered_trajectories_target)

    # Computing the errors on the mean times
    cluster_transitions_considered_in_the_error = getRamaClustersOverWhichToComputeMFPTErrorKeys(
    )
    cluster_transition_times_errors = {}
    for key in cluster_transition_times_target:
        if key in cluster_transitions_considered_in_the_error:
            # print("Transition {:} considered in the error.".format(key))
            mean_time_target = cluster_transition_times_target[key]
            mean_time_pred = cluster_transition_times_pred[key]

            if mean_time_target > 0.0:
                error = np.abs(mean_time_target - mean_time_pred)
                error = error / mean_time_target
            else:
                error = 0.0
            cluster_transition_times_errors[key] = error

    temp = []
    for key in cluster_transition_times_errors:
        temp.append(cluster_transition_times_errors[key])
    temp = np.array(temp)
    transition_times_error_mean = np.mean(temp)

    results[
        "transition_times_error_mean_msm_Dt_small"] = transition_times_error_mean
    results[
        "cluster_transition_times_errors_msm_Dt_small"] = cluster_transition_times_errors
    results[
        "cluster_transition_times_target_msm_Dt_small"] = cluster_transition_times_target
    results[
        "cluster_transition_times_pred_msm_Dt_small"] = cluster_transition_times_pred
    results["cluster_transition_times_msm_Dt_small"] = dt_msm_small
    return results


def transformToBAD(traj, save_path):
    raise ValueError("Not implemented.")

import numpy as np
import math, itertools
from math import cos, sin, sqrt, acos, atan2, fabs, pi
from sys import argv


def find_BA(dd1, dd2, dd3, dd4, angles, bonds):
    angleID = -1
    for aa in range(len(angles)):
        if (dd2 == angles[aa][0] and dd3 == angles[aa][1]
                and dd4 == angles[aa][2]):
            angleID = aa
            break
    if (angleID == -1):
        print("angle not found", dd2, dd3, dd4)
        exit()
    #find bond
    bondID = -1
    for bb in range(len(bonds)):
        if (dd3 == bonds[bb][0] and dd4 == bonds[bb][1]):
            bondID = bb
            break
    if (bondID == -1):
        print("bond not found")
        exit()
    return bondID, angleID


########################################################
def place_atom(atom_a, atom_b, atom_c, angle, torsion, bond):

    #print atom_a, atom_b, atom_c, angle, torsion, bond
    R = bond
    ab = np.subtract(atom_b, atom_a)
    bc = np.subtract(atom_c, atom_b)
    bcn = bc / np.linalg.norm(bc)

    case = 1
    okinsert = False
    while (okinsert == False):
        #case 1
        if (case == 1):
            d = np.array([
                -R * cos(angle), R * cos(torsion) * sin(angle),
                R * sin(torsion) * sin(angle)
            ])
            n = np.cross(bcn, ab)
            n = n / np.linalg.norm(n)
            nbc = np.cross(bcn, n)
        #case 2
        elif (case == 2):
            d = np.array([
                -R * cos(angle), R * cos(torsion) * sin(angle),
                R * sin(torsion) * sin(angle)
            ])
            n = np.cross(ab, bcn)
            n = n / np.linalg.norm(n)
            nbc = np.cross(bcn, n)
        #case 3
        elif (case == 3):
            d = np.array([
                -R * cos(angle), R * cos(torsion) * sin(angle),
                -R * sin(torsion) * sin(angle)
            ])
            n = np.cross(ab, bcn)
            n = n / np.linalg.norm(n)
            nbc = np.cross(bcn, n)
        #case 4
        elif (case == 4):
            d = np.array([
                -R * cos(angle), R * cos(torsion) * sin(angle),
                R * sin(torsion) * sin(angle)
            ])
            n = np.cross(ab, bcn)
            n = n / np.linalg.norm(n)
            nbc = np.cross(n, bcn)

        m = np.array([[bcn[0], nbc[0], n[0]], [bcn[1], nbc[1], n[1]],
                      [bcn[2], nbc[2], n[2]]])
        d = m.dot(d)
        atom_d = d + atom_c

        #test dihedral
        r21 = np.subtract(atom_b, atom_a)
        r23 = np.subtract(atom_b, atom_c)
        r43 = np.subtract(atom_d, atom_c)
        n1 = np.cross(r21, r23)
        n2 = np.cross(r23, r43)
        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)
        r23 = r23 / np.linalg.norm(r23)
        m = np.cross(n1, r23)
        x = np.dot(n1, n2)
        y = np.dot(m, n2)
        phi = atan2(y, x)

        #test angle
        d12 = np.subtract(atom_b, atom_c)
        d32 = np.subtract(atom_d, atom_c)
        d12 = d12 / np.linalg.norm(d12)
        d32 = d32 / np.linalg.norm(d32)
        cos_theta = np.dot(d12, d32)
        m = np.linalg.norm(np.cross(d12, d32))
        theta = atan2(m, cos_theta)

        if (fabs(theta - angle) < 0.0001 and fabs(phi - torsion) < 0.0001):
            okinsert = True
        else:
            if (case < 4): case += 1
            else:
                print("no case found", theta, angle, phi, torsion, atom_d)
                break
    return atom_d


########################################################
def test_angle(atoms, angleID, angles):
    ii, jj, kk = angles[angleID]
    d12 = np.subtract(atoms[ii], atoms[jj])
    d32 = np.subtract(atoms[kk], atoms[jj])
    d12 = d12 / np.linalg.norm(d12)
    d32 = d32 / np.linalg.norm(d32)
    cos_theta = np.dot(d12, d32)
    m = np.linalg.norm(np.cross(d12, d32))
    theta = atan2(m, cos_theta)
    return theta


########################################################
def test_dihedral(atoms, dihedralID, dih):

    ii, jj, kk, ll = dih[dihedralID]
    r21 = np.subtract(atoms[jj], atoms[ii])
    r23 = np.subtract(atoms[jj], atoms[kk])
    r43 = np.subtract(atoms[ll], atoms[kk])

    n1 = np.cross(r21, r23)
    n2 = np.cross(r23, r43)

    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    r23 = r23 / np.linalg.norm(r23)

    m = np.cross(n1, r23)
    x = np.dot(n1, n2)
    y = np.dot(m, n2)

    phi = atan2(y, x)

    return phi


########################################################
def new_config(CVsB, CVsA, CVsD, dih, angles, bonds, Nfile):

    ang = CVsA[0]

    an = -1.0 * ang
    R1 = np.array([[cos(an), -sin(an), 0.0], [sin(an), cos(an), 0.0],
                   [0.0, 0.0, 1.0]])
    R2 = np.array([[1.0, 0.0, 0.0],
                   [0.0, cos(-math.pi / 4), -sin(-math.pi / 4)],
                   [0.0, sin(-math.pi / 4),
                    cos(-math.pi / 4)]])
    R3 = np.array([[cos(-math.pi / 4), 0.0,
                    sin(-math.pi / 4)], [0.0, 1.0, 0.0],
                   [-sin(-math.pi / 4), 0.0,
                    cos(-math.pi / 4)]])

    atoms = np.zeros((10, 3), float)

    ### first 3 atoms ###
    vec01 = [1.0 / sqrt(2), 1.0 / sqrt(2), 0.0]
    vec31 = np.dot(R1, vec01)
    vec01 = np.dot(R2, vec01)
    vec31 = np.dot(R2, vec31)
    vec01 = np.dot(R3, vec01)
    vec31 = np.dot(R3, vec31)

    atoms[9] = [CVsB[0] * vec01[0], CVsB[0] * vec01[1], CVsB[0] * vec01[2]]
    atoms[8] = [0.0, 0.0, 0.0]
    atoms[6] = [CVsB[1] * vec31[0], CVsB[1] * vec31[1], CVsB[1] * vec31[2]]

    #atoms[6] = [(16.318708143-10)/10.0, (17.587712674-10)/10.0,(15.683474482-10)/10.0]
    #atoms[8]  = [(16.612946587-10)/10.0,(18.823512485 -10)/10.0,(15.209664558-10)/10.0]
    #atoms[9] = [(17.934527001-10)/10.0, (19.294894241 -10)/10.0, (14.819090178-10)/10.0]

    ### iteratively all other atoms ###
    for dd in range(len(dih)):
        dd1, dd2, dd3, dd4 = dih[dd]
        bondID, angleID = find_BA(dd1, dd2, dd3, dd4, angles, bonds)
        coord = place_atom(atoms[dd1], atoms[dd2], atoms[dd3], CVsA[angleID],
                           CVsD[dd], CVsB[bondID])
        atoms[dd4] = coord

    testBAD = True
    if (testBAD):
        #bonds
        for mm in range(len(bonds)):
            ii, jj = bonds[mm]
            dist = pow(atoms[ii][0] - atoms[jj][0], 2) + pow(
                atoms[ii][1] - atoms[jj][1], 2) + pow(
                    atoms[ii][2] - atoms[jj][2], 2)
            if (fabs(sqrt(dist) - CVsB[mm]) > 0.0001):
                print("bond", bonds[mm], CVsB[mm], sqrt(dist), atoms[ii],
                      atoms[jj], "Reading snapshot ", Nfile)
        #angles
        for mm in range(len(angles)):
            acos_theta = test_angle(atoms, mm, angles)
            #print "angle",angles[mm],CVsA[mm]*180/pi,acos_theta*180/pi
            if (fabs(acos_theta - CVsA[mm]) > 0.0001):
                print("angle", angles[mm], CVsA[mm], acos_theta,
                      "Reading snapshot ", Nfile)
        #dihedrals
        for mm in range(len(dih)):
            acos_theta = test_dihedral(atoms, mm, dih)
            #print "dihedral",dih[mm],CVsD[mm]*180/pi,acos_theta*180/pi
            if (fabs(acos_theta - CVsD[mm]) > 0.0001):
                print("dihedral", dih[mm], CVsD[mm], acos_theta,
                      "Reading snapshot ", Nfile)

    return atoms


########################################################
def remove_com(conf, mass, masstotal):
    # calculate center of mass165
    comp = [0.0, 0.0, 0.0]
    for i in range(len(conf)):
        for dim in range(3):
            comp[dim] += mass[i] * conf[i][dim]
    for dim in range(3):
        comp[dim] /= masstotal

    # substract center of mass
    conf_com = np.zeros((len(conf), 3), float)
    for i in range(len(conf)):
        for dim in range(3):
            conf_com[i, dim] = conf[i][dim] - comp[dim]

    return conf_com


#######################################################################
def rotationmatrix(coordref, coord):
    assert (coordref.shape[1] == 3)
    assert (coordref.shape == coord.shape)
    correlation_matrix = np.dot(np.transpose(coordref), coord)
    vv, ss, ww = np.linalg.svd(correlation_matrix)
    is_reflection = (np.linalg.det(vv) * np.linalg.det(ww)) < 0.0
    #if is_reflection:
    #print "is_reflection"
    #vv[-1,:] = -vv[-1,:]
    #ss[-1] = -ss[-1]
    #vv[:, -1] = -vv[:, -1]
    rotation = np.dot(vv, ww)

    confnew = []
    for i in range(len(coord)):
        xx = rotation[0][0] * coord[i][0] + rotation[0][1] * coord[i][
            1] + rotation[0][2] * coord[i][2]
        yy = rotation[1][0] * coord[i][0] + rotation[1][1] * coord[i][
            1] + rotation[1][2] * coord[i][2]
        zz = rotation[2][0] * coord[i][0] + rotation[2][1] * coord[i][
            1] + rotation[2][2] * coord[i][2]
        confnew.append((xx, yy, zz))

    return confnew


def generateXYZfileFromABDnoRotTr(badfile, ref_conf, conffile, lammps_file):
    ###### topology ########################################
    atomNames = ["C", "C", "O", "N", "C", "C", "C", "O", "N", "C"]
    atomtypes = [1, 1, 2, 3, 1, 1, 1, 2, 3, 1]
    mass = [12.01, 12.01, 16.0, 14.01, 12.01, 12.01, 12.01, 16.0, 14.01, 12.01]
    masstotal = np.sum(mass)

    dih = ((9, 8, 6, 7), (9, 8, 6, 4), (8, 6, 4, 5), (8, 6, 4, 3),
           (6, 4, 3, 1), (4, 3, 1, 2), (4, 3, 1, 0)
           )  #,(7,6,4,5),(5,4,3,1),(7,6,4,3),(0,3,1,2),(4,8,6,7),(3,6,4,5)
    angles = ((9, 8, 6), (8, 6, 7), (8, 6, 4), (6, 4, 5), (6, 4, 3), (4, 3, 1),
              (3, 1, 2), (3, 1, 0))  #,(7,6,4),(5,4,3),(2,1,0)
    bonds = ((9, 8), (8, 6), (6, 7), (6, 4), (4, 5), (4, 3), (3, 1), (1, 2),
             (1, 0))
    ########################################################

    ########################################################
    #read reference file
    f = open(ref_conf)
    for line in f:
        temp = line.split()
        poss = [float(x) for x in temp]

    coordREF = np.zeros((int(len(poss) / 3), 3), float)
    for i in range(len(coordREF)):
        for dim in range(3):
            coordREF[i, dim] = poss[3 * i + dim]

    outfileC = open(conffile, "w")
    outfileC.close()
    ########################################################
    ##### read CVs #####
    f = open(badfile)
    line = ''
    Nfile = 1
    for line in f:
        if (Nfile % 100000 == 0): print("Reading line ... ", Nfile)

        L = [float(x) for x in line.split()]
        CVsB, CVsA, CVsD = [], [], []
        for l in range(len(L)):
            if l < 9: CVsB.append(L[l])
            elif l < 17: CVsA.append(L[l])
            else: CVsD.append(L[l])

        #print len(CVsB),len(CVsA),len(CVsD)
        conf = new_config(CVsB, CVsA, CVsD, dih, angles, bonds, Nfile)
        conf_com = remove_com(conf, mass, masstotal)
        confnew = rotationmatrix(coordREF, conf_com)
        #confnew = conf

        #write to file
        outfileC = open(conffile, "a")
        outfileC.write("%d\n" % (len(confnew)))
        outfileC.write(
            'Lattice="20.0 0.0 0.0 0.0 20.0 0.0 0.0 0.0 20.0" Properties=species:S:1:pos:R:3 \n'
        )
        for i in range(len(confnew)):
            #print(atomNames[i],confnew[i][0]*10+10.0,confnew[i][1]*10+10.0,confnew[i][2]*10+10.0)
            outfileC.write(
                "%s%19.9f%17.9f%17.9f\n" %
                (atomNames[i], confnew[i][0] * 10 + 10.0,
                 confnew[i][1] * 10 + 10.0, confnew[i][2] * 10 + 10.0))
        outfileC.close()

        Nfile += 1
    f.close()

    ########################################################
    #write data file for Lammps so bonds are correct in Ovito
    # "dataLammps.txt"
    pdb = open(lammps_file, 'w')

    pdb.write("LAMMPS 'data.' description \n")
    pdb.write("\n")
    pdb.write("      %d atoms\n" % (len(atomNames)))
    pdb.write("      %d bonds\n" % (len(bonds)))
    pdb.write("\n")
    pdb.write("       3 atom types\n")
    pdb.write("       1 bond types\n")
    pdb.write("\n")
    pdb.write("    0.0 %1.2f      xlo xhi\n" % (20.0))
    pdb.write("    0.0 %1.2f      ylo yhi\n" % (20.0))
    pdb.write("    0.0 %1.2f      zlo zhi\n" % (20.0))
    pdb.write("\n\n")
    pdb.write("Atoms\n")
    pdb.write("\n")

    for i in range(len(coordREF)):
        pdb.write("     %d   %d  %d  %1.4f    %1.4f    %1.4f\n" %
                  (i + 1, 1, atomtypes[i], coordREF[i][0] * 10 + 10.0,
                   coordREF[i][1] * 10 + 10.0, coordREF[i][2] * 10 + 10.0))

    pdb.write("\n")
    pdb.write("Bonds\n")
    pdb.write("\n")
    for n in range(len(bonds)):
        pdb.write("     %d   1     %d     %d\n" %
                  (n + 1, bonds[n][0] + 1, bonds[n][1] + 1))
    pdb.close()

    return 0
