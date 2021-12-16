#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import numpy as np
import os
from ... import Utils as utils

import numpy as np
from matplotlib import pyplot as plt
from math import pi, cos, sin


def getClusterInfo():
    center = [-0.57, 1.45]
    axes = [0.15, 0.3]
    rot_angle = -pi / 4

    cluster_1 = [center, axes, rot_angle]

    center = [0.45, 0.05]
    axes = [0.35, 0.15]
    rot_angle = 0.0
    # -5*pi/6

    cluster_2 = [center, axes, rot_angle]

    return cluster_1, cluster_2


# x = u + a cos(t) ; y = v + b sin(t)
def plotEllipse(ax, center, axes, rot_angle, color="k", linewidth=2):
    u = center[0]  # x-position of the center
    v = center[1]  # y-position of the center
    a = axes[0]  # radius on the x-axis
    b = axes[1]  # radius on the y-axis
    t = np.linspace(0, 2 * pi, 100)
    Ell = np.array([a * np.cos(t), b * np.sin(t)])
    #u,v removed to keep the same center location
    R_rot = np.array([[cos(rot_angle), -sin(rot_angle)],
                      [sin(rot_angle), cos(rot_angle)]])
    #2-D rotation matrix
    Ell_rot = np.zeros((2, Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:, i] = np.dot(R_rot, Ell[:, i])
    # ax.plot( u+Ell[0,:] , v+Ell[1,:] )     #initial ellipse
    ax.plot(u + Ell_rot[0, :], v + Ell_rot[1, :], color,
            linewidth=linewidth)  #rotated ellipse
    return ax


def plotClusters(ax,
                 plot1=True,
                 plot2=True,
                 color1="tab:green",
                 color2="tab:blue",
                 linewidth=2):
    cluster_1, cluster_2 = getClusterInfo()
    center, axes, rot_angle = cluster_1

    if plot1:
        ax = plotEllipse(ax,
                         center,
                         axes,
                         rot_angle,
                         color=color1,
                         linewidth=linewidth)

    center, axes, rot_angle = cluster_2

    if plot2:
        ax = plotEllipse(ax,
                         center,
                         axes,
                         rot_angle,
                         color=color2,
                         linewidth=linewidth)
    return ax


def isInEllipse(pos, center, axes, rot_angle):
    x = pos[0] - center[0]
    y = pos[1] - center[1]
    pos_trans = np.array([x, y])
    R_rot = np.array([[cos(rot_angle), sin(rot_angle)],
                      [-sin(rot_angle), cos(rot_angle)]])
    # print(np.shape(R_rot))
    # print(np.shape(pos))
    pos_trans = np.dot(R_rot, pos_trans)
    # print(x_trans)
    a = axes[0]  # radius on the x-axis
    b = axes[1]  # radius on the y-axis
    x = pos_trans[0]
    y = pos_trans[1]
    temp = x * x / (a * a) + y * y / (b * b)
    # print(temp)
    if temp <= 1.0:
        return True
    else:
        return False


def getClusterBMP(x):
    cluster_1, cluster_2 = getClusterInfo()
    center, axes, rot_angle = cluster_1
    if isInEllipse(x, center, axes, rot_angle): return 1
    center, axes, rot_angle = cluster_2
    if isInEllipse(x, center, axes, rot_angle): return 0
    return None


def clusterTrajectoryBMP(trajectory):
    traj_clustered = np.array([getClusterBMP(x) for x in trajectory])
    return traj_clustered


def reentersCluster(current_cluster, traj_clustered, t, patience):
    next_clusters = traj_clustered[t + 1:t + 1 + patience]
    if np.any(next_clusters == current_cluster):
        return True
    else:
        return False


def estimateClusterTransitionTimesBMP(trajectories, n_clusters, dt):
    MIN_STAY_TIME = 5.0
    patience = int(MIN_STAY_TIME / dt)
    print("patience = {:}".format(patience))

    cluster_transition_times = {}
    for i in range(n_clusters):
        for j in range(n_clusters):
            # Transition time from cluster i to cluster j
            if i != j: cluster_transition_times.update({tuple((i, j)): []})

    # Estimating the transition times
    for traj in trajectories:
        traj_clustered = clusterTrajectoryBMP(traj)

        T = len(traj_clustered)
        time_in_cluster = 1
        is_first_transition = True
        entered_cluster = False

        for t in range(T - 1):

            cluster_t = getClusterBMP(traj[t])

            if (cluster_t is not None) and (entered_cluster == False):
                # Entering first time on a meta stable state cluster
                entered_cluster = True
                current_cluster = cluster_t

            if entered_cluster:
                # Once entered on a cluster, get the cluster at the next time step
                cluster_t_plus_one = getClusterBMP(traj[t + 1])

                if cluster_t_plus_one is None:
                    # If the cluster at the next time step is None, we consider that we remain on the previous cluster
                    next_cluster = current_cluster
                else:
                    next_cluster = cluster_t_plus_one

                if (next_cluster != current_cluster and not reentersCluster(
                        current_cluster, traj_clustered, t, patience)
                    ) and not is_first_transition:
                    # Changed cluster
                    # Entered a different cluster,
                    # it is not the first transition,
                    # and is not reendering the previous cluster in the next patience timesteps
                    cluster_transition_times[tuple(
                        [current_cluster,
                         next_cluster])].append(time_in_cluster)
                    time_in_cluster = 1

                elif (next_cluster != current_cluster and reentersCluster(
                        current_cluster, traj_clustered, t,
                        patience)) and not is_first_transition:
                    # Changed cluster,
                    # but is reendering the previous cluster in the next patience timesteps
                    # so, we consider that we stay in the previous cluster, and change the next clusters accordingly
                    traj_clustered[t + 1:t + 1 + patience] = current_cluster
                    time_in_cluster += 1

                elif (current_cluster != next_cluster) and is_first_transition:
                    is_first_transition = False

                else:
                    time_in_cluster += 1

                current_cluster = next_cluster

    # Multiplying by dt
    for key in cluster_transition_times:
        times_ = cluster_transition_times[key]
        times_ = [float(time) * dt for time in times_]
        cluster_transition_times[key] = times_

    # print(cluster_transition_times)
    return cluster_transition_times


def computeMean(array):
    if len(array) > 0:
        return np.max(array)
    else:
        return 0


def addResultsSystemBMP(model, results, statistics, testing_mode):
    print("# addResultsSystemBMP() #")
    if "autoencoder" in testing_mode:
        targets_all = results["input_sequence_all"]
        predictions_all = results["input_decoded_all"]
    else:
        targets_all = results["targets_all"]
        predictions_all = results["predictions_all"]
    dt = results["dt"]
    n_clusters = 2

    cluster_transition_times_target = estimateClusterTransitionTimesBMP(
        targets_all, n_clusters, dt)
    cluster_transition_times_pred = estimateClusterTransitionTimesBMP(
        predictions_all, n_clusters, dt)
    # Computing the errors on the mean times

    transition_times_errors = []
    for key in cluster_transition_times_target:
        times_targ = cluster_transition_times_target[key]
        times_pred = cluster_transition_times_pred[key]
        mean_time_target = computeMean(times_targ)
        mean_time_pred = computeMean(times_pred)
        error = np.abs(mean_time_target - mean_time_pred)
        error = error / mean_time_target
        transition_times_errors.append(error)
    transition_times_errors = np.array(transition_times_errors)
    transition_times_error_mean = np.mean(transition_times_errors)

    results["transition_times_error_mean"] = transition_times_error_mean
    results["transition_times_errors"] = transition_times_errors

    results[
        "cluster_transition_times_target"] = cluster_transition_times_target
    results["cluster_transition_times_pred"] = cluster_transition_times_pred

    results["fields_2_save_2_logfile"].append("transition_times_error_mean")

    # Estimate free energy projection on the latent space
    # covariance_factor_scale = 50.0
    covariance_factor_scale = 20.0
    # gridpoints                 = 200
    gridpoints = 100
    results = utils.calculateFreeEnergyProjection(results,
                                                  covariance_factor_scale,
                                                  gridpoints)

    # latent_range_state_percent = [0.01, 0.01] # first is the right one, second is the left one

    latent_range_state_percent = [0.01, 0.02]
    results = utils.caclulateFreeEnergyProjectionLatentClusters(
        results, latent_range_state_percent)

    print(
        "# Estimating Mean Passage Times (MPT) between Latent space clusters.")
    dt_msm = 10
    dt_save = 1
    latent_cluster_mean_passage_times, latent_clustered_trajectories = utils.estimateLatentClusterMFPT(
        results, dt, dt_msm=dt_msm, dt_save=dt_save)
    results[
        "latent_cluster_mean_passage_times"] = latent_cluster_mean_passage_times

    dt_msm_small = 0.5
    latent_cluster_mean_passage_times, latent_clustered_trajectories = utils.estimateLatentClusterMFPT(
        results, dt, dt_msm=dt_msm_small, dt_save=dt_save)
    results[
        "latent_cluster_mean_passage_times_dt_msm_small"] = latent_cluster_mean_passage_times
    results["dt_msm_small"] = dt_msm_small
    return results
