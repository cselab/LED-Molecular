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
from . import utils_data
from . import utils_statistics


def computeErrorOnTimes(cluster_transition_times_target,
                        cluster_transition_times_pred,
                        cluster_transitions_considered_in_the_error=None):
    if cluster_transitions_considered_in_the_error is None:
        cluster_transitions_considered_in_the_error = cluster_transition_times_target.keys(
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
    return cluster_transition_times_errors, transition_times_error_mean


def calculateLatentFreeEnergy(latent_states_flatten, covariance_factor,
                              latent_state_free_energy_grid):
    print("# calculateLatentFreeEnergy() #")
    latent_state_density = calculateGaussianKernelDensityEstimate(
        latent_states_flatten, covariance_factor,
        latent_state_free_energy_grid)
    latent_state_freenergy = -np.log(latent_state_density)
    latent_state_freenergy = np.array(latent_state_freenergy).T
    return latent_state_freenergy


def calculateGaussianKernelDensityEstimate(data, covariance_factor, grid):
    from scipy.stats import gaussian_kde
    try:
        kde_estimator = gaussian_kde(data)
        kde_estimator.covariance_factor = lambda: covariance_factor
        kde_estimator._compute_covariance()
        ret_ = kde_estimator(grid)
    except Exception as inst:
        print("Exception: {:}".format(inst))
        print(
            "# gaussian_kde() failed. setting the latent state freenergy to zero."
        )
        ret_ = np.zeros(np.shape(grid))
    return ret_


def clusterTrajectoryLatent(traj_latent, latent_state_regions):
    traj_clustered = []
    for t in range(np.shape(traj_latent)[0]):
        latent_state = traj_latent[t]
        cluster = getLatentCluster(latent_state, latent_state_regions)
        traj_clustered.append(cluster)
    traj_clustered = np.array(traj_clustered)
    return traj_clustered


def getLatentCluster(latent_state, latent_state_regions):
    cluster = None
    num_clusters = len(latent_state_regions)
    for cluster_id in range(num_clusters):
        latent_state_region = latent_state_regions[cluster_id]
        if latent_state >= latent_state_region[
                0] and latent_state <= latent_state_region[1]:
            cluster = cluster_id
    return cluster


def printTransitionTimes(cluster_labels, transition_times, nstates):
    # print("# printTransitionTimes()")
    order = np.arange(nstates)
    print("-" * 100)
    line_f = " " * 10 + "{:8s}, " * nstates
    line_f = line_f.format(
        *[cluster_labels[order[ii]] for ii in range(nstates)])
    print(line_f)
    line_title_base = "{:8s}, "
    line_inhalt_base = ""
    for ii in range(nstates):
        i = order[ii]
        jlist = range(nstates)
        jlist = [order[jj] for jj in jlist]
        line_inhalt = line_inhalt_base
        for j in jlist:
            if i == j:
                mtt = 0.0
            else:
                mtt = transition_times[tuple((i, j))]
            line_inhalt += "{:0>8f}, ".format(mtt)
        line_title = line_title_base.format(cluster_labels[i])
        line_f = line_title + line_inhalt
        print(line_f)
    return 0


def caclulateFreeEnergyProjectionLatentClusters(results,
                                                latent_range_state_percent=0.01
                                                ):

    print("# caclulateFreeEnergyProjectionLatentClusters() #")

    # latent_state_density             = results["latent_state_density"]
    latent_states_all = results["latent_states_all"]

    if np.shape(latent_states_all)[2] == 1:

        latent_state_free_energy_grid = results[
            "latent_state_free_energy_grid"]
        latent_states_flatten_range = results["latent_states_flatten_range"]
        latent_states_flatten = results["latent_states_all"]
        freenergy = results["latent_state_freenergy"]

        latent_states_flatten = np.array(latent_states_flatten).flatten()

        indexes = np.where(np.r_[True, freenergy[1:] < freenergy[:-1]]
                           & np.r_[freenergy[:-1] < freenergy[1:], True])[0]
        print(indexes)

        free_energy_minima = freenergy[indexes]

        num_states = len(indexes)
        print("Number of local minima in the latent space found {:}.".format(
            num_states))

        for state in range(num_states - 1):
            indx = indexes[state]
            latent_region_size = latent_state_free_energy_grid[
                indx + 1] - latent_state_free_energy_grid[indx]
            print(latent_region_size)

        # latent_range_state = 0.05 * latent_states_flatten_range
        # latent_range_state = 0.02 * latent_states_flatten_range
        # latent_range_state = 0.01 * latent_states_flatten_range

        if isinstance(latent_range_state_percent, list):
            if not len(latent_range_state_percent) == num_states:
                print(
                    "# Provided latent state region sizes to cluster the latent state based on the minima are not equal with the number of minima. Using the default values."
                )
                latent_range_state_percent = 0.01
                latent_range_state_percent = [latent_range_state_percent
                                              ] * num_states
            print("# Provided latent state region sizes {:}".format(
                latent_range_state_percent))
        else:
            latent_range_state_percent = [latent_range_state_percent
                                          ] * num_states

        latent_state_regions = []
        for state in range(num_states):
            latent_range_state_percent_state = latent_range_state_percent[
                state]
            latent_range_state = latent_range_state_percent_state * latent_states_flatten_range
            indx = indexes[state]
            latent_region_of_state_min = latent_state_free_energy_grid[
                indx] - latent_range_state
            latent_region_of_state_max = latent_state_free_energy_grid[
                indx] + latent_range_state
            latent_state_region = [
                latent_region_of_state_min, latent_region_of_state_max
            ]
            latent_state_regions.append(latent_state_region)
            print("Latent state {:}".format(state))
            print("Latent state region {:}".format(latent_state_region))

        cluster_labels = np.arange(num_states)
        cluster_labels = [
            "$S^{z}_" + "{:}".format(x) + "$" for x in cluster_labels
        ]

        results["free_energy_latent_cluster_labels"] = cluster_labels
        results["free_energy_latent_state_regions"] = latent_state_regions
        results["latent_range_state_percent"] = latent_range_state_percent
        results["free_energy_minima"] = free_energy_minima

    return results


def estimateLatentClusterMFPT(results, dt, dt_msm=10 * 1e-12, dt_save=1e-9):
    ###################################################################
    ####### Estimating the mean_passage times based on latent clusters
    ###################################################################
    print("# estimateLatentClusterMFPT() #")

    latent_states_all = results["latent_states_all"]
    cluster_labels = results["free_energy_latent_cluster_labels"]
    latent_state_regions = results["free_energy_latent_state_regions"]

    trajectories = latent_states_all

    # Estimating the mean_passage times
    clustered_trajectories = []
    for traj_num in range(len(trajectories)):
        traj_latent = trajectories[traj_num]
        print("traj_latent = {:}/{:}".format(traj_num, len(trajectories)))

        traj_clustered = clusterTrajectoryLatent(traj_latent,
                                                 latent_state_regions)
        clustered_trajectories.append(traj_clustered)
    clustered_trajectories = np.array(clustered_trajectories)

    import time
    # TODO: pyemma correctly not compiling
    # import pyemma
    time_start = time.time()

    n_clusters = len(cluster_labels)

    print("n_clusters = {:}".format(n_clusters))
    cluster_mean_passage_times = {}
    for i in range(n_clusters):
        for j in range(n_clusters):
            # mean_passage time from cluster i to cluster j
            # if i != j: cluster_mean_passage_times.update({tuple((i,j)):[]})
            if i != j: cluster_mean_passage_times.update({tuple((i, j)): 0.0})

    clustered_trajectories = np.array(clustered_trajectories)
    # Adding an auxiliary cluster (equal to None)
    clustered_trajectories[clustered_trajectories == None] = int(n_clusters)
    # print(np.shape(clustered_trajectories))
    # print(ark)

    # subsample = 10
    subsample = 1
    # dt_traj = subsample * 0.1e-12
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

        msm_active_set = msm.active_set

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
    printTransitionTimes(cluster_labels, cluster_mean_passage_times,
                         n_clusters)

    time_end = time.time()
    utils_data.printTime(time_end - time_start)

    clustered_trajectories = np.array(clustered_trajectories)
    # print(cluster_mean_passage_times)
    return cluster_mean_passage_times, clustered_trajectories

def calculateLatentFreeEnergyWithUncertainty(latent_states_all, n_splits, covariance_factor, latent_state_free_energy_grid):
    n_traj = np.shape(latent_states_all)[0]
    assert n_traj % n_splits == 0, "n_traj = {:} % n_splits = {:}".format(n_traj, n_splits)
    n_traj_bootstrap = int(n_traj/n_splits)
    latent_state_freenergy_bootstrap = []

    idx = np.random.permutation(np.arange(len(latent_states_all)))
    latent_states_all = latent_states_all[idx]
    
    for nn in range(n_splits):
        print("Bootstrap {:}/{:}".format(nn, n_splits))
        data = latent_states_all[nn*n_traj_bootstrap:(nn+1)*n_traj_bootstrap]
        latent_states_flatten = np.array(data).flatten()
        latent_state_freenergy = calculateLatentFreeEnergy(
            latent_states_flatten, covariance_factor,
            latent_state_free_energy_grid)
        latent_state_freenergy_bootstrap.append(latent_state_freenergy)
    return latent_state_freenergy_bootstrap

def calculateFreeEnergyProjectionWithUncertainty(results,
                                  covariance_factor_scale=50.0,
                                  gridpoints=100):

    print("# calculateFreeEnergyProjectionWithUncertainty() #")

    latent_states_all = results["latent_states_all"]

    if np.shape(latent_states_all)[2] == 1:

        latent_states_flatten = np.array(latent_states_all).flatten()

        latent_states_flatten_range = latent_states_flatten.max(
        ) - latent_states_flatten.min()

        # Free energy profiles projected into the (slowest) latent space coordinate F(ψ1)=−kBT ln(p(ψ1))
        # Plotting F/kbT =  - ln(p(ψ1))
        covariance_factor = latent_states_flatten_range / covariance_factor_scale
        margin = 0.2
        # covariance_factor = .25
        grid_min = latent_states_flatten.min(
        ) - margin * latent_states_flatten_range
        grid_max = latent_states_flatten.max(
        ) + margin * latent_states_flatten_range
        latent_state_free_energy_grid = np.linspace(grid_min, grid_max,
                                                    gridpoints)

        print("np.max(latent_states_flatten) = {:}".format(
            np.max(latent_states_flatten)))
        print("np.min(latent_states_flatten) = {:}".format(
            np.min(latent_states_flatten)))

        print("latent_state_free_energy_grid")
        print(latent_state_free_energy_grid)

        # Bootstrapping
        n_splits = 16
        n_traj_bootstrap = int(np.shape(latent_states_all)[0] / n_splits)
        print(n_traj_bootstrap)

        latent_state_freenergy_bootstrap = []


        idx = np.random.permutation(np.arange(len(latent_states_all)))
        latent_states_all = latent_states_all[idx]
    
        for nn in range(n_splits):
            data = latent_states_all[nn*n_traj_bootstrap:(nn+1)*n_traj_bootstrap]
            latent_states_flatten = np.array(data).flatten()

            latent_state_freenergy = calculateLatentFreeEnergy(
                latent_states_flatten, covariance_factor,
                latent_state_free_energy_grid)
            latent_state_freenergy_bootstrap.append(latent_state_freenergy)

        results["latent_state_freenergy_bootstrap"] = latent_state_freenergy_bootstrap
        results[
            "latent_state_free_energy_grid"] = latent_state_free_energy_grid
        results["latent_states_flatten_range"] = latent_states_flatten_range
        results["covariance_factor"] = covariance_factor


        latent_states_all_train = latent_states_all[:32]
        # Bootstrapping
        n_splits = 16
        n_traj_bootstrap = int(np.shape(latent_states_all_train)[0] / n_splits)
        print(n_traj_bootstrap)

        latent_state_freenergy_bootstrap_train = []

        idx = np.random.permutation(np.arange(len(latent_states_all_train)))
        latent_states_all_train = latent_states_all_train[idx]
        
        for nn in range(n_splits):
            data = latent_states_all_train[nn*n_traj_bootstrap:(nn+1)*n_traj_bootstrap]
            latent_states_flatten = np.array(data).flatten()

            latent_state_freenergy_train = calculateLatentFreeEnergy(
                latent_states_flatten, covariance_factor,
                latent_state_free_energy_grid)
            latent_state_freenergy_bootstrap_train.append(latent_state_freenergy_train)
        results["latent_state_freenergy_bootstrap_train"] = latent_state_freenergy_bootstrap_train


    return results


def calculateFreeEnergyProjection(results,
                                  covariance_factor_scale=50.0,
                                  gridpoints=100):

    print("# calculateFreeEnergyProjection() #")

    latent_states_all = results["latent_states_all"]

    if np.shape(latent_states_all)[2] == 1:

        latent_states_flatten = np.array(latent_states_all).flatten()

        latent_states_flatten_range = latent_states_flatten.max(
        ) - latent_states_flatten.min()

        # Free energy profiles projected into the (slowest) latent space coordinate F(ψ1)=−kBT ln(p(ψ1))
        # Plotting F/kbT =  - ln(p(ψ1))
        covariance_factor = latent_states_flatten_range / covariance_factor_scale
        margin = 0.2
        # covariance_factor = .25
        grid_min = latent_states_flatten.min(
        ) - margin * latent_states_flatten_range
        grid_max = latent_states_flatten.max(
        ) + margin * latent_states_flatten_range
        latent_state_free_energy_grid = np.linspace(grid_min, grid_max,
                                                    gridpoints)

        print("np.max(latent_states_flatten) = {:}".format(
            np.max(latent_states_flatten)))
        print("np.min(latent_states_flatten) = {:}".format(
            np.min(latent_states_flatten)))

        latent_state_freenergy = calculateLatentFreeEnergy(
            latent_states_flatten, covariance_factor,
            latent_state_free_energy_grid)

        print("latent_state_free_energy_grid")
        print(latent_state_free_energy_grid)

        results["latent_state_freenergy"] = latent_state_freenergy
        results[
            "latent_state_free_energy_grid"] = latent_state_free_energy_grid
        results["latent_states_flatten_range"] = latent_states_flatten_range
        results["covariance_factor"] = covariance_factor

    else:
        print(np.shape(latent_states_all))
        latent_states_flatten = np.reshape(
            latent_states_all, (-1, np.shape(latent_states_all)[2]))
        print(np.shape(latent_states_flatten))

        if np.shape(latent_states_flatten)[1] > 2:
            print("Perfoming PCA on the latent space...")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)

            pca.fit(latent_states_flatten)
            latent_states_flatten = pca.transform(latent_states_flatten)

        latent_states_flatten_range = np.max(
            latent_states_flatten, axis=0) - np.min(latent_states_flatten,
                                                    axis=0)

        grid_min = np.min(latent_states_flatten,
                          axis=0) - 0.05 * latent_states_flatten_range
        grid_max = np.max(latent_states_flatten,
                          axis=0) + 0.05 * latent_states_flatten_range

        bounds = []
        for i in range(len(grid_min)):
            min_var = grid_min[i]
            max_var = grid_max[i]
            bounds.append([min_var, max_var])

        nbins = gridpoints
        density, grid_centers = utils_statistics.get_density(
            latent_states_flatten, gridpoints, bounds)

        # density_norm = utils_statistics.evaluate2DIntegral(density, grid_centers[0], grid_centers[1])
        # print(density_norm)

        min_value = latent_states_flatten.min()
        max_value = latent_states_flatten.max()
        sigmax = (grid_max[0] - grid_min[0]) / covariance_factor_scale
        sigmay = (grid_max[1] - grid_min[1]) / covariance_factor_scale

        import scipy.ndimage as ndimage
        density = ndimage.gaussian_filter(density,
                                          sigma=(sigmax, sigmay),
                                          order=0)
        density_norm = utils_statistics.evaluate2DIntegral(
            density, grid_centers[0], grid_centers[1])

        latent_state_freenergy = -np.log(density)

        latent_state_free_energy_grid = np.meshgrid(grid_centers[0],
                                                    grid_centers[1])

        print(np.shape(latent_state_freenergy))
        print(np.shape(latent_state_free_energy_grid))

        results["latent_state_freenergy"] = latent_state_freenergy
        results[
            "latent_state_free_energy_grid"] = latent_state_free_energy_grid
        results["latent_states_flatten_range"] = latent_states_flatten_range
        results["latent_states_flatten"] = latent_states_flatten

        results["latent_states_grid_centers"] = grid_centers

    return results


def localMinimaDetection(arr):
    import numpy as np
    import scipy.ndimage.filters as filters
    import scipy.ndimage.morphology as morphology
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
    # apply the local minimum filter; all locations of minimum value
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood) == arr)
    # local_min is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    #
    # we create the mask of the background
    background = (arr == 0)
    #
    # a little technicality: we must erode the background in order to
    # successfully subtract it from local_min, otherwise a line will
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(background,
                                                  structure=neighborhood,
                                                  border_value=1)
    #
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_min mask
    detected_minima = local_min ^ eroded_background

    # print(np.where(detected_minima))
    minima_x, minima_y = np.where(detected_minima)
    # print(minima_x)
    # print(minima_y)

    minima = []
    for i in range(len(minima_x)):
        minima.append([minima_x[i], minima_y[i]])
    return minima
