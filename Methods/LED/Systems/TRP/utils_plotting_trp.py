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

from ... import Utils as utils
from . import utils_processing_trp

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


def computeLatentDynamicsDistributionErrorTRP(model, set_name):
    print("# computeLatentDynamicsDistributionErrorTRP() #")
    from scipy.integrate import simps

    testing_mode = "teacher_forcing_forecasting"
    data_path = model.getResultsDir() + "/results_{:}_{:}".format(
        testing_mode, set_name)

    try:
        results = utils.loadData(data_path, model.save_format)
    except Exception as inst:
        print(inst)
        print(
            "# Something went wrong during loading of file {:}. Not plotting the latent comparison results."
            .format(data_path))
        return 0

    # Results of teacher forcing the MD data
    tf_latent_states_all = results["latent_states_all"]
    tf_latent_states_all = np.reshape(tf_latent_states_all,
                                      (-1, np.shape(tf_latent_states_all)[2]))
    # tf_latent_states_all                = results["latent_states_flatten"]

    testing_mode = "iterative_latent_forecasting"
    data_path = model.getResultsDir() + "/results_{:}_{:}".format(
        testing_mode, set_name)
    try:
        results = utils.loadData(data_path, model.save_format)
    except Exception as inst:
        print(inst)
        print(
            "# Something went wrong during loading of file {:}. Not plotting the latent comparison results."
            .format(data_path))
        return 0

    # Results of iterative latent forecasting
    lf_latent_states_all = results["latent_states_all"]
    lf_latent_states_all = np.reshape(lf_latent_states_all,
                                      (-1, np.shape(lf_latent_states_all)[2]))
    # lf_latent_states_all                = results["latent_states_flatten"]

    print(np.shape(tf_latent_states_all))
    print(np.shape(lf_latent_states_all))

    bounds = []
    for dim in range(2):
        min_ = np.min([
            np.min(tf_latent_states_all[:, dim]),
            np.min(lf_latent_states_all[:, dim])
        ])
        max_ = np.max([
            np.max(tf_latent_states_all[:, dim]),
            np.max(lf_latent_states_all[:, dim])
        ])
        bounds.append([min_, max_])

    # nbins = 25
    # nbins = 100
    data = {}
    fields_to_write = []
    # for nbins in [10, 25, 50]:
    for nbins in [20]:

        error, error_vec, density_tf, density_lf, bin_centers = utils.evaluateL1HistErrorVector(
            tf_latent_states_all, lf_latent_states_all, nbins, bounds)

        vmin = np.min([np.min(density_tf), np.min(density_lf)])
        vmax = np.max([np.max(density_tf), np.max(density_lf)])

        logfile_comparison = model.getLogFileDir(
        ) + "/results_comparison_{:}.txt".format(set_name)

        error_name = "latent_distribution_error_nbins{:}".format(nbins)

        data.update({
            error_name: error,
        })
        fields_to_write.append(error_name)

        tf_latent_state_freenergy = -np.log(density_tf)
        lf_latent_state_freenergy = -np.log(density_lf)
        err_temp = np.abs(tf_latent_state_freenergy -
                          lf_latent_state_freenergy)
        err_temp[err_temp == -np.inf] = 0.0
        err_temp[err_temp == np.inf] = 0.0
        err_temp = np.nan_to_num(err_temp)
        error_free_energy = simps(
            [simps(zz_x, bin_centers[0]) for zz_x in err_temp], bin_centers[1])
        error_free_energy = np.sqrt(error_free_energy)
        print(
            "FREE ENERGY MEAN SQUARE ROOT ERROR {:}".format(error_free_energy))

        error_name = "latent_free_energy_error_nbins{:}".format(nbins)
        data.update({
            error_name: error_free_energy,
        })
        fields_to_write.append(error_name)

        if model.params["plot_latent_dynamics_comparison_system"]:
            for freenergy in [True, False]:
                # for freenergy in [True]:
                freenergy_str = "_freeenergy" if freenergy else ""
                for cmap in ['Blues_r', 'gist_rainbow']:
                    # for cmap in ['gist_rainbow' ]:
                    # for method_str in ["contourf", "pcolormesh"]:
                    for method_str in ["contourf", "pcolormesh"]:
                        # for method_str in ["contourf"]:

                        fig_path = model.getFigureDir(
                        ) + "/Comparison_latent_dynamics_joint_distr{:}_{:}_nbins{:}_{:}_{:}.{:}".format(
                            freenergy_str, method_str, nbins,
                            "teacher_forcing", cmap, FIGTYPE)
                        makeJointDistributionContourPlot(density_tf,
                                                         bin_centers,
                                                         fig_path,
                                                         cmap=cmap,
                                                         method=method_str,
                                                         vmin=vmin,
                                                         vmax=vmax,
                                                         freenergy=freenergy)

                        fig_path = model.getFigureDir(
                        ) + "/Comparison_latent_dynamics_joint_distr{:}_{:}_nbins{:}_{:}_{:}.{:}".format(
                            freenergy_str, method_str, nbins,
                            "iterative_latent", cmap, FIGTYPE)
                        makeJointDistributionContourPlot(density_lf,
                                                         bin_centers,
                                                         fig_path,
                                                         cmap=cmap,
                                                         method=method_str,
                                                         vmin=vmin,
                                                         vmax=vmax,
                                                         freenergy=freenergy)

    bounds = []
    for dim in range(2):
        min_ = np.min(tf_latent_states_all[:, dim])
        max_ = np.max(tf_latent_states_all[:, dim])
        bounds.append([min_, max_])

    covariance_factor_scalex = 60.0
    covariance_factor_scaley = 30.0
    gridpoints = 20
    testing_mode = "teacher_forcing_forecasting"
    # utils.makejointDistributionPlot(model, testing_mode, set_name, tf_latent_states_all, data_bounds=bounds)
    vmin, vmax, freenergyx_max, freenergyy_max = utils.makeFreeEnergyPlot(
        model,
        testing_mode,
        set_name,
        tf_latent_states_all,
        data_bounds=bounds,
        gridpoints=gridpoints,
        covariance_factor_scalex=covariance_factor_scalex,
        covariance_factor_scaley=covariance_factor_scaley)

    testing_mode = "iterative_latent_forecasting"
    # utils.makejointDistributionPlot(model, testing_mode, set_name, lf_latent_states_all, data_bounds=bounds)
    utils.makeFreeEnergyPlot(model,
                             testing_mode,
                             set_name,
                             lf_latent_states_all,
                             data_bounds=bounds,
                             vmin=vmin,
                             vmax=vmax,
                             freenergyx_max=freenergyx_max,
                             freenergyy_max=freenergyy_max,
                             gridpoints=gridpoints,
                             covariance_factor_scalex=covariance_factor_scalex,
                             covariance_factor_scaley=covariance_factor_scaley)

    if model.write_to_log:
        utils.writeToLogFile(model, logfile_comparison, data, fields_to_write,
                             'w')

    return 0


def makeJointDistributionContourPlot(density,
                                     bin_centers,
                                     fig_path,
                                     method="pcolormesh",
                                     cmap='Blues_r',
                                     vmin=None,
                                     vmax=None,
                                     num_levels=40,
                                     freenergy=False):
    print("# makeJointDistributionPlot() #")
    from matplotlib import gridspec
    from matplotlib.colorbar import Colorbar

    X, Y = np.meshgrid(bin_centers[0], bin_centers[1])
    boundsx = [np.min(X), np.max(X)]
    boundsy = [np.min(Y), np.max(Y)]

    fig = plt.figure(1, figsize=(8, 8))

    axgrid = gridspec.GridSpec(4,
                               2,
                               height_ratios=[0.2, 0.8, 0.1, 0.04],
                               width_ratios=[0.8, 0.2])

    axmain = plt.subplot(axgrid[1, 0])
    axfirst = plt.subplot(axgrid[0, 0])
    axsecond = plt.subplot(axgrid[1, 1])
    axcolorbar = plt.subplot(axgrid[3, 0])

    # axgrid.update(left=0.15, right=0.95, bottom=0.1, top=0.93, wspace=0.2, hspace=0.35)
    axgrid.update(left=0.15,
                  right=0.95,
                  bottom=0.1,
                  top=0.93,
                  wspace=0.05,
                  hspace=0.1)

    if freenergy:
        data_plot = -np.log(density)
        vmin = -np.log(vmin)
        vmax = -np.log(vmax)
        vmin = np.nanmin(data_plot[data_plot != -np.inf])
        vmax = np.nanmax(data_plot[data_plot != +np.inf])
        norm = None
        labelmain = "$F/ \kappa_B T$"

    else:
        data_plot = density
        norm = None
        labelmain = "$p \, ( \mathbf{z}_1, \mathbf{z}_2 )$"

    if method == "pcolormesh":
        mp = axmain.pcolor(
            X,
            Y,
            data_plot.T,
            cmap=plt.get_cmap(cmap),
            vmin=vmin,
            vmax=vmax,
            norm=norm,
        )
    if method == "contourf":
        mp = axmain.contourf(X,
                             Y,
                             data_plot.T,
                             cmap=plt.get_cmap(cmap),
                             levels=np.linspace(vmin, vmax, num_levels),
                             norm=norm)
    axmain.set_xlim(boundsx)
    axmain.set_ylim(boundsy)

    cb = Colorbar(ax=axcolorbar,
                  mappable=mp,
                  orientation='horizontal',
                  ticklocation='bottom',
                  label=labelmain)
    from matplotlib import ticker
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()

    gridx = bin_centers[0]
    densityx = np.sum(density, axis=1)

    axfirst.fill_between(gridx,
                         densityx,
                         np.min(densityx),
                         step="pre",
                         facecolor='blue',
                         alpha=0.5)
    axfirst.plot(gridx, densityx, "blue", linewidth=2, drawstyle="steps")
    # axfirst.step(gridx, densityx, color="blue", linewidth=2)
    axfirst.set_xlim(boundsx)

    gridy = bin_centers[1]
    densityy = np.sum(density, axis=0)

    axsecond.fill_betweenx(gridy,
                           0.0,
                           densityy,
                           step="post",
                           facecolor='blue',
                           alpha=0.5)
    axsecond.plot(densityy,
                  gridy,
                  color='blue',
                  linewidth=2,
                  drawstyle="steps")
    axsecond.set_ylim(boundsy)

    axfirst.spines['right'].set_visible(False)
    axfirst.spines['top'].set_visible(False)

    # axfirst.spines['bottom'].set_visible(False)
    axfirst.get_xaxis().set_visible(False)

    axfirst.set_ylim(ymin=0)
    # axfirst.axis('off')
    ylabel = "$p \, ( \mathbf{z}_1 )$"
    axfirst.set_ylabel(r"{:}".format(ylabel))
    axsecond.spines['right'].set_visible(False)
    axsecond.spines['top'].set_visible(False)

    # axsecond.spines['left'].set_visible(False)
    axsecond.get_yaxis().set_visible(False)

    axsecond.set_xlim(xmin=0)
    # axsecond.axis('off')
    xlabel = "$p \, ( \mathbf{z}_2 )$"
    axsecond.set_xlabel(r"{:}".format(xlabel))
    xlabel = "$\mathbf{z}_1$"
    ylabel = "$\mathbf{z}_2$"
    axmain.set_xlabel(r"{:}".format(xlabel))
    axmain.set_ylabel(r"{:}".format(ylabel))
    plt.savefig(fig_path, dpi=300)
    plt.close()


def generateXYZfromABDtrajectoriesTRP(model, results, set_name, testing_mode):
    print("# generateXYZfromABDtrajectoriesTRP() #")
    print(testing_mode)

    if "autoencoder" in testing_mode:
        trajectories = results["input_sequence_all"]
        latent_states_all = results["latent_states_all"]

    elif "teacher_forcing" in testing_mode:
        trajectories = results["targets_all"][:, :-1]
        latent_states_all = results["latent_states_all"][:, 1:]

    elif "iterative" in testing_mode:
        trajectories = results["predictions_all"][:, :-1]
        latent_states_all = results["latent_states_all"][:, 1:]

    # print(np.shape(trajectories))
    # print(np.shape(latent_states_all))

    latent_state_freenergy = results["latent_state_freenergy"]
    # latent_state_free_energy_grid   = results["latent_state_free_energy_grid"]
    latent_states_grid_centers = results["latent_states_grid_centers"]

    minima_locations = utils.localMinimaDetection(latent_state_freenergy)
    print(minima_locations)
    print(np.shape(minima_locations))
    print("Number of local minima in the latent state = {:}".format(
        len(minima_locations)))

    print(latent_states_grid_centers)
    print(np.shape(latent_states_grid_centers))

    Dx = latent_states_grid_centers[0][2] - latent_states_grid_centers[0][1]
    Dy = latent_states_grid_centers[1][2] - latent_states_grid_centers[1][1]

    # Dx /= 2.0
    # Dy /= 2.0

    # Dx *= 2.0
    # Dy *= 2.0

    # Dx *= 4.0
    # Dy *= 4.0

    # print(Dx)
    # print(Dy)

    minima_regions = []
    for minima_location in minima_locations:
        idx_x, idx_y = minima_location

        grid_x = latent_states_grid_centers[0][idx_x]
        grid_y = latent_states_grid_centers[1][idx_y]

        # define latent region
        region = [[grid_x - Dx, grid_x + Dx], [grid_y - Dy, grid_y + Dy]]
        minima_regions.append(region)

    for reg_num in range(len(minima_regions)):
        print("Latent state region {:}:".format(reg_num))

        minima_region = minima_regions[reg_num]

        minima_region_1 = minima_region[0]
        minima_region_2 = minima_region[1]

        print("Z_1: {:}".format(minima_region_1))
        print("Z_2: {:}".format(minima_region_2))

    ref_conf_file_path = model.params[
        "project_path"] + "/Methods/LED/Systems/TRP/ref.xyz"
    print(
        "# Looking for reference conf. file:\n{:}".format(ref_conf_file_path))

    if not os.path.isfile(ref_conf_file_path):
        print("# Error: reference conf. file:\n{:}\nnot found.".format(
            ref_conf_file_path))

    saving_dir = model.getFigureDir() + "/Trajectories"
    os.makedirs(saving_dir, exist_ok=True)

    print("###")
    print(np.shape(latent_states_all))
    print(np.max(latent_states_all[:, :, 0]))
    print(np.min(latent_states_all[:, :, 0]))

    print(np.max(latent_states_all[:, :, 1]))
    print(np.min(latent_states_all[:, :, 1]))

    num_ics = 2
    for minima_region_num in range(len(minima_regions)):
        print("# minima_region_num = {:}".format(minima_region_num))
        minima_region = minima_regions[minima_region_num]

        minima_region_1 = minima_region[0]
        minima_region_2 = minima_region[1]

        ic_plotted = 0
        # Generate BAD file from targets
        for ic in range(np.shape(trajectories)[0]):
            # print("IC {:}/{:}".format(ic, np.shape(trajectories)[0]))
            traj = trajectories[ic]
            latent_state = latent_states_all[ic]

            # traj = traj[:100]
            # # Write bad file
            # filename_bad = saving_dir + '/bad_file.txt'
            # np.savetxt(filename_bad, traj, fmt="%15.10f")

            # conffile    = saving_dir + '/conf_file.xyz'
            # lammps_file = saving_dir + '/lammps_file.xyz'

            # utils_processing_trp.generateXYZfileFromABDnoRotTr(filename_bad, ref_conf_file_path, conffile, lammps_file)


            idx = np.where(
                           (latent_state[:,0]>=minima_region_1[0]) & \
                           (latent_state[:,0]<=minima_region_1[1]) & \
                           (latent_state[:,1]>=minima_region_2[0]) & \
                           (latent_state[:,1]<=minima_region_2[1]) \
                           )[0]

            # print(idx)

            if len(idx) > 0:
                ic_plotted += 1
                print("# Minimum {:} found in trajectories.".format(
                    minima_region_num))

                cluster_label1 = [
                    "{:.2f}".format(minima_region_1[i])
                    for i in range(len(minima_region_1))
                ]
                cluster_label2 = [
                    "{:.2f}".format(minima_region_2[i])
                    for i in range(len(minima_region_2))
                ]

                formated_str = "{:}_{:}_{:}_{:}_ic{:}".format(
                    testing_mode, set_name, cluster_label1, cluster_label2, ic)

                # print(formated_str)

                index = np.random.choice(idx)

                max_idx = np.min([np.shape(traj)[0], index + 100])
                traj_region = traj[index:max_idx].copy()

                if ic_plotted <= num_ics:

                    # Write bad file
                    filename_bad = saving_dir + '/{:}_bad_file.txt'.format(
                        formated_str)
                    np.savetxt(filename_bad, traj_region, fmt="%15.10f")

                    conffile = saving_dir + '/{:}_conf_file.xyz'.format(
                        formated_str)
                    lammps_file = saving_dir + '/{:}_lammps_file.xyz'.format(
                        formated_str)

                    utils_processing_trp.generateXYZfileFromABDnoRotTr(
                        filename_bad, ref_conf_file_path, conffile,
                        lammps_file)
