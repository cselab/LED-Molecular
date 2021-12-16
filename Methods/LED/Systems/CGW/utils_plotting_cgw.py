#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
import socket
import os
from contextlib import redirect_stdout
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


def plot_density_trajectory_CGW(model, results, set_name, testing_mode):
    rho_target = results["density_traj_target"]
    rho_pred = results["density_traj_predict"]
    time = np.arange(len(rho_target)) * 0.1  # in ps
    # PLOT HERE THE RESULT
    fig = plt.figure(1, figsize=(5, 3))
    plt.plot(time, rho_target, 'k-', time, rho_pred, 'b-', linewidth=2.5)
    plt.xlabel('time [ps]', fontsize=18)
    plt.ylabel(r'$\rho$ [g/cm$^3$]', fontsize=18)
    fig.tight_layout()
    fig_path = model.getFigureDir() + "/{:}_density_traj_{:}.{:}".format(
        testing_mode, set_name, FIGTYPE)
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # erminas_result_single_number = results["erminas_result_single_number"]
    # # save to .txt, or ...
    txt_path = model.getFigureDir() + "/{:}_density_traj_{:}.txt".format(
        testing_mode, set_name)
    with open(txt_path, 'w') as outfile:
        with redirect_stdout(outfile):
            print('rho target average = ', np.average(rho_target), ',  max = ',
                  np.max(rho_target), ',  min = ', np.min(rho_target),
                  ' [g/cm^3]')
            print('rho predicted average = ', np.average(rho_pred),
                  ',  max = ', np.max(rho_pred), ',  min = ', np.min(rho_pred),
                  ' [g/cm^3]')


def output_diffusion_coefficient_CGW(model, results, set_name, testing_mode):
    D_target = results["diffusion_coeff_target"]
    D_pred = results["diffusion_coeff_predict"]
    txt_path = model.getFigureDir(
    ) + "/{:}_diffusion_coefficient_{:}.txt".format(testing_mode, set_name)

    with open(txt_path, 'w') as outfile:
        with redirect_stdout(outfile):
            print('target Diffusion coefficient = ', D_target,
                  ' [10^-4 cm^2/s]')
            print('predicted Diffusion coefficient = ', D_pred,
                  ' [10^-4 cm^2/s]')


def plot_com_traj_CGW(model, results, set_name, testing_mode):
    com_traj_target = results["com_target"]
    com_traj_pred = results["com_predict"]
    time = np.arange(com_traj_target.shape[1]) * 0.1  # ps
    Nics = np.shape(com_traj_target)[0]

    # Plotting the COM (x, y, z) for different initial conditions
    for dim in range(3):
        position_str = "x" if dim == 0 else "y" if dim == 1 else "z"
        fig = plt.figure(dim + 1, figsize=(7, 5))

        for ic in np.array([0, int(Nics / 2), Nics - 1]):
            labelT = 'Target' if ic == 0 else None
            labelP = 'Predicted' if ic == 0 else None
            plt.plot(time,
                     com_traj_target[ic, :, dim],
                     '-',
                     color='green',
                     linewidth=2,
                     label=labelT)
            plt.plot(time,
                     com_traj_pred[ic, :, dim],
                     '-.',
                     color='blue',
                     linewidth=2,
                     label=labelP)

        plt.xlabel('time [ps]', fontsize=20)
        plt.ylabel('{:} CoM [A]'.format(position_str), fontsize=20)
        plt.legend(loc='best', fontsize=18)
        fig.tight_layout()
        fig_path = model.getFigureDir() + "/{:}_CoM_traj_{:}_{:}.{:}".format(
            testing_mode, position_str, set_name, FIGTYPE)
        plt.savefig(fig_path, dpi=300)
        plt.close()

    # # plot for ICs
    # plt.plot(time, com_traj_target[0,:,0], '-', color='dimgray', linewidth=2, label=r'x_{T,CoM}, ic = 1')
    # if (com_traj_target.shape[0]>1):
    # 	last=com_traj_target.shape[0]-1
    # 	plt.plot(time, com_traj_target[last,:,0], '-', color='dimgray', linewidth=2, label=r'x_{T,CoM}, ic = '+str(last))

    # print absolute error of CoM in trajectory and in initial conditions
    com_error_abs = results["abs_error_com_target_predict"]
    txt_path = model.getFigureDir() + "/{:}_com_abs_error_{:}.txt".format(
        testing_mode, set_name)
    with open(txt_path, 'w') as outfile:
        with redirect_stdout(outfile):
            print('| x_(CoM, target) - x_(CoM, predict) | = ',
                  com_error_abs[0], ' (A)')
            print('| y_(CoM, target) - y_(CoM, predict) | = ',
                  com_error_abs[1], ' (A)')
            print('| z_(CoM, target) - z_(CoM, predict) | = ',
                  com_error_abs[2], ' (A)')
