#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

from .. import Utils as utils
import numpy as np
import socket
import os

######################################################################################################
## PLOTTING UTILITIES FOR EACH SYSTEM
######################################################################################################
# from .AD import utils_plotting_AD as utils_plotting_AD
# from .KS import utils_plotting_ks as utils_plotting_ks
# from .BMP import utils_plotting_bmp as utils_plotting_bmp
# from .Alanine import utils_plotting_alanine as utils_plotting_alanine
# from .TRP import utils_plotting_trp as utils_plotting_trp
# from .CGW import utils_plotting_cgw as utils_plotting_cgw

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


def getFieldsToCompare(model):
    error_labels = utils.getErrorLabelsDict(model)
    error_labels = error_labels.keys()
    fields_to_compare = [key + "_avg" for key in error_labels]
    fields_to_compare += error_labels
    # print(error_labels_avg)
    # print(ark)
    # fields_to_compare = [
    # "time_total_per_iter",
    # # "rmnse_avg_over_ics",
    # "rmnse_avg",
    # # "num_accurate_pred_050_avg",
    # # "error_freq",
    # ]
    fields_to_compare.append("time_total_per_iter")
    return fields_to_compare


def error2LabelDict():
    dict_ = {
        "rmnse_avg": "RMNSE",
        "mnad_avg_act": "MNAD$(u, \tilde{u})$",
        "mnad_avg_in": "MNAD$(v, \tilde{v})$",
        "mnad_avg": "MNAD",
        "state_dist_L1_hist_error": "L1-NHD",
        "state_dist_wasserstein_distance": "WD",
        "mse_avg": "MSE",
        "rmse_avg": "RMSE",
        "abserror_avg": "ABS",
        "state_dist_L1_hist_error_all": "L1-NHD",
        "state_dist_wasserstein_distance_all": "WD",
        "state_dist_L1_hist_error_avg": "L1-NHD",
        "state_dist_wasserstein_distance_avg": "WD",
        "rmnse_avg_over_ics": "RMNSE",
        "mnad_avg_over_ics_act": "NAD$(u, \tilde{u})$",
        "mnad_avg_over_ics_in": "NAD$(v, \tilde{v})$",
        "mnad_avg_over_ics": "NAD",
    }
    return dict_


def plotMultiscaleResultsComparison(model, dicts_to_compare, set_name,
                                    fields_to_compare, dt):
    print(fields_to_compare)

    # temp = dicts_to_compare["mse"]
    # print(np.shape(temp))

    FIGTYPE = "pdf"
    # if not (model.params["system_name"] in ["KSGP64L22", "FHN"]):
    #   return 0
    # print(dicts_to_compare)
    if model.params["system_name"] == "KSGP64L22":
        lyapunov_time = 20.83
        dt_scaled = dt / lyapunov_time
        vpt_label = "VPT ($\\times \Lambda_1$)"
        # vpt_label = "VPT"
        time_label = "$t /\Lambda_1$"
    else:
        dt_scaled = dt
        vpt_label = "VPT"
        time_label = "$t$"

    error2Label = error2LabelDict()

    for field in fields_to_compare:
        # In which fields should the self-similar error be added ?
        if field in [
                "state_dist_L1_hist_error",
                "state_dist_wasserstein_distance",
                "state_dist_L1_hist_error_avg",
                "state_dist_wasserstein_distance_avg",
        ]:
            self_similar = True
        else:
            self_similar = False

        macro_steps = []
        micro_steps = []
        result = []
        # for key, values in dicts_to_compare.items():
        #     print(key)

        for key, values in dicts_to_compare.items():
            # print(key)
            if "iterative" in key:
                result_iterative = values[field]
            elif "_macro_0" in key:
                # Used to track the time and the self-similar error
                # Only used to track the time !
                result_equations = values[field]

                if self_similar:
                    temp = key.split("_")
                    macro_steps_ = int(float(temp[-1]))
                    micro_steps_ = int(float(temp[-3]))
                    macro_steps.append(macro_steps_)
                    micro_steps.append(micro_steps_)
                    result.append(result_equations)

            elif "multiscale" in key:
                temp = key.split("_")
                macro_steps_ = int(float(temp[-1]))
                micro_steps_ = int(float(temp[-3]))
                result_ = values[field]
                macro_steps.append(macro_steps_)
                micro_steps.append(micro_steps_)
                result.append(result_)
            else:
                raise ValueError(
                    "I don't know how to process {:}.".format(key))

        macro_steps = np.array(macro_steps)
        micro_steps = np.array(micro_steps)
        result = np.array(result)
        micro_steps_set = set(micro_steps)
        for micro_step in micro_steps_set:
            if field in ["num_accurate_pred_050_avg"]:
                # CREATING BAR PLOT
                # print("micro_step\n{:}".format(micro_step))
                indexes = micro_steps == micro_step
                macro_steps_plot = macro_steps[indexes]
                result_plot = result[indexes]

                idx_sort = np.argsort(macro_steps_plot)
                macro_steps_plot = macro_steps_plot[idx_sort]
                result_plot = result_plot[idx_sort]

                rho_plot = [
                    "{:.2f}".format((temp / float(micro_step)))
                    for temp in macro_steps_plot
                ]
                rho_plot.append(str("Latent"))
                result_plot = np.concatenate(
                    (result_plot, result_iterative[np.newaxis]), axis=0)
                result_plot = result_plot * dt_scaled
                # print(rho_plot)
                # print(result_plot)
                barlist = plt.bar(rho_plot, result_plot)
                barlist[-1].set_color("tab:red")

                num_bars = len(barlist[:-1])
                color_labels_bars = color_labels[
                    1:num_bars + 1][::-1]  # Without the red for iterative
                for bn in range(
                        len(barlist[:-1])
                ):  # Without the first bar (self-similar), and the last bar (iterative)
                    barlist[bn].set_color(color_labels_bars[bn])

                plt.ylabel(r"{:}".format(vpt_label))
                plt.xlabel(r"$ \rho=T_m / T_{\mu}$")
                title_str = "{:.2f}".format(micro_step * dt)
                # micro_step_time = micro_step * dt / lyapunov_time
                title_str = "$T_{\mu}=" + title_str
                title_str += ", \, T_{f}=" + "{:.2f}".format(
                    model.prediction_horizon * dt) + "$"
                plt.title(r"{:}".format(title_str), pad=10)
                fig_path = model.getFigureDir(
                ) + "/Comparison_multiscale_macro_micro_{:}_micro{:}_{:}.{:}".format(
                    set_name, int(micro_step), field, FIGTYPE)
                plt.tight_layout()
                plt.savefig(fig_path, dpi=300)
                plt.close()
            elif field in ["time_total_per_iter"]:
                # CREATING BAR PLOT
                # print("micro_step\n{:}".format(micro_step))
                indexes = micro_steps == micro_step
                macro_steps_plot = macro_steps[indexes]
                result_plot = result[indexes]

                idx_sort = np.argsort(macro_steps_plot)
                macro_steps_plot = macro_steps_plot[idx_sort]
                result_plot = result_plot[idx_sort]

                rho_plot = [
                    "{:.2f}".format((temp / float(micro_step)))
                    for temp in macro_steps_plot
                ]
                rho_plot.append(str("Latent"))
                result_plot = list(result_plot) + list([result_iterative])
                barlist = plt.bar(rho_plot, result_plot)
                barlist[-1].set_color("tab:red")

                num_bars = len(barlist[:-1])
                color_labels_bars = color_labels[
                    1:num_bars + 1][::-1]  # Without the red for iterative
                for bn in range(
                        len(barlist[:-1])
                ):  # Without the first bar (self-similar), and the last bar (iterative)
                    barlist[bn].set_color(color_labels_bars[bn])

                plt.ylabel(r"${:}$".format("T_{iter}"))
                plt.xlabel(r"$ \rho=T_m / T_{\mu}$")
                title_str = "{:.2f}".format(micro_step * dt)
                # micro_step_time = micro_step * dt / lyapunov_time
                title_str = "$T_{\mu}=" + title_str
                title_str += ", \, T_{f}=" + "{:.2f}".format(
                    model.prediction_horizon * dt) + "$"
                plt.title(r"{:}".format(title_str), pad=10)
                fig_path = model.getFigureDir(
                ) + "/Comparison_multiscale_macro_micro_{:}_micro{:}_{:}.{:}".format(
                    set_name, int(micro_step), field, FIGTYPE)
                plt.tight_layout()
                plt.savefig(fig_path, dpi=300)
                plt.close()

                field_second = "speed_up"
                # CREATING BAR PLOT
                # print("micro_step\n{:}".format(micro_step))
                # Computing the speed-up
                result_plot = float(result_equations) / np.array(result_plot)
                barlist = plt.bar(rho_plot, result_plot)
                barlist[-1].set_color("tab:red")

                num_bars = len(barlist[:-1])
                color_labels_bars = color_labels[
                    1:num_bars + 1][::-1]  # Without the red for iterative
                for bn in range(
                        len(barlist[:-1])
                ):  # Without the first bar (self-similar), and the last bar (iterative)
                    barlist[bn].set_color(color_labels_bars[bn])

                plt.ylabel(r"{:}".format("Speed-up"))
                plt.xlabel(r"$ \rho=T_m / T_{\mu}$")
                title_str = "{:.2f}".format(micro_step * dt)
                # micro_step_time = micro_step * dt / lyapunov_time
                title_str = "$T_{\mu}=" + title_str
                title_str += ", \, T_{f}=" + "{:.2f}".format(
                    model.prediction_horizon * dt) + "$"
                plt.title(r"{:}".format(title_str), pad=10)
                fig_path = model.getFigureDir(
                ) + "/Comparison_multiscale_macro_micro_{:}_micro{:}_{:}.{:}".format(
                    set_name, int(micro_step), field_second, FIGTYPE)
                plt.tight_layout()
                plt.savefig(fig_path, dpi=300)
                plt.close()

                field_third = "speed_up_log"
                # CREATING BAR PLOT
                # print("micro_step\n{:}".format(micro_step))
                # Computing the speed-up
                result_plot = np.log10(result_plot)
                barlist = plt.bar(rho_plot, result_plot)
                barlist[-1].set_color("tab:red")

                num_bars = len(barlist[:-1])
                color_labels_bars = color_labels[
                    1:num_bars + 1][::-1]  # Without the red for iterative
                for bn in range(
                        len(barlist[:-1])
                ):  # Without the first bar (self-similar), and the last bar (iterative)
                    barlist[bn].set_color(color_labels_bars[bn])

                plt.ylabel(r"{:}".format("$\log_{10}($Speed-up$)$"))
                plt.xlabel(r"$ \rho=T_m / T_{\mu}$")
                title_str = "{:.2f}".format(micro_step * dt)
                # micro_step_time = micro_step * dt / lyapunov_time
                title_str = "$T_{\mu}=" + title_str
                title_str += ", \, T_{f}=" + "{:.2f}".format(
                    model.prediction_horizon * dt) + "$"
                plt.title(r"{:}".format(title_str), pad=10)
                fig_path = model.getFigureDir(
                ) + "/Comparison_multiscale_macro_micro_{:}_micro{:}_{:}.{:}".format(
                    set_name, int(micro_step), field_third, FIGTYPE)
                plt.tight_layout()
                plt.savefig(fig_path, dpi=300)
                plt.close()

            elif field in [
                    "mse_avg",
                    "rmse_avg",
                    "abserror_avg",
                    "rmnse_avg",
                    "mnad_avg",
                    "mnad_avg_act",
                    "mnad_avg_in",
                    "state_dist_L1_hist_error",
                    "state_dist_wasserstein_distance",
                    "state_dist_L1_hist_error_avg",
                    "state_dist_wasserstein_distance_avg",
            ]:
                if (field not in error2Label):
                    raise ValueError(
                        "Field {:} not in error2Label.".format(field))

                indexes = micro_steps == micro_step
                macro_steps_plot = macro_steps[indexes]
                result_plot = result[indexes]

                idx_sort = np.argsort(macro_steps_plot)
                macro_steps_plot = macro_steps_plot[idx_sort]
                result_plot = result_plot[idx_sort]

                macro_steps_plot_float = np.array(
                    [float(temp) for temp in macro_steps_plot])
                rho_plot = macro_steps_plot_float / float(micro_step)

                # plt.plot(rho_plot, result_plot)

                ######## BAR PLOT
                result_plot = np.concatenate(
                    (result_plot, result_iterative[np.newaxis]), axis=0)
                rho_plot = ["{:.2f}".format(temp) for temp in rho_plot]
                rho_plot.append(str("Latent"))

                # Adding the label for the self-similar result
                if self_similar:
                    idx_ = np.where(macro_steps_plot == 0)[0]
                    assert (len(idx_) == 1)
                    idx_ = idx_[0]
                    rho_plot[idx_] = "Self-similar"

                barlist = plt.bar(rho_plot, result_plot)

                # Adding the color for the iterative result
                barlist[-1].set_color("tab:red")

                if self_similar:
                    num_bars = len(barlist[1:-1])
                    color_labels_bars = color_labels[
                        1:num_bars + 1][::-1]  # Without the red for iterative
                    for bn in range(
                            len(barlist[1:-1])
                    ):  # Without the first bar (self-similar), and the last bar (iterative)
                        barlist[bn].set_color(color_labels_bars[bn - 1])
                    barlist[0].set_color("tab:grey")
                    # print(color_labels_bars)
                    # print(arl)
                else:
                    num_bars = len(barlist[:-1])
                    color_labels_bars = color_labels[
                        1:num_bars + 1][::-1]  # Without the red for iterative
                    for bn in range(
                            len(barlist[:-1])
                    ):  # Without the first bar (self-similar), and the last bar (iterative)
                        barlist[bn].set_color(color_labels_bars[bn])

                ylabel = error2Label[field]
                plt.ylabel(r"{:}".format(ylabel))

                plt.xlabel(r"$\rho = T_m / T_{\mu}$")
                # micro_step_time = micro_step * dt / lyapunov_time
                title_str = "$T_{\mu}=" + "{:.2f}".format(micro_step * dt)
                # +"="+"{:.2f}".format(micro_step_time) + "\, \Lambda_1"
                title_str += ", \, T_{f}=" + "{:.2f}".format(
                    model.prediction_horizon * dt) + "$"
                # print(title_str)
                plt.title(r"{:}".format(title_str), pad=10)
                fig_path = model.getFigureDir(
                ) + "/Comparison_multiscale_macro_micro_{:}_micro{:}_{:}.{:}".format(
                    set_name, int(micro_step), field, FIGTYPE)
                plt.tight_layout()
                plt.savefig(fig_path, dpi=300)
                plt.close()

            elif field in [
                    "state_dist_wasserstein_distance_all",
                    "state_dist_L1_hist_error_all",
                    "rmnse_avg_over_ics",
                    "mnad_avg_over_ics",
                    "mnad_avg_over_ics_act",
                    "mnad_avg_over_ics_in",
            ]:

                for with_legend in [True, False]:
                    legend_str = "_legend" if with_legend else ""

                    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

                    # print("micro_step\n{:}".format(micro_step))
                    indexes = micro_steps == micro_step
                    macro_steps_plot = macro_steps[indexes]
                    result_plot = result[indexes]
                    result_iterative = np.array(result_iterative)
                    result_plot = np.array(result_plot)

                    idx_sort = np.argsort(macro_steps_plot)
                    macro_steps_plot = macro_steps_plot[idx_sort]
                    result_plot = result_plot[idx_sort]

                    # result_iterative = result_iterative[:320]
                    # result_plot = result_plot[:,:320]

                    prediction_horizon = len(result_iterative)
                    num_runs, T = np.shape(result_plot)
                    time_vector = np.arange(T) * dt_scaled

                    i_color = 0
                    markevery = int(np.max([int(T / 12), 1]))
                    plt.plot(
                        time_vector,
                        result_iterative,
                        label=r"{:}".format("Iterative Latent Forecasting"),
                        linestyle=linestyles[i_color],
                        marker=linemarkers[i_color],
                        markeredgewidth=linemarkerswidth[i_color],
                        markersize=10,
                        markevery=markevery,
                        color=color_labels[i_color],
                        linewidth=2)

                    color_labels_bars = color_labels[
                        1:num_bars + 1][::-1]  # Without the red for iterative
                    for i in range(num_runs - 1, -1, -1):
                        i_color = i + 1

                        # label = "T_{m}=" + "{:.2f}".format(float(macro_steps_plot[i]) * dt)
                        # if micro_step>0: label = label + ", \, \\rho=" + "{:.2f}".format(float(macro_steps_plot[i]/micro_step))
                        # label = "$" + label + "$"
                        label = "Multiscale Forecasting $T_{\mu}=" + "{:.0f}".format(
                            float(micro_step) *
                            dt) + "$, $T_{m}=" + "{:.0f}".format(
                                float(macro_steps_plot[i]) *
                                dt) + "$" + ", $\\rho=" + "{:.2f}".format(
                                    float(macro_steps_plot[i] /
                                          micro_step)) + "$"

                        plt.plot(time_vector,
                                 result_plot[i],
                                 label=r"{:}".format(label),
                                 linestyle=linestyles[i_color],
                                 marker=linemarkers[i_color],
                                 markeredgewidth=linemarkerswidth[i_color],
                                 markersize=10,
                                 markevery=markevery,
                                 color=color_labels_bars[i],
                                 linewidth=2)

                    plt.xlim([np.min(time_vector), np.max(time_vector)])
                    plt.ylim([
                        np.min(np.array(result_iterative)),
                        1.1 * np.max(np.array(result_iterative))
                    ])

                    ylabel = error2Label[field]
                    plt.ylabel(r"{:}".format(ylabel))

                    plt.xlabel(r"{:}".format(time_label))

                    # title_str = "$T_{\mu}="+"{:.2f}".format(micro_step * dt)
                    # title_str += ", \, T_{f}=" + "{:.2f}".format(prediction_horizon * dt) + "$"

                    plt.title(r"{:}".format(title_str), pad=10)
                    if with_legend:
                        plt.legend(loc="upper left",
                                   bbox_to_anchor=(1.05, 1),
                                   borderaxespad=0.,
                                   frameon=False)
                    plt.tight_layout()
                    fig_path = model.getFigureDir(
                    ) + "/Comparison_multiscale_macro_micro_{:}_micro{:}_{:}{:}.{:}".format(
                        set_name, int(micro_step), field, legend_str, FIGTYPE)
                    plt.savefig(fig_path, dpi=300)
                    # plt.show()
                    plt.close()
